"""DistilBERT (distilbert-base-uncased) forward pass in pure NumPy.

Why NumPy?  The project must run on machines where PyTorch is not installed
(and the build environment used to produce the artifacts had no PyTorch).  The
pretrained weights are read directly from the official `model.safetensors`
file, so the embeddings are *exactly* those of the HuggingFace checkpoint (up to
float32 rounding) - this is inference only, no fine-tuning.

Architecture (Sanh et al., 2019): word + learned position embeddings -> LayerNorm
-> 6 x [multi-head self-attention (12 heads, d=768) -> residual + LayerNorm ->
FFN 768->3072->768 with exact GELU -> residual + LayerNorm].

Sentence embedding = **mean pooling of the last hidden states over valid
(non-padding) tokens**, then L2 normalisation so that a dot product equals the
cosine similarity.

Limitation: DistilBERT was pre-trained with masked-language-modelling, not with a
sentence-similarity objective (unlike Sentence-BERT).  Mean-pooled vectors are
therefore an *unsupervised* semantic representation: they capture topical /
lexical-semantic proximity well but similarity scores are compressed (most pairs
fall in 0.6-1.0) and are not calibrated probabilities.
"""
from __future__ import annotations

import json
import struct
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np

try:  # exact GELU uses erf; scipy ships with scikit-learn
    from scipy.special import erf as _erf
except Exception:  # pragma: no cover - fallback tanh approximation
    _erf = None

from .tokenizer import WordPieceTokenizer

_DTYPES = {"F32": np.float32, "F16": np.float16, "BF16": None, "I64": np.int64}


def load_safetensors(path: str | Path) -> Dict[str, np.ndarray]:
    """Minimal reader for the safetensors format (8-byte header length, JSON header, raw data)."""
    path = Path(path)
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
    raw = np.memmap(path, dtype=np.uint8, mode="r", offset=8 + n)
    tensors = {}
    for name, meta in header.items():
        if name == "__metadata__":
            continue
        start, end = meta["data_offsets"]
        if meta["dtype"] == "BF16":
            u16 = np.frombuffer(raw[start:end], dtype=np.uint16)
            arr = (u16.astype(np.uint32) << 16).view(np.float32)
        else:
            arr = np.frombuffer(raw[start:end], dtype=_DTYPES[meta["dtype"]])
        tensors[name] = np.ascontiguousarray(arr.reshape(meta["shape"]).astype(np.float32))
    return tensors


def _layer_norm(x, g, b, eps=1e-12):
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + eps) * g + b


def _gelu(x):
    if _erf is not None:
        return 0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x ** 3)))


class DistilBertEncoder:
    """Loads distilbert-base-uncased weights and produces mean-pooled sentence embeddings."""

    def __init__(self, model_dir: str | Path, weights_path: Optional[str | Path] = None,
                 max_len: int = 128):
        model_dir = Path(model_dir)
        self.config = json.loads((model_dir / "config.json").read_text())
        self.tokenizer = WordPieceTokenizer(model_dir / "vocab.txt", max_len=max_len)
        self.max_len = max_len
        w = load_safetensors(weights_path or model_dir / "model.safetensors")
        p = "distilbert." if any(k.startswith("distilbert.") for k in w) else ""
        self.word_emb = w[f"{p}embeddings.word_embeddings.weight"]
        self.pos_emb = w[f"{p}embeddings.position_embeddings.weight"]
        self.emb_ln = (w[f"{p}embeddings.LayerNorm.weight"], w[f"{p}embeddings.LayerNorm.bias"])
        self.n_heads = self.config["n_heads"]
        self.dim = self.config["dim"]
        self.layers = []
        for i in range(self.config["n_layers"]):
            q = f"{p}transformer.layer.{i}."
            g = lambda k: w[q + k]
            # pre-transpose + fuse QKV into one matmul for speed
            wqkv = np.concatenate([g("attention.q_lin.weight"), g("attention.k_lin.weight"),
                                   g("attention.v_lin.weight")], 0).T.copy()
            bqkv = np.concatenate([g("attention.q_lin.bias"), g("attention.k_lin.bias"),
                                   g("attention.v_lin.bias")])
            self.layers.append(dict(
                wqkv=wqkv, bqkv=bqkv,
                wo=g("attention.out_lin.weight").T.copy(), bo=g("attention.out_lin.bias"),
                ln1=(g("sa_layer_norm.weight"), g("sa_layer_norm.bias")),
                w1=g("ffn.lin1.weight").T.copy(), b1=g("ffn.lin1.bias"),
                w2=g("ffn.lin2.weight").T.copy(), b2=g("ffn.lin2.bias"),
                ln2=(g("output_layer_norm.weight"), g("output_layer_norm.bias")),
            ))
        # masked-LM head (only used for the sanity check `fill_mask`)
        self.mlm = None
        if "vocab_transform.weight" in w:
            self.mlm = dict(wt=w["vocab_transform.weight"].T.copy(), bt=w["vocab_transform.bias"],
                            ln=(w["vocab_layer_norm.weight"], w["vocab_layer_norm.bias"]),
                            bp=w["vocab_projector.bias"])
        self.n_params = int(sum(v.size for v in w.values()))
        del w

    # ------------------------------------------------------------------
    def hidden_states(self, ids: np.ndarray, mask: np.ndarray) -> np.ndarray:
        B, L = ids.shape
        H, D = self.n_heads, self.dim
        dh = D // H
        x = self.word_emb[ids] + self.pos_emb[:L][None]
        x = _layer_norm(x, *self.emb_ln)
        neg = ((1.0 - mask) * -1e9)[:, None, None, :].astype(np.float32)  # [B,1,1,L]
        for ly in self.layers:
            qkv = x @ ly["wqkv"] + ly["bqkv"]                             # [B,L,3D]
            qkv = qkv.reshape(B, L, 3, H, dh).transpose(2, 0, 3, 1, 4)    # [3,B,H,L,dh]
            qh, kh, vh = qkv[0] / np.sqrt(dh), qkv[1], qkv[2]
            scores = qh @ kh.transpose(0, 1, 3, 2) + neg                  # [B,H,L,L]
            scores -= scores.max(-1, keepdims=True)
            att = np.exp(scores)
            att /= att.sum(-1, keepdims=True)
            ctx = (att @ vh).transpose(0, 2, 1, 3).reshape(B, L, D)
            x = _layer_norm(ctx @ ly["wo"] + ly["bo"] + x, *ly["ln1"])
            h = _gelu(x @ ly["w1"] + ly["b1"])
            x = _layer_norm(h @ ly["w2"] + ly["b2"] + x, *ly["ln2"])
        return x.astype(np.float32)

    @staticmethod
    def mean_pool(hidden: np.ndarray, mask: np.ndarray) -> np.ndarray:
        m = mask[..., None]
        return (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)

    def encode(self, texts: List[str], batch_size: int = 64, normalize: bool = True,
               verbose: bool = False) -> np.ndarray:
        """Embeds texts in length-sorted batches (minimises padding), returns [N,768] float32."""
        texts = [t if isinstance(t, str) and t.strip() else "[UNK]" for t in texts]
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        self.n_truncated = 0
        t0 = time.time()
        for bi, s in enumerate(range(0, len(texts), batch_size)):
            idx = order[s: s + batch_size]
            ids, mask, nt = self.tokenizer.batch_encode([texts[i] for i in idx], self.max_len)
            self.n_truncated += nt
            out[idx] = self.mean_pool(self.hidden_states(ids, mask), mask)
            if verbose and bi % 50 == 0:
                done = min(s + batch_size, len(texts))
                rate = done / max(time.time() - t0, 1e-6)
                print(f"  {done:>6}/{len(texts)} texts  ({rate:.0f} texts/s)", flush=True)
        if normalize:
            out /= np.linalg.norm(out, axis=1, keepdims=True).clip(1e-9)
        return out

    def fill_mask(self, text: str, top_k: int = 5) -> List[str]:
        """Sanity check of the weights: predict the [MASK] token with the MLM head."""
        if self.mlm is None:
            raise RuntimeError("MLM head not present in weights")
        ids = self.tokenizer.encode(text.replace("[MASK]", " [MASK] "))
        # the basic tokenizer splits "[MASK]" into "[", "mask", "]" -> rebuild manually
        toks = []
        for part in text.split("[MASK]"):
            toks.extend(self.tokenizer.tokenize_ids(part)); toks.append(self.tokenizer.mask_id)
        toks = [self.tokenizer.cls_id] + toks[:-1] + [self.tokenizer.sep_id]
        ids = np.array([toks]); mask = np.ones_like(ids, dtype=np.float32)
        h = self.hidden_states(ids, mask)[0]
        pos = toks.index(self.tokenizer.mask_id)
        z = _gelu(h[pos] @ self.mlm["wt"] + self.mlm["bt"])
        z = _layer_norm(z, *self.mlm["ln"])
        logits = z @ self.word_emb.T + self.mlm["bp"]
        return [self.tokenizer.inv_vocab[i] for i in np.argsort(-logits)[:top_k]]
