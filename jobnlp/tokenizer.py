"""Pure-Python WordPiece tokenizer compatible with `distilbert-base-uncased`.

Re-implements the two stages of the HuggingFace `BertTokenizer`:
1. BasicTokenizer: lower-casing, accent stripping, whitespace + punctuation split,
   CJK character isolation.
2. WordpieceTokenizer: greedy longest-match-first sub-word splitting with the
   `##` continuation prefix.

Having it in plain Python means the project needs no `transformers`/`tokenizers`
install to generate embeddings; when `transformers` *is* installed the notebook
cross-checks the token ids against the official tokenizer.
"""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Dict, List

import numpy as np


def _is_whitespace(ch: str) -> bool:
    if ch in (" ", "\t", "\n", "\r"):
        return True
    return unicodedata.category(ch) == "Zs"


def _is_control(ch: str) -> bool:
    if ch in ("\t", "\n", "\r"):
        return False
    return unicodedata.category(ch).startswith("C")


def _is_punctuation(ch: str) -> bool:
    cp = ord(ch)
    if (33 <= cp <= 47) or (58 <= cp <= 64) or (91 <= cp <= 96) or (123 <= cp <= 126):
        return True
    return unicodedata.category(ch).startswith("P")


def _is_cjk(cp: int) -> bool:
    return (
        0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF or 0x20000 <= cp <= 0x2A6DF
        or 0x2A700 <= cp <= 0x2B73F or 0x2B740 <= cp <= 0x2B81F or 0x2B820 <= cp <= 0x2CEAF
        or 0xF900 <= cp <= 0xFAFF or 0x2F800 <= cp <= 0x2FA1F
    )


class WordPieceTokenizer:
    def __init__(self, vocab_file: str | Path, max_len: int = 128):
        self.vocab: Dict[str, int] = {}
        with open(vocab_file, encoding="utf-8") as fh:
            for i, line in enumerate(fh):
                self.vocab[line.rstrip("\n")] = i
        self.inv_vocab = {i: t for t, i in self.vocab.items()}
        self.max_len = max_len
        self.cls_id = self.vocab["[CLS]"]
        self.sep_id = self.vocab["[SEP]"]
        self.pad_id = self.vocab["[PAD]"]
        self.unk_id = self.vocab["[UNK]"]
        self.mask_id = self.vocab["[MASK]"]
        self._cache: Dict[str, List[int]] = {}

    # ---------- basic tokenizer ----------
    def _clean(self, text: str) -> str:
        out = []
        for ch in text:
            cp = ord(ch)
            if cp == 0 or cp == 0xFFFD or _is_control(ch):
                continue
            if _is_whitespace(ch):
                out.append(" ")
            elif _is_cjk(cp):
                out.append(f" {ch} ")
            else:
                out.append(ch)
        return "".join(out)

    @staticmethod
    def _strip_accents(text: str) -> str:
        text = unicodedata.normalize("NFD", text)
        return "".join(ch for ch in text if unicodedata.category(ch) != "Mn")

    def basic_tokenize(self, text: str) -> List[str]:
        text = self._clean(text).lower()
        tokens = []
        for word in text.split():
            word = self._strip_accents(word)
            cur = []
            for ch in word:
                if _is_punctuation(ch):
                    if cur:
                        tokens.append("".join(cur)); cur = []
                    tokens.append(ch)
                else:
                    cur.append(ch)
            if cur:
                tokens.append("".join(cur))
        return tokens

    # ---------- wordpiece ----------
    def _wordpiece(self, word: str) -> List[int]:
        if word in self._cache:
            return self._cache[word]
        if len(word) > 100:
            return [self.unk_id]
        ids, start = [], 0
        while start < len(word):
            end, cur = len(word), None
            while start < end:
                sub = word[start:end]
                if start > 0:
                    sub = "##" + sub
                if sub in self.vocab:
                    cur = self.vocab[sub]
                    break
                end -= 1
            if cur is None:
                ids = [self.unk_id]
                break
            ids.append(cur)
            start = end
        if len(self._cache) < 200_000:
            self._cache[word] = ids
        return ids

    def tokenize_ids(self, text: str) -> List[int]:
        ids: List[int] = []
        for w in self.basic_tokenize(text):
            ids.extend(self._wordpiece(w))
        return ids

    def tokenize(self, text: str) -> List[str]:
        return [self.inv_vocab[i] for i in self.tokenize_ids(text)]

    def encode(self, text: str, max_len: int | None = None) -> List[int]:
        """[CLS] tokens [SEP], truncated to max_len (head truncation)."""
        max_len = max_len or self.max_len
        ids = self.tokenize_ids(text)[: max_len - 2]
        return [self.cls_id] + ids + [self.sep_id]

    def batch_encode(self, texts: List[str], max_len: int | None = None):
        """Dynamic padding: pads to the longest sequence in the batch.
        Returns (input_ids [B,L] int64, attention_mask [B,L] float32, n_truncated)."""
        max_len = max_len or self.max_len
        encoded, n_trunc = [], 0
        for t in texts:
            raw = self.tokenize_ids(t)
            if len(raw) > max_len - 2:
                n_trunc += 1
            encoded.append([self.cls_id] + raw[: max_len - 2] + [self.sep_id])
        L = max(len(e) for e in encoded)
        ids = np.full((len(encoded), L), self.pad_id, dtype=np.int64)
        mask = np.zeros((len(encoded), L), dtype=np.float32)
        for i, e in enumerate(encoded):
            ids[i, : len(e)] = e
            mask[i, : len(e)] = 1.0
        return ids, mask, n_trunc
