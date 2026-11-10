@echo off
title JobLens - Push to GitHub
cd /d "%~dp0"
set "REPO=https://github.com/ramichatti/AI-Powered-Learning-Graph-Intelligent-Learning-Path-Recommendation-System.git"
set "LOG=%~dp0push_log.txt"
echo Push started %date% %time% > "%LOG%"
echo ============================================================
echo   Pushing this project to GitHub
echo   %REPO%
echo ============================================================

rem --- 1. Find git (install it with winget if missing)
set "GIT="
where git >nul 2>nul && set "GIT=git"
if not defined GIT if exist "%ProgramFiles%\Git\cmd\git.exe" set GIT="%ProgramFiles%\Git\cmd\git.exe"
if not defined GIT (
  echo Git not found - installing Git for Windows with winget ...
  winget install -e --id Git.Git --silent --accept-package-agreements --accept-source-agreements >> "%LOG%" 2>&1
  if exist "%ProgramFiles%\Git\cmd\git.exe" set GIT="%ProgramFiles%\Git\cmd\git.exe"
)
if not defined GIT (
  echo Could not find or install Git. Install it from https://git-scm.com/download/win and run this file again.
  echo GIT NOT FOUND >> "%LOG%"
  pause & exit /b 1
)
%GIT% --version >> "%LOG%" 2>&1

rem --- 2. Repository, identity, remote
if not exist ".git" %GIT% init -b main >> "%LOG%" 2>&1
%GIT% config user.name >nul 2>nul || %GIT% config user.name "ramichatti"
%GIT% config user.email >nul 2>nul || %GIT% config user.email "ramichatti@users.noreply.github.com"
%GIT% remote get-url origin >nul 2>nul && (%GIT% remote set-url origin "%REPO%") || (%GIT% remote add origin "%REPO%")

rem --- 3. Commit everything that .gitignore allows
echo [1/2] Committing files ...
%GIT% add -A >> "%LOG%" 2>&1
%GIT% commit -q -m "JobLens: data-job NLP analytics app (simplified UI, robust Windows launcher)" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>" -m "Claude-Session: https://claude.ai/code/session_017CzgxiNmDetVzVdpKYK7oo" >> "%LOG%" 2>&1
%GIT% branch -M main >> "%LOG%" 2>&1
%GIT% log --oneline -1 >> "%LOG%" 2>&1

rem --- 4. Push (a GitHub sign-in window may open the first time)
echo [2/2] Pushing to GitHub - if a sign-in window opens, log in to GitHub ...
%GIT% push -u origin main >> "%LOG%" 2>&1
if errorlevel 1 (
  echo PUSH FAILED >> "%LOG%"
  echo.
  echo Push failed. Last lines of push_log.txt:
  powershell -NoProfile -Command "Get-Content -Tail 15 '%LOG%'"
  pause & exit /b 1
)
echo PUSH OK >> "%LOG%"
echo.
echo Done! https://github.com/ramichatti/AI-Powered-Learning-Graph-Intelligent-Learning-Path-Recommendation-System
pause
