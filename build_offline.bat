@echo off
REM ============================================================
REM  코로나채권 AI추천 — 오프라인(폐쇄망) 빌드
REM  사전 준비(외부망 노트북): pip download -r requirements.txt -d wheels
REM  이 스크립트는 wheels\ 폴더의 휠만으로 설치/빌드한다(네트워크 미사용).
REM  산출물: dist\코로나채권AI추천.exe
REM ============================================================
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

if not exist wheels (
    echo [오류] wheels\ 폴더가 없습니다.
    echo         외부망 노트북에서 먼저 아래를 실행해 휠을 받아오세요:
    echo             pip download -r requirements.txt -d wheels
    exit /b 1
)

echo [1/5] 가상환경(venv) 생성...
if not exist venv (
    python -m venv venv
    if errorlevel 1 goto :fail
)
call venv\Scripts\activate.bat

echo [2/5] 오프라인 설치(--no-index --find-links wheels)...
pip install --no-index --find-links wheels -r requirements.txt
if errorlevel 1 goto :fail

echo [3/5] 테스트 실행(실패 시 빌드 중단)...
python -m pytest corona_reco\tests -q
if errorlevel 1 (
    echo [중단] 테스트 실패 — 빌드를 진행하지 않습니다.
    goto :fail
)

echo [4/5] 기존 산출물 정리...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo [5/5] PyInstaller 단일 exe 빌드...
pyinstaller corona_reco.spec --noconfirm
if errorlevel 1 goto :fail

echo.
echo ============================================================
echo  오프라인 빌드 완료:  dist\코로나채권AI추천.exe
echo ============================================================
goto :eof

:fail
echo.
echo [오류] 오프라인 빌드에 실패했습니다.
exit /b 1
