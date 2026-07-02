@echo off
REM ============================================================
REM  코로나채권 AI추천 — 원클릭 빌드 (외부망 노트북에서 실행)
REM  venv 생성 -> 의존성 설치 -> pytest(실패 시 중단) -> PyInstaller 빌드
REM  산출물: dist\코로나채권AI추천.exe
REM ============================================================
setlocal enabledelayedexpansion
chcp 65001 >nul
cd /d "%~dp0"

echo [1/5] 가상환경(venv) 생성...
if not exist venv (
    python -m venv venv
    if errorlevel 1 goto :fail
)
call venv\Scripts\activate.bat

echo [2/5] 의존성 설치...
python -m pip install --upgrade pip
pip install -r requirements.txt
if errorlevel 1 goto :fail

echo [3/5] 테스트 실행(누수 게이팅 포함, 실패 시 빌드 중단)...
python -m pytest corona_reco\tests -q
if errorlevel 1 (
    echo.
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
echo  빌드 완료:  dist\코로나채권AI추천.exe
echo  이 exe는 실행(추론) 중 외부 네트워크가 전혀 필요 없습니다.
echo ============================================================
goto :eof

:fail
echo.
echo [오류] 빌드에 실패했습니다. 위 로그를 확인하세요.
exit /b 1
