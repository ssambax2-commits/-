# -*- coding: utf-8 -*-
"""
run_app.py — PyInstaller 진입 스크립트
=======================================
패키지 corona_reco 의 tkinter GUI를 실행한다. .exe 빌드는 이 파일을 진입점으로 한다.
"""
from corona_reco.main import main

if __name__ == "__main__":
    main()
