# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller 스펙 — 단일 exe(--onefile --windowed)
==================================================
CatBoost/numpy/pandas 등 hidden import·데이터 파일 누락을 방지한다.
빌드: pyinstaller corona_reco.spec
"""
from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []

# CatBoost: 데이터 파일 + 동적 라이브러리 + 서브모듈 전량 수집
for pkg in ("catboost",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# 자주 누락되는 의존성 보강
hiddenimports += collect_submodules("openpyxl")
hiddenimports += [
    "pandas", "numpy", "joblib",
    "corona_reco", "corona_reco.pipeline", "corona_reco.model",
    "corona_reco.features", "corona_reco.scoring", "corona_reco.grading",
    "corona_reco.feedback", "corona_reco.store", "corona_reco.report_excel",
    "corona_reco.feedback_excel", "corona_reco.reasons", "corona_reco.payments",
    "corona_reco.exclusions", "corona_reco.io_loader", "corona_reco.util",
    "corona_reco.config",
]

block_cipher = None

a = Analysis(
    ["run_app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["matplotlib", "IPython", "notebook", "PyQt5", "PySide2"],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="코로나채권AI추천",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,          # --windowed
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
