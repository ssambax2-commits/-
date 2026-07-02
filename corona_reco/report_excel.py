# -*- coding: utf-8 -*-
"""
report_excel.py — 실무 추천 보고서 (openpyxl, §8-1)
====================================================
시트: 전체/1팀/2팀 추천리스트, 피드백 작성 안내/요약, (옵션) 상세진단(개발용).
팀별 리스트는 8개 컬럼만 노출. 헤더 고정/색상/AutoFilter, 금액 콤마, 고객번호 텍스트,
S/A 등급 행 정적 강조.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# 표시 컬럼 (borrower 컬럼 -> 헤더)
DISPLAY_MAP = [
    ("고객번호", "고객번호"),
    ("성명", "이름"),
    ("원금잔액", "원금잔액"),
    ("다중계좌원금잔액", "다중계좌원금잔액"),
    ("부담당자", "부담당자"),
    ("등급", "등급"),
    ("회생", "회생"),
    ("추천사유", "추천사유"),
]
AMOUNT_HEADERS = {"원금잔액", "다중계좌원금잔액", "실제입금액", "추천점수"}
TEXT_HEADERS = {"고객번호"}

# 스타일 팔레트 (과하지 않게)
HEADER_FILL = PatternFill("solid", fgColor="44546A")
HEADER_FONT = Font(color="FFFFFF", bold=True)
S_FILL = PatternFill("solid", fgColor="F8CBAD")   # S: 진한 강조
A_FILL = PatternFill("solid", fgColor="FFF2CC")   # A: 연한 강조
BOLD = Font(bold=True)
THIN = Side(style="thin", color="D0D0D0")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)


def _autosize(ws, headers: List[str], rows: List[list]):
    for c, h in enumerate(headers, start=1):
        maxlen = len(str(h))
        for r in rows:
            v = r[c - 1]
            l = len(str(v)) if v is not None else 0
            # 한글은 폭이 넓으므로 가중
            l = sum(2 if ord(ch) > 0x1100 else 1 for ch in str(v)) if v is not None else 0
            maxlen = max(maxlen, l)
        ws.column_dimensions[get_column_letter(c)].width = min(max(maxlen + 2, 8), 60)


def _write_grid(ws, headers: List[str], rows: List[list], grades: Optional[List[str]] = None):
    # 헤더
    for c, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = BORDER
    # 데이터
    for ri, row in enumerate(rows, start=2):
        grade = grades[ri - 2] if grades is not None else None
        for c, h in enumerate(headers, start=1):
            val = row[c - 1]
            cell = ws.cell(row=ri, column=c, value=val)
            cell.border = BORDER
            if h in AMOUNT_HEADERS:
                cell.number_format = "#,##0"
                cell.alignment = Alignment(horizontal="right")
            if h in TEXT_HEADERS:
                cell.number_format = "@"
            # 등급 행 강조
            if grade == "S":
                cell.fill = S_FILL
                cell.font = BOLD
            elif grade == "A":
                cell.fill = A_FILL
                cell.font = BOLD
    # 고정/필터
    ws.freeze_panes = "A2"
    last_col = get_column_letter(len(headers))
    ws.auto_filter.ref = f"A1:{last_col}{max(1, len(rows) + 1)}"
    _autosize(ws, headers, rows)


def _display_rows(borrowers: pd.DataFrame):
    headers = [h for _, h in DISPLAY_MAP]
    rows = []
    grades = []
    for _, r in borrowers.iterrows():
        row = []
        for src, _h in DISPLAY_MAP:
            v = r.get(src, "")
            if src in ("원금잔액", "다중계좌원금잔액"):
                v = float(v) if pd.notna(v) else 0
            elif src == "고객번호":
                v = str(v)
            row.append(v)
        rows.append(row)
        grades.append(str(r.get("등급", "")))
    return headers, rows, grades


def write_report(path: str, borrowers: pd.DataFrame,
                 diagnostics: Optional[Dict] = None,
                 detail_df: Optional[pd.DataFrame] = None,
                 excluded_df: Optional[pd.DataFrame] = None,
                 include_detail: bool = False,
                 month: str = "") -> str:
    """추천 보고서 엑셀 저장. 반환: 저장 경로."""
    diagnostics = diagnostics or {}
    wb = Workbook()

    reco = borrowers[borrowers.get("추천여부", True) == True].copy() \
        if "추천여부" in borrowers.columns else borrowers.copy()

    # 1) 전체 추천리스트
    ws = wb.active
    ws.title = "전체 추천리스트"
    h, rows, grades = _display_rows(reco)
    _write_grid(ws, h, rows, grades)

    # 2) 1팀 / 2팀
    for team in ("1팀", "2팀"):
        ws_t = wb.create_sheet(f"{team} 추천리스트")
        sub = reco[reco["팀"] == team] if "팀" in reco.columns else reco.iloc[0:0]
        h, rows, grades = _display_rows(sub)
        _write_grid(ws_t, h, rows, grades)

    # 3) 피드백 작성 안내/요약
    _write_guide_sheet(wb, reco, borrowers, diagnostics, excluded_df, month)

    # 4) (옵션) 상세진단(개발용)
    if include_detail and detail_df is not None:
        _write_detail_sheet(wb, detail_df, diagnostics)

    wb.save(path)
    return path


def _write_guide_sheet(wb, reco, borrowers, diagnostics, excluded_df, month):
    ws = wb.create_sheet("피드백 작성 안내·요약")
    lines = [
        ["■ 피드백 작성 안내"],
        ["1) 함께 생성된 '피드백_*.xlsx' 파일을 여세요."],
        ["2) 다음 달에 각 추천 건의 실제입금여부/실제입금액/입금일자를 채워주세요."],
        ["3) 실제입금여부: Y/예/입금/성공 = 입금, N/아니오/미입금 = 미입금."],
        ["4) 다음 달 실행 시 활동데이터와 함께 이 피드백 파일을 업로드하면 자동 학습됩니다."],
        ["   (학습데이터 재업로드는 필요 없습니다.)"],
        [""],
        ["■ 이번 달 요약", f"추천월: {month}"],
    ]
    # 등급별/팀별 요약
    if len(borrowers) > 0 and "등급" in borrowers.columns:
        gcount = borrowers[borrowers.get("추천여부", True) == True]["등급"].value_counts()
        lines.append(["등급별 추천 인원"])
        for g in ["S", "A", "B", "C", "D"]:
            lines.append([f"  {g}등급", int(gcount.get(g, 0))])
        if "팀" in reco.columns:
            lines.append(["팀별 추천 인원"])
            tcount = reco["팀"].value_counts()
            for t, c in tcount.items():
                lines.append([f"  {t}", int(c)])
    lines.append([""])
    lines.append(["■ 진단"])
    if excluded_df is not None:
        lines.append(["제외 계좌 수", int(len(excluded_df))])
    for k, v in diagnostics.items():
        if isinstance(v, (str, int, float)):
            lines.append([str(k), v])

    for ri, row in enumerate(lines, start=1):
        for ci, val in enumerate(row, start=1):
            cell = ws.cell(row=ri, column=ci, value=val)
            if ri == 1 or (len(row) == 1 and str(row[0]).startswith("■")):
                cell.font = BOLD
    ws.column_dimensions["A"].width = 40
    ws.column_dimensions["B"].width = 30


def _write_detail_sheet(wb, detail_df: pd.DataFrame, diagnostics: Dict):
    ws = wb.create_sheet("상세진단(개발용)")
    headers = list(detail_df.columns)
    rows = detail_df.astype(object).where(pd.notna(detail_df), "").values.tolist()
    _write_grid(ws, headers, rows, grades=None)
