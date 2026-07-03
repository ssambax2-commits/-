# -*- coding: utf-8 -*-
"""
feedback_excel.py — 피드백용 엑셀 (다음 달 재업로드용, §8-2)
=============================================================
- 사용자 입력 시트(단순): 실제입금여부/실제입금액/입금일자/비고는 빈 값.
- 숨김 시트(모델용): 추천시점 피처벡터, 대출번호(계좌키), 모델버전 등.
"""
from __future__ import annotations

from typing import Dict

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

_HEADER_FILL = PatternFill("solid", fgColor="44546A")
_HEADER_FONT = Font(color="FFFFFF", bold=True)
_INPUT_FILL = PatternFill("solid", fgColor="FFF2CC")  # 사용자 입력칸 연한 강조

USER_COLUMNS = [
    "추천월", "고객번호", "이름", "원금잔액", "다중계좌원금잔액", "부담당자", "팀",
    "등급", "회생", "추천사유", "추천점수",
    "실제입금여부", "실제입금액", "입금일자", "비고",
]
INPUT_COLUMNS = {"실제입금여부", "실제입금액", "입금일자", "비고"}
AMOUNT_COLUMNS = {"원금잔액", "다중계좌원금잔액", "실제입금액"}
SCORE_COLUMNS = {"추천점수"}   # 소수점 유지 표시
TEXT_COLUMNS = {"고객번호"}

HIDDEN_COLUMNS = ["추천월", "고객번호", "성명", "대출번호", "등급", "모델버전", "피처벡터"]


def _autosize(ws, headers, rows):
    for c, h in enumerate(headers, start=1):
        maxlen = sum(2 if ord(ch) > 0x1100 else 1 for ch in str(h))
        for r in rows:
            v = r[c - 1]
            if v is not None and str(v) != "":
                l = sum(2 if ord(ch) > 0x1100 else 1 for ch in str(v))
                maxlen = max(maxlen, l)
        ws.column_dimensions[get_column_letter(c)].width = min(max(maxlen + 2, 8), 60)


def write_feedback_file(path: str, borrowers: pd.DataFrame,
                        feature_json_by_key: Dict[str, str],
                        model_version: str, month: str) -> str:
    """피드백 파일 저장. 반환: 저장 경로."""
    reco = borrowers[borrowers.get("추천여부", True) == True].copy() \
        if "추천여부" in borrowers.columns else borrowers.copy()

    wb = Workbook()
    ws = wb.active
    ws.title = "피드백입력"

    # 헤더
    for c, h in enumerate(USER_COLUMNS, start=1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.fill = _HEADER_FILL
        cell.font = _HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")

    rows = []
    for _, r in reco.iterrows():
        row = [
            month,
            str(r.get("고객번호", "")),
            r.get("성명", ""),
            float(r.get("원금잔액", 0) or 0),
            float(r.get("다중계좌원금잔액", 0) or 0),
            r.get("부담당자", ""),
            r.get("팀", ""),
            r.get("등급", ""),
            r.get("회생", ""),
            r.get("추천사유", ""),
            round(float(r.get("borrower_score", 0) or 0), 2),
            "", "", "", "",  # 사용자 입력칸
        ]
        rows.append(row)

    for ri, row in enumerate(rows, start=2):
        for c, h in enumerate(USER_COLUMNS, start=1):
            cell = ws.cell(row=ri, column=c, value=row[c - 1])
            if h in AMOUNT_COLUMNS:
                cell.number_format = "#,##0"
            if h in SCORE_COLUMNS:
                cell.number_format = "0.00"
            if h in TEXT_COLUMNS:
                cell.number_format = "@"
            if h in INPUT_COLUMNS:
                cell.fill = _INPUT_FILL

    ws.freeze_panes = "A2"
    last_col = get_column_letter(len(USER_COLUMNS))
    ws.auto_filter.ref = f"A1:{last_col}{max(1, len(rows) + 1)}"
    _autosize(ws, USER_COLUMNS, rows)
    # 인쇄 설정: 가로 방향 + 폭 맞춤, 매 페이지 헤더 반복
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_title_rows = "1:1"

    # 숨김 시트(모델용)
    wsh = wb.create_sheet("_모델메타")
    for c, h in enumerate(HIDDEN_COLUMNS, start=1):
        wsh.cell(row=1, column=c, value=h)
    for ri, (_, r) in enumerate(reco.iterrows(), start=2):
        loan_no = str(r.get("대표대출번호", ""))
        vals = [
            month,
            str(r.get("고객번호", "")),
            r.get("성명", ""),
            loan_no,
            r.get("등급", ""),
            model_version,
            feature_json_by_key.get(loan_no, ""),
        ]
        for c, v in enumerate(vals, start=1):
            wsh.cell(row=ri, column=c, value=v)
    wsh.sheet_state = "hidden"

    wb.save(path)
    return path
