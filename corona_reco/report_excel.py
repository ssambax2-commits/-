# -*- coding: utf-8 -*-
"""
report_excel.py — 실무 추천 보고서 (openpyxl, §5 전면 개선판)
==============================================================
시트: 표지 / 1팀·2팀(·기타) 추천리스트(계좌 단위) / 팀별 요약 / 제외채권 /
법조치추천 / 화해·정상·약속자 관리현황 / (옵션) 상세진단(개발용).
스타일: S 진한·A 연한 강조, 금액 콤마(원 단위), 필터·틀고정·너비 자동,
가로 인쇄 폭 맞춤. 불필요한 장식 없음.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from . import config

# ---- 스타일 ----
HEADER_FILL = PatternFill("solid", fgColor="44546A")
HEADER_FONT = Font(color="FFFFFF", bold=True)
S_FILL = PatternFill("solid", fgColor="F8CBAD")     # 즉시관리 진한 강조
A_FILL = PatternFill("solid", fgColor="FFF2CC")     # 당월관리 연한 강조
OVER_FILL = PatternFill("solid", fgColor="DDEBF7")  # 정원초과 고스코어 편입(파랑 배지)
BOLD = Font(bold=True)
TITLE_FONT = Font(bold=True, size=15)
THIN = Side(style="thin", color="D0D0D0")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

AMOUNT_HEADERS = {"원금잔액", "동일차주계좌합산원금잔액", "최근입금액",
                  "1개월예상회수액", "3개월예상회수액", "피드백입금액",
                  "원금잔액합", "S+A예상회수1M", "S+A예상회수3M",
                  "예상회수1M", "예상회수3M", "실제입금액"}
SCORE_HEADERS = {"모델점수", "차주점수"}
PCT_HEADERS = {"누적변제율", "최근입금의존도"}
TEXT_HEADERS = {"고객번호", "대출번호"}

# 팀 추천리스트 표시 컬럼(§5 필수 + 추천/참고 분리 컬럼)
TEAM_LIST_COLUMNS = [
    "팀", "부담당자", "고객번호", "대출번호", "성명", "채권구분", "대분류",
    "중분류", "상품명", "채권상태", "원금잔액", "동일차주계좌합산원금잔액",
    "동일차주계좌수", "최근입금일", "최근입금액", "누적변제율", "모델점수",
    "원등급", "최종등급", "추천순위_차주기준", "추천순위_계좌기준",
    "1개월예상회수액", "3개월예상회수액", "추천사유", "등급하향사유",
    "정원초과편입여부", "상한적용여부", "상한하향사유", "추천여부",
    "피드백입금일", "피드백입금액", "피드백결과",
]


def _autosize(ws, headers, rows, max_w=46):
    for c, h in enumerate(headers, start=1):
        w = sum(2 if ord(ch) > 0x1100 else 1 for ch in str(h))
        for r in rows[:200]:
            v = r[c - 1]
            if v is not None and str(v) != "":
                w = max(w, sum(2 if ord(ch) > 0x1100 else 1 for ch in str(v)[:max_w]))
        ws.column_dimensions[get_column_letter(c)].width = min(max(w + 2, 8), max_w)


def _print_setup(ws):
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth = 1
    ws.page_setup.fitToHeight = 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.print_title_rows = "1:1"


def _write_grid(ws, headers: List[str], rows: List[list],
                grades: Optional[List[str]] = None,
                ref_flags: Optional[List[bool]] = None):
    for c, h in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=c, value=h)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = BORDER
    for ri, row in enumerate(rows, start=2):
        grade = grades[ri - 2] if grades is not None else None
        is_ref = ref_flags[ri - 2] if ref_flags is not None else False
        for c, h in enumerate(headers, start=1):
            cell = ws.cell(row=ri, column=c, value=row[c - 1])
            cell.border = BORDER
            if h in AMOUNT_HEADERS:
                cell.number_format = "#,##0"
                cell.alignment = Alignment(horizontal="right")
            elif h in SCORE_HEADERS:
                cell.number_format = "0.00"
            elif h in PCT_HEADERS:
                cell.number_format = "0.0%"
            if h in TEXT_HEADERS:
                cell.number_format = "@"
            if grade == config.TIER_IMMEDIATE:
                cell.fill = S_FILL
                cell.font = BOLD
            elif is_ref:      # 정원초과 고스코어 편입
                cell.fill = OVER_FILL
                cell.font = BOLD
            elif grade == config.TIER_MONTH:
                cell.fill = A_FILL
    ws.freeze_panes = "A2"
    if rows:
        ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{len(rows) + 1}"
    _autosize(ws, headers, rows)
    _print_setup(ws)


def _df_rows(df: pd.DataFrame, columns: List[str]):
    rows = []
    for _, r in df.iterrows():
        row = []
        for c in columns:
            v = r.get(c, "")
            if isinstance(v, float) and pd.isna(v):
                v = ""
            if isinstance(v, bool):
                v = "Y" if v else ""
            row.append(v)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
def write_report(path: str, month: str, run_date: str,
                 input_files: Dict[str, str],
                 accounts: pd.DataFrame, borrowers: pd.DataFrame,
                 excluded_compliance: pd.DataFrame,
                 excluded_business: pd.DataFrame,
                 managed: pd.DataFrame, legal: pd.DataFrame,
                 diagnostics: Dict, include_detail: bool = False,
                 excluded_small: pd.DataFrame = None) -> str:
    wb = Workbook()

    listed = accounts[accounts["추천여부"]].copy() if len(accounts) else accounts

    # 1) 표지
    _write_cover(wb, month, run_date, input_files, accounts, borrowers,
                 excluded_compliance, excluded_business, listed, diagnostics)

    # 2~3) 팀 추천리스트(계좌 단위)
    teams = ["1팀", "2팀"] + (["기타"] if len(listed) and (listed["팀"] == "기타").any() else [])
    for team in teams:
        ws = wb.create_sheet(f"{team} 추천리스트")
        sub = listed[listed["팀"] == team].copy() if len(listed) else listed
        if len(sub):
            sub = sub.sort_values(["부담당자", "추천순위_차주기준", "추천순위_계좌기준"],
                                  na_position="last")
        rows = _df_rows(sub, TEAM_LIST_COLUMNS)
        grades = list(sub["최종등급"]) if len(sub) else []
        refs = list(sub["정원초과편입여부"]) if len(sub) else []
        _write_grid(ws, TEAM_LIST_COLUMNS, rows, grades, refs)

    # 4) 팀별 요약
    _write_team_summary(wb, accounts, borrowers)

    # 5) 제외채권 (컴플라이언스 + 업무 + 소액 차주합산 500만↓)
    _write_excluded(wb, excluded_compliance, excluded_business, excluded_small)

    # 6) 법조치추천
    ws_l = wb.create_sheet("법조치추천(참고)")
    if legal is not None and len(legal):
        cols = list(legal.columns)
        _write_grid(ws_l, cols, _df_rows(legal, cols))
    else:
        ws_l["A1"] = "법조치 검토 후보 없음"

    # 7) 화해·정상·약속자 관리현황
    ws_m = wb.create_sheet("화해·정상·약속자 관리현황")
    if managed is not None and len(managed):
        cols = list(managed.columns)
        _write_grid(ws_m, cols, _df_rows(managed, cols))
        # 상단 요약 별도 행 추가 대신 표지에 합계 표기
    else:
        ws_m["A1"] = "해당 상태 계좌 없음"

    # 8) 진단(항상 포함 — 시간축/모델/편향 투명성)
    _write_diagnostics(wb, diagnostics)

    # 9) (옵션) 차주 상세
    if include_detail and len(borrowers):
        ws_d = wb.create_sheet("차주상세(개발용)")
        cols = [c for c in borrowers.columns if not c.startswith("_")]
        _write_grid(ws_d, cols, _df_rows(borrowers, cols))

    wb.save(path)
    return path


# ---------------------------------------------------------------------------
def _write_cover(wb, month, run_date, input_files, accounts, borrowers,
                 exc_comp, exc_biz, listed, diagnostics):
    ws = wb.active
    ws.title = "표지"
    ws["B2"] = "코로나채권 우선관리 추천 보고서"
    ws["B2"].font = TITLE_FONT
    ws["B3"] = f"기준월 {month} · 실행일자 {run_date}"

    meta = [("입력파일(활동데이터)", input_files.get("활동데이터", "-")),
            ("입력파일(학습데이터)", input_files.get("학습데이터", "-")),
            ("입력파일(전월 피드백)", input_files.get("전월 피드백", "-"))]
    r = 5
    for k, v in meta:
        ws.cell(row=r, column=2, value=k).font = BOLD
        ws.cell(row=r, column=3, value=v)
        r += 1

    n_total = int(diagnostics.get("활동데이터 행수", 0))
    n_exc = int(len(exc_comp)) + int(len(exc_biz))
    n_reco_b = int(borrowers["추천여부"].sum()) if len(borrowers) else 0
    n_imm = int((borrowers["최종등급"] == config.TIER_IMMEDIATE).sum()) if len(borrowers) else 0
    n_mon = int((borrowers["최종등급"] == config.TIER_MONTH).sum()) if len(borrowers) else 0
    n_over = int(borrowers["정원초과편입여부"].sum()) if len(borrowers) else 0
    e1 = float(listed["1개월예상회수액"].sum()) if len(listed) else 0
    e3 = float(listed["3개월예상회수액"].sum()) if len(listed) else 0

    key = [("전체 대상건수(계좌)", n_total),
           ("제외건수(컴플라이언스+업무)", n_exc),
           ("채점 대상 계좌수", diagnostics.get("채점 대상 계좌 수", 0)),
           ("추천대상 차주수(즉시+당월)", n_reco_b),
           ("즉시관리 차주수", n_imm),
           ("당월관리 차주수", n_mon),
           ("  ↳ 정원초과 고스코어 편입", n_over),
           ("1개월 예상회수액(추천분)", round(e1)),
           ("3개월 예상회수액(추천분)", round(e3))]
    r += 1
    ws.cell(row=r, column=2, value="■ 핵심 요약").font = BOLD
    r += 1
    for k, v in key:
        ws.cell(row=r, column=2, value=k)
        c = ws.cell(row=r, column=3, value=v)
        if "예상회수" in k or "잔액" in k:
            c.number_format = "#,##0"
        r += 1

    # 팀별 요약 피벗(미니)
    r += 1
    ws.cell(row=r, column=2, value="■ 팀별 요약").font = BOLD
    r += 1
    headers = ["팀", "계좌수", "차주수", "원금잔액합", "즉시", "당월", "보류",
               "추천예상회수1M", "추천예상회수3M"]
    for c, h in enumerate(headers, start=2):
        cell = ws.cell(row=r, column=c, value=h)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.border = BORDER
    r += 1
    if len(borrowers):
        for team in ["1팀", "2팀", "기타"]:
            tb = borrowers[borrowers["팀"] == team]
            ta = accounts[accounts["팀"] == team] if len(accounts) else accounts
            if len(tb) == 0 and (ta is None or len(ta) == 0):
                continue
            reco_mask = tb["추천여부"] == True
            vals = [team, int(len(ta)) if ta is not None else 0, int(len(tb)),
                    float(ta["원금잔액"].sum()) if ta is not None and len(ta) else 0,
                    int((tb["최종등급"] == config.TIER_IMMEDIATE).sum()),
                    int((tb["최종등급"] == config.TIER_MONTH).sum()),
                    int((tb["최종등급"] == config.TIER_HOLD).sum()),
                    float(tb.loc[reco_mask, "예상회수1M"].sum()),
                    float(tb.loc[reco_mask, "예상회수3M"].sum())]
            for c, v in enumerate(vals, start=2):
                cell = ws.cell(row=r, column=c, value=v)
                cell.border = BORDER
                if headers[c - 2] in AMOUNT_HEADERS or "예상회수" in headers[c - 2]:
                    cell.number_format = "#,##0"
            r += 1

    for col, w in (("B", 30), ("C", 34), ("D", 14), ("E", 12), ("F", 8),
                   ("G", 8), ("H", 8), ("I", 18), ("J", 18)):
        ws.column_dimensions[col].width = w


def _write_team_summary(wb, accounts, borrowers):
    ws = wb.create_sheet("팀별 요약")
    if len(borrowers) == 0:
        ws["A1"] = "데이터 없음"
        return
    # 블록1: 팀 단위(차주 기준)
    h1 = ["팀", "계좌수", "차주수", "원금잔액합", "즉시", "당월", "보류",
          "정원초과편입", "추천예상회수1M", "추천예상회수3M"]
    rows1 = []
    for team in ["1팀", "2팀", "기타"]:
        tb = borrowers[borrowers["팀"] == team]
        ta = accounts[accounts["팀"] == team] if len(accounts) else None
        if len(tb) == 0 and (ta is None or len(ta) == 0):
            continue
        reco = tb["추천여부"] == True
        rows1.append([team,
                      int(len(ta)) if ta is not None else 0,
                      int(len(tb)),
                      float(ta["원금잔액"].sum()) if ta is not None and len(ta) else 0,
                      int((tb["최종등급"] == config.TIER_IMMEDIATE).sum()),
                      int((tb["최종등급"] == config.TIER_MONTH).sum()),
                      int((tb["최종등급"] == config.TIER_HOLD).sum()),
                      int(tb["정원초과편입여부"].sum()),
                      float(tb.loc[reco, "예상회수1M"].sum()),
                      float(tb.loc[reco, "예상회수3M"].sum())])
    _write_grid(ws, h1, rows1)

    # 블록2: 부담당자 단위 — 아래쪽에 이어서
    start = len(rows1) + 3
    ws.cell(row=start, column=1, value="■ 부담당자별 (정상≤150 / 초과편입151~200 / 보류)").font = BOLD
    h2 = ["팀", "부담당자", "계좌수", "차주수", "원금잔액합",
          "즉시", "당월", "정원초과편입", "보류", "추천예상회수1M", "추천예상회수3M"]
    r = start + 1
    for c, h in enumerate(h2, start=1):
        cell = ws.cell(row=r, column=c, value=h)
        cell.fill = HEADER_FILL
        cell.font = HEADER_FONT
        cell.border = BORDER
    r += 1
    for (team, mgr), tb in borrowers.groupby(["팀", "부담당자"], sort=True):
        ta = accounts[accounts["부담당자"] == mgr] if len(accounts) else None
        reco = tb["추천여부"] == True
        vals = [team, mgr,
                int(len(ta)) if ta is not None else 0,
                int(len(tb)),
                float(ta["원금잔액"].sum()) if ta is not None and len(ta) else 0,
                int((tb["최종등급"] == config.TIER_IMMEDIATE).sum()),
                int((tb["최종등급"] == config.TIER_MONTH).sum()),
                int(tb["정원초과편입여부"].sum()),
                int((tb["최종등급"] == config.TIER_HOLD).sum()),
                float(tb.loc[reco, "예상회수1M"].sum()),
                float(tb.loc[reco, "예상회수3M"].sum())]
        for c, v in enumerate(vals, start=1):
            cell = ws.cell(row=r, column=c, value=v)
            cell.border = BORDER
            if h2[c - 1] in ("원금잔액합",) or "예상회수" in h2[c - 1]:
                cell.number_format = "#,##0"
        r += 1


def _write_excluded(wb, exc_comp, exc_biz, exc_small=None):
    ws = wb.create_sheet("제외채권")
    cols = ["구분", "제외사유", "고객번호", "대출번호", "성명", "부담당자",
            "원금잔액", "채권상태(중)", "회생"]
    rows = []

    def _row(src, kind, reason, principal_key="현재원금"):
        return [kind, reason,
                str(src.get("고객번호", "")), str(src.get("대출번호", "")),
                str(src.get("성명", "")), str(src.get("부담당자", "")),
                src.get(principal_key, ""), str(src.get("채권상태(중)", "")),
                str(src.get("회생", ""))]

    if exc_small is not None and len(exc_small):
        for _, r in exc_small.iterrows():
            rows.append(_row(r, "소액 제외(차주합산)",
                             "차주합산 원금잔액 500만원 이하", "동일차주계좌합산원금잔액"))
    if exc_biz is not None and len(exc_biz):
        for _, r in exc_biz.iterrows():
            rows.append(_row(r, "업무 제외", str(r.get("_업무제외사유", ""))))
    if exc_comp is not None and len(exc_comp):
        for _, r in exc_comp.iterrows():
            rows.append(_row(r, "기타 제외사유(컴플라이언스)", str(r.get("_제외사유", ""))))
    # 사유별 정렬(구분 → 사유)
    rows.sort(key=lambda x: (x[0], x[1]))
    # 원금 숫자화
    for row in rows:
        try:
            row[6] = float(str(row[6]).replace(",", "")) if str(row[6]) != "" else ""
        except ValueError:
            pass
    _write_grid(ws, cols, rows)


def _write_diagnostics(wb, diagnostics: Dict):
    ws = wb.create_sheet("진단·검증")
    ws.cell(row=1, column=1, value="항목").fill = HEADER_FILL
    ws.cell(row=1, column=1).font = HEADER_FONT
    ws.cell(row=1, column=2, value="값").fill = HEADER_FILL
    ws.cell(row=1, column=2).font = HEADER_FONT
    r = 2
    for k, v in diagnostics.items():
        if isinstance(v, (str, int, float, bool)):
            ws.cell(row=r, column=1, value=str(k))
            ws.cell(row=r, column=2, value=v if not isinstance(v, bool) else ("Y" if v else "N"))
            r += 1
    ws.column_dimensions["A"].width = 42
    ws.column_dimensions["B"].width = 46
    ws.freeze_panes = "A2"
