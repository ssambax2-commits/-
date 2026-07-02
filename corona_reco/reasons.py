# -*- coding: utf-8 -*-
"""
reasons.py — 추천사유 문장 생성 (§8-4)
=======================================
점수 구성의 지배적 신호 + 등급으로 사람이 읽는 한 줄 문구를 만든다.
"""
from __future__ import annotations

import pandas as pd

from . import config, util

_GRADE_PREFIX = {
    "S": "S등급 고우선",
    "A": "A등급 관리 우선",
    "B": "B등급 확인 권장",
    "C": "C등급 참고 대상",
    "D": "D등급 후순위",
}

_M = 10_000


def _dominant_clause(row) -> str:
    """지배적 신호 한 개를 문구로."""
    within180 = bool(row.get("within180", False))
    within365 = bool(row.get("within365", False))
    has_pay = bool(row.get("has_payment", False))
    fb = float(row.get("feedback_adj", 0) or 0)
    sim = float(row.get("paid_similarity_score", 50) or 50)
    burden = float(row.get("burden_score", 50) or 50)

    if within180:
        return "최근 입금이력 양호"
    if within365:
        return "최근 1년 내 입금이력"
    if has_pay:
        return "과거 입금이력 보유"
    if fb >= 2:
        return "전월 피드백상 유사구간 입금률 높음"
    if sim >= 75:
        return "동일 등급 대비 회수 가능성 높음"
    if burden >= 85:
        return "잔액 조건상 우선 확인"
    return "잔액·상태 조건 참고"


def _modifier_clauses(row) -> list:
    mods = []
    coll_key = str(row.get("collateral_key", "normal"))
    if coll_key != "normal":
        if row.get("has_payment", False):
            mods.append("담보부NPL·입금이력으로 제한 추천")
        else:
            mods.append("담보부NPL 제한 추천")

    principal = float(row.get("원금잔액", 0) or 0)
    if principal > 3000 * _M:
        mods.append("잔액 규모 큼")

    rehab = util.clean_str(row.get("회생", ""))
    if rehab and rehab not in ("정상",) and rehab not in config.REHAB_IGNORE:
        mods.append("회생 여부 주의")

    penalty = float(row.get("sensitive_penalty", 0) or 0)
    if penalty > 0:
        mods.append("민원/법조치 이력 주의")

    return mods


def make_reason(row) -> str:
    """borrower 행(dict/Series) → 추천사유 한 줄."""
    grade = str(row.get("등급", "D"))
    prefix = _GRADE_PREFIX.get(grade, f"{grade}등급")
    dominant = _dominant_clause(row)
    parts = [f"{prefix} · {dominant}"]
    mods = _modifier_clauses(row)
    # 한 줄 과밀 방지: 최대 2개 수식어
    parts.extend(mods[:2])
    return " · ".join(parts)


def add_reasons(borrowers: pd.DataFrame) -> pd.DataFrame:
    if len(borrowers) == 0:
        out = borrowers.copy()
        out["추천사유"] = []
        return out
    out = borrowers.copy()
    out["추천사유"] = [make_reason(out.iloc[i]) for i in range(len(out))]
    return out
