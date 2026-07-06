# -*- coding: utf-8 -*-
"""
reasons.py — 추천사유/하향사유 문장 생성 (§15)
===============================================
실무자가 납득할 수 있는 한 줄 사유. 등급하향사유·S상한하향사유는 grading에서
별도 컬럼으로 관리하고, 여기서는 추천사유(긍정 근거)를 만든다.
"""
from __future__ import annotations

import pandas as pd

from . import config, util

_GRADE_PREFIX = {
    config.TIER_IMMEDIATE: "즉시관리 · 접촉 시 변제가능성 최상위",
    config.TIER_MONTH: "당월관리 · 이번 달 우선 관리",
    config.TIER_HOLD: "보류 · 차월 재평가",
}

_M = 10_000


def _dominant_clause(row) -> str:
    within180 = bool(row.get("within180", False))
    within365 = bool(row.get("within365", False))
    has_pay = bool(row.get("has_payment", False))
    fb = float(row.get("feedback_adj", 0) or 0)
    sim = float(row.get("paid_similarity_score", 50) or 50)
    burden = float(row.get("burden_score", 50) or 50)

    if within180:
        return "최근 3개월 내 유효입금 존재"
    if within365:
        return "최근 1년 내 입금이력"
    if has_pay:
        return "과거 입금이력 보유"
    if fb >= 2:
        return "과거 동일군 대비 회수확률 높음"
    if sim >= 75:
        return "동일상품군 평균 대비 점수 우수"
    if burden >= 85:
        return "잔액 조건상 회수 현실성 높음"
    return "잔액·상태 조건 참고"


def _modifier_clauses(row) -> list:
    mods = []
    rank = row.get("추천순위_차주기준")
    if rank is not None and pd.notna(rank) and int(rank) <= 20:
        mods.append(f"담당자 내 상위 {int(rank)}위")

    multi = float(row.get("동일차주계좌합산원금잔액", row.get("원금잔액", 0)) or 0)
    if multi >= 3000 * _M:
        mods.append("차주합산 원금잔액 큼")

    rr = row.get("누적변제율")
    if rr is not None and pd.notna(rr) and float(rr) >= 0.3:
        mods.append("누적변제율 양호")

    coll_key = str(row.get("collateral_key", "normal"))
    if coll_key != "normal":
        mods.append("담보부NPL 제한 추천")

    if bool(row.get("최근입금단독추천여부", False)):
        mods.append("최근입금 단독요인 주의")
    if bool(row.get("최근입금가점제한여부", False)):
        mods.append("소액입금 가점 제한 적용")

    rehab = util.clean_str(row.get("회생", ""))
    if rehab and rehab not in ("정상",) and rehab not in config.REHAB_IGNORE:
        mods.append("회생 여부 주의")

    penalty = float(row.get("sensitive_penalty", 0) or 0)
    if penalty > 0:
        mods.append("민원/법조치 이력 주의")

    if bool(row.get("정원초과편입여부", False)):
        mods.append("정원초과 고스코어 편입(151~200)")
    elif bool(row.get("상한적용여부", False)) and str(row.get("최종등급")) == config.TIER_HOLD:
        mods.append("정원 초과로 보류(사유 별도표시)")
    return mods


def make_reason(row) -> str:
    grade = str(row.get("최종등급", row.get("등급", config.TIER_HOLD)))
    prefix = _GRADE_PREFIX.get(grade, grade)
    parts = [f"{prefix} · {_dominant_clause(row)}"]
    parts.extend(_modifier_clauses(row)[:3])
    return " · ".join(parts)


def add_reasons(borrowers: pd.DataFrame) -> pd.DataFrame:
    out = borrowers.copy()
    if len(out) == 0:
        out["추천사유"] = []
        return out
    out["추천사유"] = [make_reason(out.iloc[i]) for i in range(len(out))]
    return out
