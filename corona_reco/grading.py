# -*- coding: utf-8 -*-
"""
grading.py — 차주 통합, 등급, 부담당자 상한/최소보장, 팀 배정
================================================================
- 관리담당자 = 부담당자(공란이면 담당자 fallback, 없으면 미지정)
- 팀 = 부담당자 roster 우선(데이터 팀 컬럼과 충돌 시 roster가 이김)
- 차주(고객번호+성명) 통합 점수 → 등급(절대컷) → 담보부NPL 등급상한
- 부담당자별 S≤20 / A≤100 초과분 하향, 추천리스트 최소 20명 보장
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config, util
from .io_loader import get_col

GRADE_RANK = {g: i for i, g in enumerate(config.GRADE_ORDER)}  # S=0 ... D=4


def manager_of_row(bu_dam, dam) -> str:
    """관리담당자 결정: 부담당자 우선, 공란이면 담당자, 없으면 미지정."""
    b = util.clean_str(bu_dam)
    if b:
        return b
    d = util.clean_str(dam)
    if d:
        return d
    return config.UNASSIGNED_LABEL


def assign_managers(df: pd.DataFrame) -> pd.Series:
    bu = get_col(df, "부담당자")
    da = get_col(df, "담당자")
    return pd.Series(
        [manager_of_row(bu.iloc[i], da.iloc[i]) for i in range(len(df))],
        index=df.index, name="관리담당자",
    )


def team_of_manager(manager: str) -> str:
    """부담당자 roster 우선 팀 매핑. 미등록이면 기타."""
    return config.ROSTER_LOOKUP.get(util.clean_str(manager), config.TEAM_ETC)


def assign_teams(manager_series: pd.Series) -> pd.Series:
    return pd.Series([team_of_manager(m) for m in manager_series],
                     index=manager_series.index, name="팀_roster")


def team_mismatch_count(df: pd.DataFrame, manager_series: pd.Series) -> int:
    """데이터의 팀 컬럼과 roster 팀이 충돌하는 건수(진단용)."""
    data_team = get_col(df, "팀")
    cnt = 0
    for i in range(len(df)):
        dt = util.clean_str(data_team.iloc[i])
        rt = team_of_manager(manager_series.iloc[i])
        if dt and rt != config.TEAM_ETC and dt != rt:
            cnt += 1
    return cnt


def _worse_grade(g1: str, g2: str) -> str:
    """두 등급 중 더 낮은(제한적인) 등급."""
    return g1 if GRADE_RANK.get(g1, 99) >= GRADE_RANK.get(g2, 99) else g2


def grade_from_score(score: float, cuts: Dict[str, float]) -> str:
    if score >= cuts["S"]:
        return "S"
    if score >= cuts["A"]:
        return "A"
    if score >= cuts["B"]:
        return "B"
    if score >= cuts["C"]:
        return "C"
    return "D"


def aggregate_borrowers(df: pd.DataFrame, scores: pd.DataFrame,
                        pay: pd.DataFrame, manager_series: pd.Series,
                        team_series: pd.Series,
                        age_bands: Optional[pd.Series] = None,
                        biz_series: Optional[pd.Series] = None) -> pd.DataFrame:
    """비제외 계좌를 차주(고객번호+성명)로 통합.

    borrower_score = 0.70*max(final) + 0.30*mean(final) - 다중계좌부담패널티
    age_bands/biz_series: 세그먼트 저장용 대표값(나이대/개인사업자).
    """
    work = df.copy()
    if age_bands is not None:
        work["_나이대"] = age_bands.reindex(df.index).fillna("unknown").values
    else:
        work["_나이대"] = "unknown"
    if biz_series is not None:
        work["_개인사업자"] = biz_series.reindex(df.index).fillna(0).astype(int).values
    else:
        work["_개인사업자"] = 0
    work["_final"] = scores["final_score"]
    work["_manager"] = manager_series
    work["_team"] = team_series
    work["_coll_cap"] = scores["collateral_cap"]
    work["_coll_key"] = scores["collateral_key"]
    work["_cur_principal"] = [util.to_number_or(v, 0.0) for v in get_col(df, "현재원금")]
    work["_multi_sum"] = [util.to_number(v) for v in get_col(df, "다중계좌 원금합계")]
    work["_rehab"] = [util.clean_str(v) for v in get_col(df, "회생")]
    work["_status_mid"] = [util.clean_str(v) for v in get_col(df, "채권상태(중)")]
    work["_loan_no"] = [util.clean_str(v) for v in get_col(df, "대출번호")]
    work["_gonum"] = [util.clean_str(v) for v in get_col(df, "고객번호")]
    work["_name"] = [util.clean_str(v) for v in get_col(df, "성명")]
    work["_has_pay"] = pay["입금_이력유무"].values
    work["_within365"] = pay["입금_최근365일"].values
    work["_within180"] = pay["입금_최근180일"].values
    # reasons/저장용 대표 세부점수
    for c in ["base_score", "paid_similarity_score", "payment_history_score",
              "burden_score", "external_prior_adj", "feedback_adj", "sensitive_penalty",
              "stage2_adj"]:
        if c in scores.columns:
            work["_" + c] = scores[c]
        else:
            work["_" + c] = 0.0

    rows = []
    for bkey, grp in work.groupby("_차주키", sort=False):
        finals = grp["_final"].astype(float)
        max_f = finals.max()
        mean_f = finals.mean()
        n_acct = len(grp)
        penalty = 0.0
        if n_acct >= 5:
            penalty = config.MULTI_ACCT_PENALTY[5]
        elif n_acct >= 3:
            penalty = config.MULTI_ACCT_PENALTY[3]
        bscore = (config.BORROWER_MAX_WEIGHT * max_f
                  + config.BORROWER_MEAN_WEIGHT * mean_f - penalty)
        bscore = util.clamp(bscore, 0, 100)

        # 대표 계좌 = final 최대 계좌
        rep_i = finals.idxmax()
        rep = grp.loc[rep_i]

        # 등급상한: 차주의 계좌 중 가장 제한적인 cap
        cap = "S"
        for c in grp["_coll_cap"]:
            cap = _worse_grade(cap, str(c))

        # 원금잔액 = 비제외 계좌 현재원금 합
        principal = grp["_cur_principal"].sum()
        # 다중계좌원금잔액 = 다중계좌 원금합계(있으면) 아니면 현재원금 합
        multi_vals = grp["_multi_sum"].dropna()
        multi_sum = float(multi_vals.max()) if len(multi_vals) else float(principal)

        # 부담당자: 미지정이 아닌 대표값 우선
        mgrs = [m for m in grp["_manager"] if m != config.UNASSIGNED_LABEL]
        manager = mgrs[0] if mgrs else config.UNASSIGNED_LABEL
        team = team_of_manager(manager)

        rows.append({
            "_차주키": bkey,
            "대표index": rep_i,
            "고객번호": rep["_gonum"],
            "성명": rep["_name"],
            "대표대출번호": rep["_loan_no"],
            "부담당자": manager,
            "팀": team,
            "원금잔액": principal,
            "다중계좌원금잔액": multi_sum,
            "회생": rep["_rehab"],
            "채권상태(중)": rep["_status_mid"],
            "나이대": rep["_나이대"],
            "개인사업자": int(rep["_개인사업자"]),
            "활동계좌수": n_acct,
            "borrower_score": bscore,
            "등급상한": cap,
            "collateral_key": rep["_coll_key"],
            "has_payment": bool(rep["_has_pay"]),
            "within365": bool(rep["_within365"]),
            "within180": bool(rep["_within180"]),
            "base_score": rep["_base_score"],
            "paid_similarity_score": rep["_paid_similarity_score"],
            "payment_history_score": rep["_payment_history_score"],
            "burden_score": rep["_burden_score"],
            "external_prior_adj": rep["_external_prior_adj"],
            "feedback_adj": rep["_feedback_adj"],
            "sensitive_penalty": rep["_sensitive_penalty"],
            "stage2_adj": rep["_stage2_adj"],
            "대표final": float(max_f),
        })

    return pd.DataFrame(rows)


def assign_grades(borrowers: pd.DataFrame, cuts: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """절대컷 등급 + 담보부NPL 등급상한 적용."""
    if cuts is None:
        cuts = dict(config.GRADE_CUTS)
    out = borrowers.copy()
    raw = [grade_from_score(s, cuts) for s in out["borrower_score"]]
    capped = [_worse_grade(raw[i], out["등급상한"].iloc[i]) for i in range(len(out))]
    out["등급_raw"] = raw
    out["등급"] = capped
    return out


def apply_assignee_caps(borrowers: pd.DataFrame,
                        cap_s: int = None, cap_a: int = None) -> pd.DataFrame:
    """부담당자별 S≤cap_s, A≤cap_a. 초과분은 한 단계씩 하향.

    담보부NPL 등급상한을 다시 어기지 않도록 하향 시에도 상한 준수.
    """
    if cap_s is None:
        cap_s = config.CAP_S_PER_ASSIGNEE
    if cap_a is None:
        cap_a = config.CAP_A_PER_ASSIGNEE
    out = borrowers.copy()
    out["등급"] = out["등급"].astype(object)

    for mgr, grp in out.groupby("부담당자", sort=False):
        idx_sorted = grp.sort_values("borrower_score", ascending=False).index

        # S 상한
        s_idx = [i for i in idx_sorted if out.at[i, "등급"] == "S"]
        for j, i in enumerate(s_idx):
            if j >= cap_s:
                # 하향: S -> A (상한 준수)
                out.at[i, "등급"] = _worse_grade("A", out.at[i, "등급상한"])

        # A 상한 (하향된 것 포함해 재계산)
        a_idx = [i for i in idx_sorted if out.at[i, "등급"] == "A"]
        for j, i in enumerate(a_idx):
            if j >= cap_a:
                out.at[i, "등급"] = _worse_grade("B", out.at[i, "등급상한"])

    return out


def apply_min_guarantee(borrowers: pd.DataFrame,
                        min_reco: int = None) -> pd.DataFrame:
    """부담당자별 추천리스트 최소 min_reco명 보장.

    추천여부=True: 등급 ∈ {S,A,B,C}(=score≥C컷). 부족 시 상위 점수 D도 편입해
    최소 인원 확보(등급은 D 유지 — 억지 S/A 승급 금지). 적격자 부족 시 전원.
    """
    if min_reco is None:
        min_reco = config.MIN_RECO_PER_ASSIGNEE
    out = borrowers.copy()
    reco = out["등급"].isin(["S", "A", "B", "C"]).values.copy()
    out["추천여부"] = reco

    for mgr, grp in out.groupby("부담당자", sort=False):
        idx_sorted = list(grp.sort_values("borrower_score", ascending=False).index)
        current = [i for i in idx_sorted if out.at[i, "추천여부"]]
        if len(current) >= min_reco:
            continue
        # 부족분을 상위 점수(추천여부 False)로 채움
        need = min_reco - len(current)
        fillers = [i for i in idx_sorted if not out.at[i, "추천여부"]]
        for i in fillers[:need]:
            out.at[i, "추천여부"] = True
    return out


def full_grading_pipeline(df: pd.DataFrame, scores: pd.DataFrame, pay: pd.DataFrame,
                          manager_series: pd.Series, team_series: pd.Series,
                          cuts: Optional[Dict[str, float]] = None,
                          cap_s: int = None, cap_a: int = None,
                          min_reco: int = None,
                          age_bands: Optional[pd.Series] = None,
                          biz_series: Optional[pd.Series] = None) -> pd.DataFrame:
    """비제외 계좌 → 차주 통합 → 등급 → 상한 → 최소보장. borrower 테이블 반환."""
    borrowers = aggregate_borrowers(df, scores, pay, manager_series, team_series,
                                    age_bands=age_bands, biz_series=biz_series)
    if len(borrowers) == 0:
        return borrowers
    borrowers = assign_grades(borrowers, cuts)
    borrowers = apply_assignee_caps(borrowers, cap_s, cap_a)
    borrowers = apply_min_guarantee(borrowers, min_reco)
    # 정렬: 등급, 점수 내림차순
    borrowers["_grank"] = borrowers["등급"].map(lambda g: GRADE_RANK.get(g, 99))
    borrowers = borrowers.sort_values(["_grank", "borrower_score"],
                                      ascending=[True, False]).reset_index(drop=True)
    return borrowers
