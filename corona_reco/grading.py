# -*- coding: utf-8 -*-
"""
grading.py — 차주 통합, 원등급/최종등급, S상한(다중키 동점처리), 팀 배정
==========================================================================
- 관리담당자 = 부담당자(공란이면 담당자 fallback, 없으면 미지정)
- 팀 = 부담당자 roster 우선
- 차주(고객번호+성명) 통합점수 = 0.70×max(계좌점수) + 0.30×mean(계좌점수)
  − 다중계좌부담패널티  (계좌합산 잔액은 별도 컬럼으로 관리)
- 원등급: 절대컷 → 담보부NPL 등급상한 → 최근입금 단독요인 등급상한
- 최종등급: 부담당자별 S≤20(차주 기준) / A≤100 상한 적용 후 등급
  S상한 동점/경계 처리: ①모델점수 ②차주합산 원금잔액 ③1M 예상회수액
  ④3M 예상회수액 ⑤최근입금일 최신 ⑥고객번호/대출번호 안정 정렬
- 추천여부=True 는 S/A/B 만. 최소건수 보정 편입분은 참고대상(별도 컬럼).
"""
from __future__ import annotations

import datetime as _dt
from typing import Dict, Optional

import pandas as pd

from . import config, util
from .io_loader import get_col

GRADE_RANK = {g: i for i, g in enumerate(config.GRADE_ORDER)}  # S=0 ... D=4


# ---------------------------------------------------------------------------
# 담당자/팀
# ---------------------------------------------------------------------------
def manager_of_row(bu_dam, dam) -> str:
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
    return pd.Series([manager_of_row(bu.iloc[i], da.iloc[i]) for i in range(len(df))],
                     index=df.index, name="관리담당자")


def team_of_manager(manager: str) -> str:
    return config.ROSTER_LOOKUP.get(util.clean_str(manager), config.TEAM_ETC)


def assign_teams(manager_series: pd.Series) -> pd.Series:
    return pd.Series([team_of_manager(m) for m in manager_series],
                     index=manager_series.index, name="팀_roster")


def team_mismatch_count(df: pd.DataFrame, manager_series: pd.Series) -> int:
    data_team = get_col(df, "팀")
    cnt = 0
    for i in range(len(df)):
        dt_ = util.clean_str(data_team.iloc[i])
        rt = team_of_manager(manager_series.iloc[i])
        if dt_ and rt != config.TEAM_ETC and dt_ != rt:
            cnt += 1
    return cnt


def _worse_grade(g1: str, g2: str) -> str:
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


# ---------------------------------------------------------------------------
# 차주 통합
# ---------------------------------------------------------------------------
def aggregate_borrowers(df: pd.DataFrame, scores: pd.DataFrame,
                        pay: pd.DataFrame, manager_series: pd.Series,
                        team_series: pd.Series,
                        age_bands: Optional[pd.Series] = None,
                        biz_series: Optional[pd.Series] = None,
                        exp_recovery: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """비제외 계좌 → 차주(고객번호+성명) 통합.

    borrower_score = 0.70*max(계좌 final) + 0.30*mean(계좌 final) − 다중계좌패널티
    exp_recovery: 계좌 index 기준 e1m/e3m (예상회수액). 차주값 = 계좌 합.
    """
    work = df.copy()
    work["_나이대"] = (age_bands.reindex(df.index).fillna("unknown").values
                     if age_bands is not None else "unknown")
    work["_개인사업자"] = (biz_series.reindex(df.index).fillna(0).astype(int).values
                       if biz_series is not None else 0)
    work["_final"] = scores["final_score"]
    work["_manager"] = manager_series
    work["_team"] = team_series
    work["_coll_cap"] = scores["collateral_cap"]
    work["_coll_key"] = scores["collateral_key"]
    work["_cur_principal"] = [util.to_number_or(v, 0.0) for v in get_col(df, "현재원금")]
    work["_first_principal"] = [util.to_number(v) for v in get_col(df, "최초원금")]
    work["_multi_sum"] = [util.to_number(v) for v in get_col(df, "다중계좌 원금합계")]
    work["_rehab"] = [util.clean_str(v) for v in get_col(df, "회생")]
    work["_status_mid"] = [util.clean_str(v) for v in get_col(df, "채권상태(중)")]
    work["_loan_no"] = [util.clean_str(v) for v in get_col(df, "대출번호")]
    work["_gonum"] = [util.clean_str(v) for v in get_col(df, "고객번호")]
    work["_name"] = [util.clean_str(v) for v in get_col(df, "성명")]
    work["_has_pay"] = pay["입금_이력유무"].values
    work["_within365"] = pay["입금_최근365일"].values
    work["_within180"] = pay["입금_최근180일"].values
    work["_last_pay_date"] = pay["입금_최근일"].values
    work["_last_pay_amt"] = pay["입금_최근액"].values
    work["_e1m"] = (exp_recovery["e1m"].reindex(df.index).fillna(0.0).values
                    if exp_recovery is not None else 0.0)
    work["_e3m"] = (exp_recovery["e3m"].reindex(df.index).fillna(0.0).values
                    if exp_recovery is not None else 0.0)
    for c in ["base_score", "paid_similarity_score", "payment_history_score",
              "burden_score", "external_prior_adj", "feedback_adj",
              "sensitive_penalty", "stage2_adj", "최근입금의존도",
              "최근입금가점제한여부"]:
        work["_" + c] = scores[c] if c in scores.columns else 0.0

    rows = []
    for bkey, grp in work.groupby("_차주키", sort=False):
        finals = grp["_final"].astype(float)
        max_f, mean_f = finals.max(), finals.mean()
        n_acct = len(grp)
        penalty = 0.0
        if n_acct >= 5:
            penalty = config.MULTI_ACCT_PENALTY[5]
        elif n_acct >= 3:
            penalty = config.MULTI_ACCT_PENALTY[3]
        bscore = util.clamp(config.BORROWER_MAX_WEIGHT * max_f
                            + config.BORROWER_MEAN_WEIGHT * mean_f - penalty, 0, 100)

        rep_i = finals.idxmax()
        rep = grp.loc[rep_i]

        cap = "S"
        for c in grp["_coll_cap"]:
            cap = _worse_grade(cap, str(c))

        principal = float(grp["_cur_principal"].sum())
        multi_vals = grp["_multi_sum"].dropna()
        # 동일차주 계좌합산 원금잔액: 데이터 컬럼(있으면 최대) vs 계좌 실합산 중 큰 값
        multi_sum = max(float(multi_vals.max()) if len(multi_vals) else 0.0, principal)

        # 누적변제율(차주): 1 − Σ현재원금/Σ최초원금 (최초원금 없으면 공란)
        fp = grp["_first_principal"].dropna()
        repay_rate = None
        if len(fp) and fp.sum() > 0:
            repay_rate = util.clamp(1.0 - principal / float(fp.sum()), 0.0, 1.0)

        # 최근입금일/액: 차주 내 가장 최신
        dates = [d for d in grp["_last_pay_date"] if d is not None]
        last_date = max(dates) if dates else None
        if last_date is not None:
            amt_rows = grp[grp["_last_pay_date"] == last_date]
            last_amt = float(pd.to_numeric(amt_rows["_last_pay_amt"],
                                           errors="coerce").fillna(0).max())
        else:
            last_amt = float(pd.to_numeric(grp["_last_pay_amt"],
                                           errors="coerce").fillna(0).max())

        mgrs = [m for m in grp["_manager"] if m != config.UNASSIGNED_LABEL]
        manager = mgrs[0] if mgrs else config.UNASSIGNED_LABEL

        rows.append({
            "_차주키": bkey,
            "대표index": rep_i,
            "고객번호": rep["_gonum"],
            "성명": rep["_name"],
            "대표대출번호": rep["_loan_no"],
            "부담당자": manager,
            "팀": team_of_manager(manager),
            "원금잔액": principal,
            "다중계좌원금잔액": multi_sum,
            "동일차주계좌합산원금잔액": multi_sum,
            "동일차주계좌수": n_acct,
            "누적변제율": repay_rate,
            "최근입금일": last_date,
            "최근입금액": last_amt,
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
            "예상회수1M": float(grp["_e1m"].sum()),
            "예상회수3M": float(grp["_e3m"].sum()),
            "최근입금의존도": float(rep["_최근입금의존도"]),
            "최근입금가점제한여부": bool(rep["_최근입금가점제한여부"]),
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


# ---------------------------------------------------------------------------
# 정렬 다중키 (§1-3): ①점수 ②차주합산잔액 ③1M ④3M ⑤최근입금일 ⑥안정키
# ---------------------------------------------------------------------------
def _sort_key_frame(df: pd.DataFrame) -> pd.DataFrame:
    k = pd.DataFrame(index=df.index)
    k["k1"] = -pd.to_numeric(df["borrower_score"], errors="coerce").fillna(0)
    k["k2"] = -pd.to_numeric(df["동일차주계좌합산원금잔액"], errors="coerce").fillna(0)
    k["k3"] = -pd.to_numeric(df.get("예상회수1M"), errors="coerce").fillna(0)
    k["k4"] = -pd.to_numeric(df.get("예상회수3M"), errors="coerce").fillna(0)
    k["k5"] = [-(d.toordinal() if isinstance(d, _dt.date) else 0)
               for d in df.get("최근입금일", pd.Series([None] * len(df), index=df.index))]
    k["k6"] = df["고객번호"].astype(str)
    k["k7"] = df.get("대표대출번호", pd.Series([""] * len(df), index=df.index)).astype(str)
    return k


_SORTCOLS = ["k1", "k2", "k3", "k4", "k5", "k6", "k7"]
_TIER_RANK = {config.TIER_IMMEDIATE: 0, config.TIER_MONTH: 1, config.TIER_HOLD: 2}


def _immediate_boundary(scores_desc) -> float:
    """즉시 vs 당월 경계(데이터 주도): 상위 점수 갭 최대 지점, 실패 시 분위수."""
    import numpy as np
    s = np.asarray(scores_desc, dtype=float)
    m = len(s)
    if m == 0:
        return 100.0
    if m <= 3:
        return float(s[0])  # 소수면 최상위만 즉시
    lo = max(1, int(m * config.IMMEDIATE_GAP_SEARCH_LO))
    hi = max(lo + 1, int(m * config.IMMEDIATE_GAP_SEARCH_HI))
    best_gap, best_i = -1.0, None
    for i in range(lo, min(hi, m)):
        gap = s[i - 1] - s[i]
        if gap > best_gap:
            best_gap, best_i = gap, i
    if best_i is not None and best_gap > 1e-9:
        return float(s[best_i - 1])
    return float(np.quantile(s, 1 - config.IMMEDIATE_QUANTILE_FALLBACK))


# ---------------------------------------------------------------------------
# 원등급 (모델 판정: 즉시/당월/보류) — 부담당자별 점수분포 기반, 상한 미적용
# ---------------------------------------------------------------------------
def assign_original_tier(borrowers: pd.DataFrame) -> pd.DataFrame:
    """500만 필터 → 원등급(즉시/당월/보류) + 즉시경계 진단.

    변경 전(v3): 절대컷 S/A/B/C/D + 담보부NPL/최근입금 상한.
    변경 후(v4): 부담당자별 점수분포에서 즉시경계를 데이터 주도로 산출,
                MONTH_SCORE_FLOOR 미만·차주합산 500만↓는 보류/제외.
    """
    import numpy as np
    out = borrowers.copy()
    out["소액제외여부"] = out["동일차주계좌합산원금잔액"] < config.BORROWER_MIN_MULTI_PRINCIPAL
    out["원등급"] = config.TIER_HOLD
    out["등급하향사유"] = ""
    out["즉시경계점수"] = None

    for mgr, grp in out.groupby("부담당자", sort=False):
        elig = grp.index[(~out.loc[grp.index, "소액제외여부"])
                         & (out.loc[grp.index, "borrower_score"] >= config.MONTH_SCORE_FLOOR)]
        if len(elig) == 0:
            continue
        scores_desc = np.sort(out.loc[elig, "borrower_score"].astype(float).values)[::-1]
        boundary = _immediate_boundary(scores_desc)
        for i in elig:
            sc = float(out.at[i, "borrower_score"])
            out.at[i, "원등급"] = config.TIER_IMMEDIATE if sc >= boundary else config.TIER_MONTH
            out.at[i, "즉시경계점수"] = round(boundary, 2)

    # 제외/보류 사유 문구
    out.loc[out["소액제외여부"], "등급하향사유"] = "차주합산 원금잔액 500만원 이하 - 추천 제외"
    hold_low = (~out["소액제외여부"]) & (out["borrower_score"] < config.MONTH_SCORE_FLOOR)
    out.loc[hold_low, "등급하향사유"] = f"변제가능성 점수 {config.MONTH_SCORE_FLOOR} 미만 - 보류"

    # §7 최근입금 단독요인 과다 → 즉시에서 당월로 강등(원등급 단계에서 제한)
    dep_cap = (out["원등급"] == config.TIER_IMMEDIATE) & \
        (pd.to_numeric(out["최근입금의존도"], errors="coerce").fillna(0) >= config.RECENT_PAY_DEP_THRESHOLD) & \
        (pd.to_numeric(out["base_score"], errors="coerce").fillna(50) < 60) & \
        (pd.to_numeric(out["paid_similarity_score"], errors="coerce").fillna(50) < 65)
    out.loc[dep_cap, "원등급"] = config.TIER_MONTH
    for i in out.index[dep_cap]:
        prev = str(out.at[i, "등급하향사유"] or "")
        out.at[i, "등급하향사유"] = (prev + "; 최근입금 단독요인 과다로 즉시→당월 제한").strip("; ")
    out["최근입금단독추천여부"] = dep_cap.values
    out["즉시관리후보여부"] = out["원등급"] == config.TIER_IMMEDIATE
    out["등급"] = out["원등급"]  # 하위 호환
    return out


# ---------------------------------------------------------------------------
# 부담당자별 상한 (§1): 즉시+당월 기본 150 / 고스코어 200 초과편입 / 200 하드캡
# ---------------------------------------------------------------------------
def apply_assignee_caps(borrowers: pd.DataFrame,
                        base_cap: int = None, hard_cap: int = None) -> pd.DataFrame:
    """원등급(즉시/당월) 차주를 부담당자별 순위화 → 최종등급 산출.

    - 순위 ≤ base_cap(150): 원등급 유지(즉시/당월)
    - base_cap < 순위 ≤ hard_cap(200): 점수 ≥ 초과편입 임계면 당월(정원초과편입),
      아니면 보류
    - 순위 > hard_cap: 보류
    동점/경계는 다중키(점수→차주합산잔액→1M→3M→최근입금일→안정키) 우선.
    """
    import numpy as np
    if base_cap is None:
        base_cap = config.ASSIGNEE_BASE_CAP
    if hard_cap is None:
        hard_cap = config.ASSIGNEE_HARD_CAP
    out = borrowers.copy()
    out["최종등급"] = out["원등급"].astype(object)
    out["정원초과편입여부"] = False
    out["상한적용여부"] = False
    out["상한하향사유"] = ""
    out["상한전_담당자내순위"] = None
    out["상한후_담당자내순위"] = None

    keys = _sort_key_frame(out)

    for mgr, grp in out.groupby("부담당자", sort=False):
        cand = [i for i in grp.index
                if out.at[i, "원등급"] in (config.TIER_IMMEDIATE, config.TIER_MONTH)]
        if not cand:
            continue
        ordered = keys.loc[cand].sort_values(_SORTCOLS).index.tolist()
        for rank, i in enumerate(ordered, start=1):
            out.at[i, "상한전_담당자내순위"] = rank

        normal = ordered[:base_cap]
        overflow_zone = ordered[base_cap:hard_cap]
        beyond = ordered[hard_cap:]

        # 초과편입 임계 = 정상편입 점수 중앙값 (절대 하한 병행)
        if normal:
            med = float(np.median([float(out.at[i, "borrower_score"]) for i in normal]))
        else:
            med = config.OVERFLOW_ABS_FLOOR
        theta = max(med, config.OVERFLOW_ABS_FLOOR)
        boundary_score = float(out.at[normal[-1], "borrower_score"]) if normal else None

        final_rank = 0
        for i in normal:
            final_rank += 1
            out.at[i, "상한후_담당자내순위"] = final_rank
            # 최종등급은 원등급 그대로(즉시/당월)

        for i in overflow_zone:
            sc = float(out.at[i, "borrower_score"])
            if sc >= theta:
                out.at[i, "최종등급"] = config.TIER_MONTH
                out.at[i, "정원초과편입여부"] = True
                out.at[i, "상한적용여부"] = True
                final_rank += 1
                out.at[i, "상한후_담당자내순위"] = final_rank
                out.at[i, "상한하향사유"] = (
                    f"정원({base_cap}) 초과 고스코어 편입(151~{hard_cap}, "
                    f"임계 {theta:.1f} 이상)")
            else:
                out.at[i, "최종등급"] = config.TIER_HOLD
                out.at[i, "상한적용여부"] = True
                same = (boundary_score is not None
                        and abs(sc - boundary_score) < 1e-9)
                out.at[i, "상한하향사유"] = (
                    f"부담당자별 정원({base_cap}/{hard_cap}) 초과 — "
                    + ("동점 경계에서 차주합산 원금잔액 기준 보류" if same
                       else f"초과편입 임계({theta:.1f}) 미달로 보류"))

        for i in beyond:
            out.at[i, "최종등급"] = config.TIER_HOLD
            out.at[i, "상한적용여부"] = True
            out.at[i, "상한하향사유"] = (
                f"부담당자별 하드캡({hard_cap}) 초과 — 보류(차월 재평가)")

    out["등급"] = out["최종등급"]  # 하위 호환
    return out


# ---------------------------------------------------------------------------
# 추천여부 / 추천순위
# ---------------------------------------------------------------------------
def apply_recommendation_flags(borrowers: pd.DataFrame) -> pd.DataFrame:
    """추천여부=True 는 즉시관리·당월관리(초과편입 포함)만(§1-4). 보류=False.

    변경 전(v3): 최소 20건 강제 편입(참고대상). 변경 후(v4): 최소건수 보정 삭제.
    """
    out = borrowers.copy()
    out["추천여부"] = out["최종등급"].isin(sorted(config.RECO_TIERS))
    return out


def assign_ranks(borrowers: pd.DataFrame) -> pd.DataFrame:
    """추천순위_차주기준: 부담당자 내 (최종등급, 다중키) 순위(1-base)."""
    out = borrowers.copy()
    keys = _sort_key_frame(out)
    keys["g"] = out["최종등급"].map(lambda g: _TIER_RANK.get(g, 9))
    out["추천순위_차주기준"] = None
    for mgr, grp in out.groupby("부담당자", sort=False):
        order = keys.loc[grp.index].sort_values(["g"] + _SORTCOLS).index
        for rank, i in enumerate(order, start=1):
            out.at[i, "추천순위_차주기준"] = rank
    return out


def full_grading_pipeline(df: pd.DataFrame, scores: pd.DataFrame, pay: pd.DataFrame,
                          manager_series: pd.Series, team_series: pd.Series,
                          cuts: Optional[Dict[str, float]] = None,
                          cap_s: int = None, cap_a: int = None,
                          min_reco: int = None,
                          base_cap: int = None, hard_cap: int = None,
                          age_bands: Optional[pd.Series] = None,
                          biz_series: Optional[pd.Series] = None,
                          exp_recovery: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """차주 통합 → 원등급(즉시/당월/보류) → 150/200 상한(최종등급) → 추천/순위."""
    borrowers = aggregate_borrowers(df, scores, pay, manager_series, team_series,
                                    age_bands=age_bands, biz_series=biz_series,
                                    exp_recovery=exp_recovery)
    if len(borrowers) == 0:
        return borrowers
    borrowers = assign_original_tier(borrowers)
    borrowers = apply_assignee_caps(borrowers, base_cap, hard_cap)
    borrowers = apply_recommendation_flags(borrowers)
    borrowers = assign_ranks(borrowers)
    borrowers["_grank"] = borrowers["최종등급"].map(lambda g: _TIER_RANK.get(g, 9))
    borrowers = borrowers.sort_values(["_grank", "borrower_score"],
                                      ascending=[True, False]).reset_index(drop=True)
    return borrowers


# ---------------------------------------------------------------------------
# §3 랜덤박스 (앱 표시 전용) — 합산 1,000만↑ 상위 20 중 무작위 5
# ---------------------------------------------------------------------------
def randombox_pick(borrowers: pd.DataFrame, seed: Optional[int] = None,
                   min_multi: int = None, top_pool: int = None,
                   pick: int = None) -> pd.DataFrame:
    """접촉가능(비제외) & 차주합산 1,000만↑ 차주를 점수순 상위 top_pool에서 무작위 pick.

    borrowers는 이미 강제제외를 통과한 채점 차주. 매 호출 새 시드 → 다른 5명.
    반환 컬럼: 고객번호 · 부담당자 · 동일차주계좌합산원금잔액.
    """
    import numpy as np
    if min_multi is None:
        min_multi = config.RANDOMBOX_MIN_MULTI_PRINCIPAL
    if top_pool is None:
        top_pool = config.RANDOMBOX_TOP_POOL
    if pick is None:
        pick = config.RANDOMBOX_PICK
    if len(borrowers) == 0:
        return pd.DataFrame(columns=["고객번호", "부담당자", "동일차주계좌합산원금잔액"])
    pool = borrowers[borrowers["동일차주계좌합산원금잔액"] >= min_multi].copy()
    if len(pool) == 0:
        return pd.DataFrame(columns=["고객번호", "부담당자", "동일차주계좌합산원금잔액"])
    pool = pool.sort_values("borrower_score", ascending=False).head(top_pool)
    rng = np.random.RandomState(seed)
    n_pick = min(pick, len(pool))
    chosen = rng.choice(pool.index.values, size=n_pick, replace=False)
    return pool.loc[chosen, ["고객번호", "부담당자", "동일차주계좌합산원금잔액"]].reset_index(drop=True)
