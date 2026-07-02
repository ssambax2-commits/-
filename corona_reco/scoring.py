# -*- coding: utf-8 -*-
"""
scoring.py — 점수 컴포넌트 & 최종 방정식 (§5)
==============================================
각 컴포넌트는 0~100 정규화/clamp. 최종식:
  pre = w.base*base + w.sim*sim + w.pay*pay + w.burden*burden
  final = clamp(pre * collateral_mult - sensitive_penalty
                + external_prior_adj + feedback_adj, 0, 100)
external_prior_adj / feedback_adj / sensitive_penalty 는 가중합 밖 additive.
"""
from __future__ import annotations

import datetime as _dt
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import config, util
from .io_loader import get_col

# 만원/억 단위 상수
_M = 10_000  # 만원


# ---------------------------------------------------------------------------
# 가중치 정규화 및 ML 가용성에 따른 자동 재분배
# ---------------------------------------------------------------------------
def normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    keys = ["base", "paid_similarity", "payment_history", "burden"]
    vals = {k: max(0.0, float(weights.get(k, 0.0))) for k in keys}
    total = sum(vals.values())
    if total <= 0:
        return dict(config.DEFAULT_WEIGHTS)
    return {k: v / total for k, v in vals.items()}


def adjust_weights_for_ml(weights: Dict[str, float], ml_active: bool) -> Dict[str, float]:
    """ML 비활성 시 paid_similarity 가중치를 base/payment/burden 로 재분배.

    positive가 적을 때 규칙점수 비중을 자동으로 높인다(§1-3).
    """
    w = normalize_weights(weights)
    if ml_active:
        return w
    sim_w = w.get("paid_similarity", 0.0)
    if sim_w <= 0:
        return w
    remain = {k: w[k] for k in ("base", "payment_history", "burden")}
    rtotal = sum(remain.values())
    if rtotal <= 0:
        # 전부 sim에 있었다면 base로 이관
        return {"base": 1.0, "paid_similarity": 0.0, "payment_history": 0.0, "burden": 0.0}
    redistributed = {k: remain[k] + sim_w * (remain[k] / rtotal) for k in remain}
    redistributed["paid_similarity"] = 0.0
    return normalize_weights(redistributed)


def to_percentile(prob: pd.Series) -> pd.Series:
    """확률(0~1)을 채점대상 모집단 백분위(0~100)로 변환. NaN은 중앙값(50).

    스케일 붕괴 방지: 실제 값 분포의 rank 기반.
    """
    s = pd.to_numeric(prob, errors="coerce")
    valid = s.dropna()
    if len(valid) == 0:
        return pd.Series([50.0] * len(prob), index=prob.index)
    if valid.nunique() == 1:
        # 전부 동일 → 중앙값
        out = pd.Series([50.0] * len(prob), index=prob.index)
        return out
    ranks = s.rank(method="average", pct=True) * 100.0
    return ranks.fillna(50.0)


# ---------------------------------------------------------------------------
# base_score (HTML v8-1 회수가능성 규칙점수 계승, 50 기준 가감)
# ---------------------------------------------------------------------------
# 채권상태(중) 가점/감점 (안내우선순위 보조)
_STATUS_MID_BASE_ADJ = {
    "약속자": config.BASE_ADJ_PROMISE,        # +3
    "약불자": -config.BASE_ADJ_BROKEN,        # -3
    "정상": 3, "정상_정상": 3,
    "재통화": 1, "연락처": 1,
    "연체": 0, "기타": 0, "취소": -1,
    "폐지_신복": 1, "폐지_회생": 1, "폐지_파산": 1,
}


def compute_base_score(df: pd.DataFrame, pay: pd.DataFrame, ref_date: _dt.date) -> pd.Series:
    n = len(df)
    status_mid = get_col(df, "채권상태(중)")
    last_int_income = get_col(df, "최종이자수입일")
    cur_principal = get_col(df, "현재원금")
    reg_reason = get_col(df, "등록사유발생일")
    multi = get_col(df, "다중계좌 활동/총건수")

    scores = np.full(n, 50.0)
    for i in range(n):
        s = 50.0
        st = util.clean_str(status_mid.iloc[i])
        s += _STATUS_MID_BASE_ADJ.get(st, 0)

        # 최종이자수입일 약한 회계활동 신호 — 입금내역 있으면 입금 우선(중복금지)
        has_pay = bool(pay["입금_이력유무"].iloc[i]) if "입금_이력유무" in pay else False
        if not has_pay:
            d = util.parse_date(last_int_income.iloc[i])
            days = util.days_between(ref_date, d)
            if days is not None and days >= 0:
                if days <= 365:
                    s += 5
                elif days <= 730:
                    s += 2

        # 현재원금 규모(약한 신호; ML 아닌 규칙점수)
        cp = util.to_number(cur_principal.iloc[i])
        if cp is not None:
            if cp <= 300 * _M:
                s += 3
            elif cp > 5000 * _M:
                s -= 5

        # 등록사유 경과년수(연체 장기화)
        rd = util.parse_date(reg_reason.iloc[i])
        yrs = util.months_between(ref_date, rd)
        if yrs is not None:
            yrs = yrs / 12.0
            if yrs > 10:
                s -= 5
            elif yrs > 5:
                s -= 3

        # 다중계좌 활동건수
        mv = _multi_active(multi.iloc[i])
        if mv is not None and mv >= 3:
            s -= 2

        scores[i] = util.clamp(s, 0, 100)
    return pd.Series(scores, index=df.index, name="base_score")


def _multi_active(val):
    """'활동/총건수'에서 활동건수."""
    s = util.clean_str(val)
    if s == "":
        return None
    if "/" in s:
        n = util.to_number(s.split("/")[0])
        return n
    return util.to_number(s)


# ---------------------------------------------------------------------------
# payment_history_score (실제 입금이력 직접 우대)
# ---------------------------------------------------------------------------
def compute_payment_history_score(pay: pd.DataFrame) -> pd.Series:
    n = len(pay)
    scores = np.zeros(n)
    for i in range(n):
        if not bool(pay["입금_이력유무"].iloc[i]):
            scores[i] = 0.0
            continue
        if bool(pay["입금_최근180일"].iloc[i]):
            base = 100.0
        elif bool(pay["입금_최근365일"].iloc[i]):
            base = 80.0
        elif bool(pay["입금_최근730일"].iloc[i]):
            base = 60.0
        elif pay["입금_최근일"].iloc[i] is not None:
            base = 40.0  # 그 이전(오래된 입금)
        elif bool(pay["입금_일자불명금액존재"].iloc[i]):
            base = 50.0  # 일자불명·금액만
        else:
            base = 40.0

        # 보정(건수/총액) — 차원별 최고 티어만
        bonus = 0.0
        if bool(pay["입금_건수3이상"].iloc[i]):
            bonus += 10
        elif bool(pay["입금_건수2이상"].iloc[i]):
            bonus += 5
        if bool(pay["입금_총액50만이상"].iloc[i]):
            bonus += 10
        elif bool(pay["입금_총액10만이상"].iloc[i]):
            bonus += 5

        scores[i] = util.clamp(base + bonus, 0, 100)
    return pd.Series(scores, index=pay.index, name="payment_history_score")


# ---------------------------------------------------------------------------
# burden_score (현재원금 기준 상환 현실성)
# ---------------------------------------------------------------------------
def compute_burden_score(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    cur_principal = get_col(df, "현재원금")
    multi_sum = get_col(df, "다중계좌 원금합계")
    scores = np.full(n, 50.0)
    for i in range(n):
        cp = util.to_number(cur_principal.iloc[i])
        if cp is None:
            s = 50.0
        elif cp <= 100 * _M:
            s = 90.0
        elif cp <= 300 * _M:
            s = 100.0
        elif cp <= 1000 * _M:
            s = 85.0
        elif cp <= 3000 * _M:
            s = 65.0
        elif cp <= 5000 * _M:
            s = 45.0
        else:
            s = 25.0

        ms = util.to_number(multi_sum.iloc[i])
        if ms is not None:
            if ms > 5000 * _M:
                s -= 20
            elif ms > 3000 * _M:
                s -= 10
            elif ms > 1000 * _M:
                s -= 5
        scores[i] = util.clamp(s, 0, 100)
    return pd.Series(scores, index=df.index, name="burden_score")


# ---------------------------------------------------------------------------
# external_prior_adj (나이대 ±3 + 개인사업자 + macro, [-8,+5] clamp)
# ---------------------------------------------------------------------------
def compute_external_prior_adj(age_bands: pd.Series, biz_series: pd.Series,
                               pay: pd.DataFrame) -> pd.Series:
    n = len(pay)
    ep = config.EXTERNAL_PRIORS
    age_map = ep["age_band_adj"]
    biz_map = ep["biz_adj"]
    macro = ep.get("macro_adj", 0)
    cap_up = ep["prior_cap_up"]
    cap_dn = ep["prior_cap_dn"]

    out = np.zeros(n)
    for i in range(n):
        band = str(age_bands.iloc[i]) if age_bands is not None else "unknown"
        age_adj = age_map.get(band, 0)
        age_adj = util.clamp(age_adj, -config.AGE_ADJ_CAP, config.AGE_ADJ_CAP)

        is_biz = bool(biz_series.iloc[i]) if biz_series is not None else False
        if not is_biz:
            biz_adj = 0
        else:
            if bool(pay["입금_최근365일"].iloc[i]):
                biz_adj = biz_map["recent365"]
            elif bool(pay["입금_이력유무"].iloc[i]):
                biz_adj = biz_map["hasPay"]
            else:
                biz_adj = biz_map["none"]

        total = age_adj + biz_adj + macro
        out[i] = util.clamp(total, cap_dn, cap_up)
    return pd.Series(out, index=pay.index, name="external_prior_adj")


# ---------------------------------------------------------------------------
# sensitive_penalty (민원 -25 중복1회, 법조치 -20). 양수로 반환(최종식에서 뺌).
# ---------------------------------------------------------------------------
def compute_sensitive_penalty(df: pd.DataFrame) -> pd.Series:
    n = len(df)
    minwon_flag = get_col(df, "민원여부")
    status_mid = get_col(df, "채권상태(중)")
    out = np.zeros(n)
    for i in range(n):
        penalty = 0.0
        st = util.clean_str(status_mid.iloc[i])
        is_minwon = util.is_present(minwon_flag.iloc[i]) or (st == "민원")
        if is_minwon:
            penalty += config.PENALTY_MINWON  # 중복이어도 1회만
        if st == "법조치":
            penalty += config.PENALTY_LEGAL
        out[i] = penalty
    return pd.Series(out, index=df.index, name="sensitive_penalty")


# ---------------------------------------------------------------------------
# collateral_mult / cap (담보부NPL 표)
# ---------------------------------------------------------------------------
def compute_collateral(df: pd.DataFrame, pay: pd.DataFrame):
    n = len(df)
    bond_type = get_col(df, "채권구분")
    coll = config.EXTERNAL_PRIORS["collateral"]
    mult = np.ones(n)
    cap = ["S"] * n
    key = ["normal"] * n
    for i in range(n):
        bt = util.clean_str(bond_type.iloc[i])
        if bt != config.COLLATERAL_NPL_LABEL:
            m, c = coll["normal"]
            mult[i], cap[i], key[i] = m, c, "normal"
            continue
        if bool(pay["입금_최근365일"].iloc[i]):
            k = "recent365"
        elif bool(pay["입금_이력유무"].iloc[i]):
            k = "hasPay"
        else:
            k = "none"
        m, c = coll[k]
        mult[i], cap[i], key[i] = m, c, k
    return (pd.Series(mult, index=df.index, name="collateral_mult"),
            pd.Series(cap, index=df.index, name="collateral_cap"),
            pd.Series(key, index=df.index, name="collateral_key"))


# ---------------------------------------------------------------------------
# 최종 조립
# ---------------------------------------------------------------------------
def compute_scores(df: pd.DataFrame, pay: pd.DataFrame, ref_date: _dt.date,
                   age_bands: Optional[pd.Series] = None,
                   biz_series: Optional[pd.Series] = None,
                   paid_prob: Optional[pd.Series] = None,
                   feedback_adj: Optional[pd.Series] = None,
                   stage2_adj: Optional[pd.Series] = None,
                   weights: Optional[Dict[str, float]] = None,
                   ml_active: bool = None) -> pd.DataFrame:
    """전체 채점. 반환 DataFrame(각 세부점수 + pre_score + final_corona_score).

    stage2_adj: Stage2(예상회수) 활성 시 소폭 additive 보너스(기본 없음=0).
    """
    if weights is None:
        weights = dict(config.DEFAULT_WEIGHTS)
    if ml_active is None:
        ml_active = paid_prob is not None
    eff_w = adjust_weights_for_ml(weights, ml_active)

    base = compute_base_score(df, pay, ref_date)
    payh = compute_payment_history_score(pay)
    burden = compute_burden_score(df)
    prior_adj = compute_external_prior_adj(age_bands, biz_series, pay)
    penalty = compute_sensitive_penalty(df)
    coll_mult, coll_cap, coll_key = compute_collateral(df, pay)

    if paid_prob is not None and ml_active:
        sim = to_percentile(paid_prob)
    else:
        # ML 비활성: sim 가중치 0 이므로 값은 무의미(중립 50)
        sim = pd.Series([50.0] * len(df), index=df.index)

    if feedback_adj is None:
        fb = pd.Series(np.zeros(len(df)), index=df.index)
    else:
        fb = feedback_adj.reindex(df.index).fillna(0.0)

    if stage2_adj is None:
        s2 = pd.Series(np.zeros(len(df)), index=df.index)
    else:
        s2 = stage2_adj.reindex(df.index).fillna(0.0)

    pre = (eff_w["base"] * base
           + eff_w["paid_similarity"] * sim
           + eff_w["payment_history"] * payh
           + eff_w["burden"] * burden)

    final = (pre * coll_mult) - penalty + prior_adj + fb + s2
    final = final.clip(lower=0, upper=100)

    out = pd.DataFrame(index=df.index)
    out["base_score"] = base
    out["paid_similarity_score"] = sim
    out["payment_history_score"] = payh
    out["burden_score"] = burden
    out["external_prior_adj"] = prior_adj
    out["sensitive_penalty"] = penalty
    out["collateral_mult"] = coll_mult
    out["collateral_cap"] = coll_cap
    out["collateral_key"] = coll_key
    out["feedback_adj"] = fb
    out["stage2_adj"] = s2
    out["pre_score"] = pre
    out["final_score"] = final
    return out
