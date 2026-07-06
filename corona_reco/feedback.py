# -*- coding: utf-8 -*-
"""
feedback.py — 피드백 파싱 + EB 세그먼트 캘리브레이션 + 혼합비중 (§6-3/6-4)
==========================================================================
주력: 세그먼트 성과 캘리브레이션(empirical Bayes shrinkage) → additive feedback_adj.
보조: DB 추천시점 피처 + 성공여부 → ML 재학습 풀(pipeline/model에서 사용).
입금 성과는 회수비율(실제입금액/원금잔액)을 주신호로, 로그금액 보조·상한.
등급은 세그먼트 변수로 쓰지 않는다(순환 방지).
"""
from __future__ import annotations

import datetime as _dt
import re
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config, util
from .io_loader import load_excel_all_sheets, _norm_key

# 피드백 컬럼 별칭 (canonical -> 후보 표기들)
_FEEDBACK_ALIASES = {
    "실제입금액": ["실제입금액", "입금액", "납입금액"],
    "실제입금여부": ["실제입금여부", "입금여부"],
    "원금잔액": ["원금잔액", "현재원금"],
    "다중계좌원금잔액": ["다중계좌원금잔액", "다중계좌 원금합계", "다중계좌원금합계"],
    "입금일자": ["입금일자", "납입일자"],
    "추천월": ["추천월"],
    "고객번호": ["고객번호", "고객 번호"],
    "성명": ["성명", "이름", "고객명"],
    "대출번호": ["대출번호", "계좌번호", "계좌키"],
    "등급": ["등급", "최종등급"],
    "부담당자": ["부담당자", "부담당"],
    "팀": ["팀"],
    "회생": ["회생"],
    "활동여부": ["활동여부", "활동", "컨택여부"],
}

_SUCCESS_YES = {"y", "예", "입금", "성공", "1", "o", "ok", "yes", "t", "true"}
_SUCCESS_NO = {"n", "아니오", "미입금", "0", "x", "no", "f", "false", "없음"}


def _resolve_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """canonical 필드 -> 실제 컬럼명(없으면 None)."""
    norm_to_actual = {_norm_key(c): c for c in df.columns}
    resolved = {}
    for canon, cands in _FEEDBACK_ALIASES.items():
        found = None
        for cand in cands:
            k = _norm_key(cand)
            if k in norm_to_actual:
                found = norm_to_actual[k]
                break
        resolved[canon] = found
    return resolved


def success_of(paid_flag, amount) -> Optional[int]:
    """성공판정(§6-3). 반환 1/0/None(불명)."""
    amt = util.to_number(amount)
    # 금액 우선: >0 이면 성공
    if amt is not None and amt > 0:
        return 1
    flag = None if paid_flag is None else str(paid_flag).strip().lower()
    if flag:
        if flag in _SUCCESS_YES:
            return 1
        if flag in _SUCCESS_NO:
            # 미입금 + (금액 없거나 0) → 실패
            if amt is None or amt <= 0:
                return 0
    # 금액 0/없음 + 여부 불명 → 불명
    if amt is not None and amt <= 0:
        return 0
    return None


def _recovery_ratio(amount, principal) -> Optional[float]:
    amt = util.to_number(amount)
    prin = util.to_number(principal)
    if amt is None or prin is None or prin <= 0:
        return None
    return util.clamp(amt / prin, 0.0, 3.0)  # 상한(대형/완제 편향 방지)


def parse_feedback_file(path: str, store=None) -> Tuple[pd.DataFrame, List[str]]:
    """피드백 엑셀/CSV 파싱 → 정규화 rows + 경고 목록.

    우리가 생성한 파일(사용자시트+숨김시트)이면 숨김시트로 대출번호/추천월 보강.
    임의 구조 파일도 별칭으로 흡수. 대출번호 없으면 store로 (추천월,고객번호,성명)
    → 대표대출번호 복구 시도.
    """
    warnings: List[str] = []
    ext = path.lower().rsplit(".", 1)[-1]
    if ext in ("xlsx", "xlsm", "xls"):
        sheets = load_excel_all_sheets(path)
    else:
        from .io_loader import load_table
        sheets = {"__csv__": load_table(path)}

    # 사용자 데이터 시트 선택: 금액/여부 별칭이 있는 시트 우선
    user_df = None
    hidden_df = None
    for name, df in sheets.items():
        cols_norm = {_norm_key(c) for c in df.columns}
        has_amt = any(_norm_key(a) in cols_norm for a in _FEEDBACK_ALIASES["실제입금액"])
        has_flag = any(_norm_key(a) in cols_norm for a in _FEEDBACK_ALIASES["실제입금여부"])
        has_key = _norm_key("대출번호") in cols_norm or _norm_key("계좌키") in cols_norm
        has_fv = _norm_key("피처벡터") in cols_norm
        if has_fv or (has_key and not (has_amt or has_flag)):
            hidden_df = df
        if (has_amt or has_flag) and user_df is None:
            user_df = df
    if user_df is None:
        # 아무 시트나 사용
        user_df = next(iter(sheets.values()))

    res = _resolve_columns(user_df)
    n = len(user_df)
    out = pd.DataFrame(index=range(n))

    def col(canon):
        c = res.get(canon)
        if c is not None and c in user_df.columns:
            return user_df[c].reset_index(drop=True)
        return pd.Series([None] * n)

    out["추천월"] = col("추천월")
    out["고객번호"] = col("고객번호").map(util.clean_str)
    out["성명"] = col("성명").map(util.clean_str)
    out["대출번호"] = col("대출번호").map(util.clean_str)
    out["실제입금여부_raw"] = col("실제입금여부")
    out["실제입금액"] = col("실제입금액").map(util.to_number)
    out["입금일자"] = col("입금일자")
    out["원금잔액"] = col("원금잔액").map(util.to_number)
    out["활동여부"] = [
        (1 if util.is_present(v) else (0 if util.clean_str(v) != "" else None))
        for v in col("활동여부")]

    # 숨김시트로 대출번호/추천월 보강 (우리 파일)
    if hidden_df is not None:
        hres = _resolve_columns(hidden_df)
        h_key = hres.get("대출번호")
        h_gonum = hres.get("고객번호")
        h_month = hres.get("추천월")
        if h_key and h_gonum:
            hmap = {}
            for _, hr in hidden_df.iterrows():
                gk = util.clean_str(hr.get(h_gonum))
                hmap[gk] = {
                    "대출번호": util.clean_str(hr.get(h_key)),
                    "추천월": (util.clean_str(hr.get(h_month)) if h_month else ""),
                }
            for i in range(n):
                gk = out.at[i, "고객번호"]
                if not out.at[i, "대출번호"] and gk in hmap:
                    out.at[i, "대출번호"] = hmap[gk]["대출번호"]
                if (out.at[i, "추천월"] in (None, "")) and gk in hmap and hmap[gk]["추천월"]:
                    out.at[i, "추천월"] = hmap[gk]["추천월"]

    # store로 대출번호/추천월 복구
    if store is not None and (out["대출번호"] == "").any():
        recos = store.get_recommendations()
        if len(recos) > 0:
            lut = {}
            for _, rr in recos.iterrows():
                lut[(str(rr["추천월"]), str(rr["고객번호"]), str(rr["성명"]))] = str(rr["대출번호"])
                lut.setdefault((str(rr["고객번호"]), str(rr["성명"])), str(rr["대출번호"]))
            for i in range(n):
                if not out.at[i, "대출번호"]:
                    key3 = (util.clean_str(out.at[i, "추천월"]), out.at[i, "고객번호"], out.at[i, "성명"])
                    key2 = (out.at[i, "고객번호"], out.at[i, "성명"])
                    out.at[i, "대출번호"] = lut.get(key3) or lut.get(key2) or ""

    # 성공여부/회수비율 파생
    succ = []
    rr = []
    for i in range(n):
        s = success_of(out.at[i, "실제입금여부_raw"], out.at[i, "실제입금액"])
        succ.append(s)
        rr.append(_recovery_ratio(out.at[i, "실제입금액"], out.at[i, "원금잔액"]))
    out["성공여부"] = succ
    out["회수비율"] = rr
    out["실제입금여부"] = [s if s is not None else None for s in succ]

    # 입금일자 검증: 추천월 이후만 유효
    valid_dates = []
    bad_date = 0
    for i in range(n):
        dstr = out.at[i, "입금일자"]
        d = util.parse_date(dstr)
        month = util.clean_str(out.at[i, "추천월"])
        if d is not None and month:
            mstart = _month_start(month)
            if mstart is not None and d < mstart:
                bad_date += 1
                valid_dates.append(None)
                continue
        valid_dates.append(d.isoformat() if d is not None else None)
    out["입금일자"] = valid_dates
    if bad_date:
        warnings.append(f"입금일자가 추천월 이전인 {bad_date}건은 무효 처리했습니다.")

    # 결측 안내
    if res.get("실제입금액") is None and res.get("실제입금여부") is None:
        warnings.append("입금액/입금여부 컬럼을 찾지 못했습니다. 성공판정이 제한됩니다.")
    if res.get("원금잔액") is None:
        warnings.append("원금잔액(현재원금) 컬럼이 없어 회수비율 계산이 제한됩니다.")
    if (out["대출번호"] == "").all():
        warnings.append("대출번호(계좌키)를 복구하지 못해 DB 매칭이 어려울 수 있습니다.")

    keep = ["추천월", "고객번호", "성명", "대출번호", "실제입금여부",
            "실제입금액", "입금일자", "원금잔액", "회수비율", "성공여부", "활동여부"]
    return out[keep].copy(), warnings


def _month_start(month: str) -> Optional[_dt.date]:
    m = re.match(r"(\d{4})[.\-/]?(\d{1,2})", str(month).strip())
    if not m:
        return None
    try:
        return _dt.date(int(m.group(1)), int(m.group(2)), 1)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# EB 세그먼트 캘리브레이션
# ---------------------------------------------------------------------------
def _balance_bucket(principal) -> str:
    p = util.to_number(principal)
    if p is None:
        return "미상"
    M = 10_000
    if p <= 100 * M:
        return "~100만"
    if p <= 300 * M:
        return "~300만"
    if p <= 1000 * M:
        return "~1000만"
    if p <= 3000 * M:
        return "~3000만"
    if p <= 5000 * M:
        return "~5000만"
    return "5000만+"


def _status_bucket(status_mid: str) -> str:
    s = util.clean_str(status_mid)
    if s in ("민원", "법조치"):
        return "주의"
    if s in config.STATUS_MID_ALLOW:
        return "안내가능"
    return "기타"


# EB 세그먼트 차원 정의 (등급 제외 — 순환 방지)
_SEGMENT_DIMS = ["회생", "잔액구간", "팀", "부담당자", "나이대", "담보부NPL", "개인사업자", "상태버킷"]


def _row_segments(row) -> Dict[str, str]:
    return {
        "회생": util.clean_str(row.get("회생", "")) or "공란",
        "잔액구간": _balance_bucket(row.get("추천원금잔액", row.get("원금잔액"))),
        "팀": util.clean_str(row.get("팀", "")) or "기타",
        "부담당자": util.clean_str(row.get("부담당자", "")) or config.UNASSIGNED_LABEL,
        "나이대": util.clean_str(row.get("나이대", "")) or "unknown",
        "담보부NPL": "Y" if int(row.get("담보부NPL", 0) or 0) == 1 else "N",
        "개인사업자": "Y" if int(row.get("개인사업자", 0) or 0) == 1 else "N",
        "상태버킷": _status_bucket(row.get("채권상태중", "")),
    }


def build_segment_calibration(feedback_seg: pd.DataFrame,
                              m: int = None) -> dict:
    """feedback+세그먼트 조인 → 차원별 EB 성공률/회수비율.

    반환: {"global":{p,r}, "dims":{dim:{value:{n,succ,p_eb,r_eb}}}}
    """
    if m is None:
        m = config.EB_PRIOR_STRENGTH
    result = {"global": {"p": 0.0, "r": 0.0, "n": 0}, "dims": {}}
    if feedback_seg is None or len(feedback_seg) == 0:
        return result

    succ = pd.to_numeric(feedback_seg["성공여부"], errors="coerce").fillna(0).astype(int)
    rr = pd.to_numeric(feedback_seg["회수비율"], errors="coerce")
    p_global = float(succ.mean()) if len(succ) else 0.0
    r_global = float(rr.dropna().mean()) if rr.notna().any() else 0.0
    result["global"] = {"p": p_global, "r": r_global, "n": int(len(succ))}

    seg_rows = [_row_segments(feedback_seg.iloc[i]) for i in range(len(feedback_seg))]
    for dim in _SEGMENT_DIMS:
        by_val: Dict[str, dict] = {}
        for i in range(len(feedback_seg)):
            v = seg_rows[i][dim]
            d = by_val.setdefault(v, {"n": 0, "succ": 0, "rsum": 0.0, "rn": 0})
            d["n"] += 1
            d["succ"] += int(succ.iloc[i])
            if pd.notna(rr.iloc[i]):
                d["rsum"] += float(rr.iloc[i])
                d["rn"] += 1
        for v, d in by_val.items():
            p_eb = (d["succ"] + m * p_global) / (d["n"] + m)
            r_eb = (d["rsum"] + m * r_global) / (d["rn"] + m) if (d["rn"] + m) > 0 else r_global
            d["p_eb"] = p_eb
            d["r_eb"] = r_eb
        result["dims"][dim] = by_val
    return result


def mixing_weight(n_obs: int) -> float:
    """누적 관측수 → 피드백 혼합비중(0~1). 관측 적으면 규칙/prior 위주."""
    w = 0.0
    for thr, val in config.FEEDBACK_MIX_THRESHOLDS:
        if n_obs >= thr:
            w = val
    return w


def compute_feedback_adj(seg_frame: pd.DataFrame, calibration: dict,
                         mix_weight: float, cap: int = None) -> pd.Series:
    """행별 feedback_adj(additive). 차원별 상대성과 평균 → ±cap, mix_weight 스케일.

    seg_frame 필수 컬럼: 회생, 원금잔액, 팀, 부담당자, 나이대, 담보부NPL(0/1),
    개인사업자(0/1), 채권상태중. 등급은 세그먼트로 쓰지 않는다(순환 방지).
    계좌·차주 어느 레벨에도 동일 계약으로 적용된다.
    """
    if cap is None:
        cap = config.FEEDBACK_ADJ_CAP
    n = len(seg_frame)
    if n == 0 or not calibration or calibration["global"]["n"] == 0 or mix_weight <= 0:
        return pd.Series(np.zeros(n), index=seg_frame.index)

    p_g = calibration["global"]["p"] or 1e-6
    r_g = calibration["global"]["r"] or 1e-6
    dims = calibration["dims"]

    adj = np.zeros(n)
    for i in range(n):
        row = seg_frame.iloc[i]
        seg = _row_segments({
            "회생": row.get("회생", ""),
            "추천원금잔액": row.get("원금잔액"),
            "팀": row.get("팀", ""),
            "부담당자": row.get("부담당자", ""),
            "나이대": row.get("나이대", ""),
            "담보부NPL": int(row.get("담보부NPL", 0) or 0),
            "개인사업자": int(row.get("개인사업자", 0) or 0),
            "채권상태중": row.get("채권상태중", ""),
        })
        vals = []
        for dim in _SEGMENT_DIMS:
            v = seg.get(dim)
            info = dims.get(dim, {}).get(v)
            if not info:
                continue
            rel_p = (info["p_eb"] - p_g) / p_g
            rel_r = (info["r_eb"] - r_g) / r_g
            rel = 0.5 * rel_p + 0.5 * rel_r
            vals.append(util.clamp(rel, -1.0, 1.0))
        if vals:
            mean_rel = float(np.mean(vals))
            adj[i] = util.clamp(cap * mean_rel * mix_weight, -cap, cap)
    return pd.Series(adj, index=seg_frame.index)


def build_segment_frame(act_keep, managers, teams, age_bands, biz_series,
                        collateral_npl) -> pd.DataFrame:
    """계좌 레벨 세그먼트 프레임 구성(compute_feedback_adj 입력)."""
    from .io_loader import get_col
    return pd.DataFrame({
        "회생": get_col(act_keep, "회생").map(util.clean_str).values,
        "원금잔액": [util.to_number(v) for v in get_col(act_keep, "현재원금")],
        "팀": teams.values if hasattr(teams, "values") else teams,
        "부담당자": managers.values if hasattr(managers, "values") else managers,
        "나이대": age_bands.reindex(act_keep.index).fillna("unknown").values,
        "담보부NPL": [int(v) for v in collateral_npl],
        "개인사업자": [int(v) for v in biz_series],
        "채권상태중": get_col(act_keep, "채권상태(중)").map(util.clean_str).values,
    }, index=act_keep.index)


def segment_stats_for_store(calibration: dict) -> List[dict]:
    """segment_stats 테이블 저장용 평탄화."""
    rows = []
    for dim, by_val in calibration.get("dims", {}).items():
        for v, d in by_val.items():
            rows.append({
                "세그먼트키": f"{dim}={v}",
                "관측수": d.get("n", 0),
                "입금성공수": d.get("succ", 0),
                "EB보정율": d.get("p_eb", 0.0),
                "EB회수비율": d.get("r_eb", 0.0),
            })
    return rows
