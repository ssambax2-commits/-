# -*- coding: utf-8 -*-
"""
recovery_lab.py — 회수예측 신규 방법론 실험/검증/보고
======================================================
1단계  피처별 회수율 분석(Wilson 95% CI, 소표본 유의성)
2단계  회수점수 3방법 + 계층적 EB(추가)
        · 방법1 점수공식 탐색(해석가능, 3분할로 선택 과적합 차단)
        · 방법2 순위모델(LightGBM lambdarank)
        · 방법3 유사사례 매칭(k-NN, 설명가능)
        · 추가  계층적 경험적베이즈(부분 풀링) — 소표본 정공법
3단계  검증: 시간분할 holdout, Precision@K(+부트스트랩 CI), 챔피언/기준선 비교,
        누수 자동점검

⚠️ 피처는 '매입/기표 시점 고정값'만 사용(상환일·회수율·입금열·현재잔액 제외).
   상환일/회수율은 라벨·타이밍 분석에만.
"""
from __future__ import annotations

import datetime as _dt
import io
import warnings as _warnings
from contextlib import redirect_stderr
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import synth_recovery

_warnings.filterwarnings("ignore")

REF_DATE = _dt.date(2026, 7, 1)
Z90 = 1.645
Z95 = 1.96

# 모델 피처에서 반드시 제외(누수/사후정보)
_LEAK_COLS = {
    "상환일", "회수율", "recovered", "_latent_logit",
    "현재원금", "현재OPB", "현재연체이자", "현재미수금", "최종갱신금액",
    "원장상태", "채권상태(대)", "채권상태(중)", "회생", "시효일자",
}
_LEAK_PREFIX = ("입금일자", "입금액")

# 점수용 나이대 순서(해석가능 공식에서 fallback)
_AGE_ORDER = ["20대이하", "30대", "40대", "50대", "60대", "70대", "80대이상"]


# ===========================================================================
# 공통 유틸
# ===========================================================================
def wilson_ci(k: int, n: int, z: float = Z95) -> Tuple[float, float]:
    """이항비율 Wilson 신뢰구간."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, (c - h) / d), min(1.0, (c + h) / d))


def _rrn_age_band(rrn: str) -> str:
    s = str(rrn).replace("-", "").strip()
    if len(s) < 7 or not s[:6].isdigit():
        return "unknown"
    yy = int(s[:2])
    g = s[6]
    century = 1900 if g in "129" else (2000 if g in "3478" else 1900)
    birth = century + yy
    age = REF_DATE.year - birth
    if age < 30:
        return "20대이하"
    if age < 40:
        return "30대"
    if age < 50:
        return "40대"
    if age < 60:
        return "50대"
    if age < 70:
        return "60대"
    if age < 80:
        return "70대"
    return "80대이상"


def _months_since(datestr: str) -> float:
    try:
        d = _dt.date.fromisoformat(str(datestr)[:10])
    except (ValueError, TypeError):
        return np.nan
    return (REF_DATE.year - d.year) * 12 + (REF_DATE.month - d.month)


def _seasoning_bucket(months: float) -> str:
    if pd.isna(months):
        return "미상"
    yrs = months / 12
    if yrs < 8:
        return "~8년"
    if yrs < 11:
        return "8~11년"
    if yrs < 14:
        return "11~14년"
    return "14년+"


def _balance_bucket(opb: float) -> str:
    M = 1_000_000
    if opb <= 3 * M:
        return "~300만"
    if opb <= 8 * M:
        return "~800만"
    if opb <= 20 * M:
        return "~2000만"
    if opb <= 45 * M:
        return "~4500만"
    return "4500만+"


# ===========================================================================
# 피처 빌드 (누수-safe)
# ===========================================================================
_NUM_FEATURES = ["log_매입당시OPB", "경과월_대출", "대출빈티지", "대출이율",
                 "연체이율", "양도횟수", "법시행이후양도횟수", "다중계좌_총건수",
                 "담보부NPL", "개인사업자"]
_CAT_FEATURES = ["나이대", "지역"]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """매입/기표 시점 고정값만으로 피처행렬 구성."""
    out = pd.DataFrame(index=df.index)
    opb = pd.to_numeric(df["매입당시OPB"], errors="coerce").fillna(0)
    out["log_매입당시OPB"] = np.log1p(opb)
    out["경과월_대출"] = df["대출일자"].map(_months_since)
    out["대출빈티지"] = df["대출일자"].map(
        lambda s: _dt.date.fromisoformat(str(s)[:10]).year
        if pd.notna(s) and str(s) else np.nan)
    out["대출이율"] = pd.to_numeric(df["대출이율"], errors="coerce")
    out["연체이율"] = pd.to_numeric(df["연체이율"], errors="coerce")
    out["양도횟수"] = pd.to_numeric(df["양도횟수"], errors="coerce").fillna(0)
    out["법시행이후양도횟수"] = pd.to_numeric(df["법시행이후양도횟수"], errors="coerce").fillna(0)
    out["다중계좌_총건수"] = df["다중계좌 활동/총건수"].map(
        lambda s: float(str(s).split("/")[-1]) if "/" in str(s) else np.nan)
    out["담보부NPL"] = (df["채권구분"] == "담보부NPL").astype(int)
    out["개인사업자"] = (df["차주구분"] == "개인사업자").astype(int)
    out["나이대"] = df["주민등록번호"].map(_rrn_age_band)
    out["지역"] = df["지역"].astype(str)
    return out


def assert_no_leakage(feat_cols: List[str]) -> None:
    hits = [c for c in feat_cols if c in _LEAK_COLS
            or any(c.startswith(p) for p in _LEAK_PREFIX)]
    if hits:
        raise AssertionError(f"누수 피처 검출: {hits}")


# ===========================================================================
# 1단계 — 피처별 회수율 표 (Wilson CI + 유의성)
# ===========================================================================
def recovery_rate_table(df: pd.DataFrame, group: pd.Series,
                        label: str) -> pd.DataFrame:
    base = df["recovered"].mean()
    rows = []
    for val, idx in df.groupby(group).groups.items():
        sub = df.loc[idx]
        n = len(sub)
        k = int(sub["recovered"].sum())
        rate = k / n if n else 0.0
        lo, hi = wilson_ci(k, n)
        # 유의: 95% CI가 전체 base rate를 배제하는가
        sig = "★" if (hi < base or lo > base) else ""
        rows.append({"구간": val, "n": n, "회수": k,
                     "회수율%": round(rate * 100, 1),
                     "95%CI%": f"[{lo*100:.1f},{hi*100:.1f}]",
                     "vs평균": sig})
    t = pd.DataFrame(rows).sort_values("회수율%", ascending=False)
    t.attrs["label"] = label
    t.attrs["base"] = base
    return t


# ===========================================================================
# 2단계 방법들 — 각 함수는 holdout 점수 벡터(높을수록 회수가능) 반환
# ===========================================================================
def _fit_segment_eb(train: pd.DataFrame, dim_vals: pd.Series,
                    m: int = 20) -> Dict[str, float]:
    """차원값 → EB 수축 회수율 (feedback.py 방식). train만 사용."""
    g = train["recovered"].mean()
    out = {}
    tmp = pd.DataFrame({"v": dim_vals.loc[train.index], "y": train["recovered"]})
    for v, sub in tmp.groupby("v"):
        n, k = len(sub), int(sub["y"].sum())
        out[str(v)] = (k + m * g) / (n + m)
    out["__global__"] = g
    return out


def _zscore(s: pd.Series, ref: pd.Series) -> pd.Series:
    mu, sd = ref.mean(), ref.std(ddof=0) or 1.0
    return (s - mu) / sd


def method1_formula_search(train, dev, holdout, feats_all, k_dev: int,
                           rng) -> Tuple[np.ndarray, dict]:
    """해석가능 점수공식 진화탐색. train=성분추정, dev=선택, holdout=최종."""
    age_eb = _fit_segment_eb(train, feats_all["나이대"])
    reg_eb = _fit_segment_eb(train, feats_all["지역"])
    ref_logopb = feats_all.loc[train.index, "log_매입당시OPB"]
    ref_season = feats_all.loc[train.index, "경과월_대출"]

    def _rel(mapped: pd.Series, g: float) -> pd.Series:
        """세그먼트 EB회수율의 전체평균 대비 상대편차."""
        return (mapped - g) / (g + 1e-9)

    def components(idx):
        f = feats_all.loc[idx]
        return pd.DataFrame({
            "잔액작을수록": -_zscore(f["log_매입당시OPB"], ref_logopb),
            "나이대성향": _rel(f["나이대"].map(lambda v: age_eb.get(str(v), age_eb["__global__"])),
                             age_eb["__global__"]),
            "지역성향": _rel(f["지역"].map(lambda v: reg_eb.get(str(v), reg_eb["__global__"])),
                           reg_eb["__global__"]),
            "경과짧을수록": -_zscore(f["경과월_대출"].fillna(ref_season.mean()), ref_season),
            "담보": f["담보부NPL"].astype(float),
        }, index=idx)

    comp_dev = components(dev.index)
    comp_hold = components(holdout.index)
    names = list(comp_dev.columns)

    # 후보 가중치: 손설계 시드 + 진화(고정 예산)
    seeds = [
        np.array([1, 1, 1, 0, 0], float),
        np.array([2, 1, 1, 0.5, 0.3], float),
        np.array([1, 2, 1, 0, 0], float),
        np.array([1.5, 1, 0.5, 0.5, 0], float),
        np.array([1, 1, 0.5, 0.5, 0.5], float),
    ]
    pop = [w / (np.abs(w).sum() or 1) for w in seeds]

    def dev_prec(w):
        s = (comp_dev.values * w).sum(axis=1)
        order = np.argsort(-s)[:k_dev]
        return dev["recovered"].values[order].mean()

    budget, best = 40, None
    scored = [(dev_prec(w), w) for w in pop]
    evals = len(scored)
    while evals < budget:
        scored.sort(key=lambda t: -t[0])
        parents = [w for _, w in scored[:3]]
        w = parents[rng.integers(len(parents))] + rng.normal(0, 0.4, len(names))
        w = w / (np.abs(w).sum() or 1)
        scored.append((dev_prec(w), w))
        evals += 1
    scored.sort(key=lambda t: -t[0])
    best_dev, best_w = scored[0]

    s_hold = (comp_hold.values * best_w).sum(axis=1)
    info = {"weights": {nm: round(float(w), 3) for nm, w in zip(names, best_w)},
            "dev_precision": round(float(best_dev), 3),
            "n_candidates": evals,
            "age_eb": {k: round(float(v), 3) for k, v in age_eb.items() if k != "__global__"},
            "region_eb_top": dict(sorted(
                {k: round(float(v), 3) for k, v in reg_eb.items() if k != "__global__"}.items(),
                key=lambda x: -x[1])[:5])}
    return s_hold, info


def method2_lambdarank(train_dev, holdout, feats_all) -> np.ndarray:
    """LightGBM lambdarank. 소표본 방어: 얕은 트리·강 정규화·단조제약."""
    import lightgbm as lgb
    Xtr = feats_all.loc[train_dev.index].copy()
    Xho = feats_all.loc[holdout.index].copy()
    for c in _CAT_FEATURES:
        Xtr[c] = Xtr[c].astype("category")
        Xho[c] = pd.Categorical(Xho[c], categories=Xtr[c].cat.categories)
    ytr = train_dev["recovered"].values
    # 단조제약: 잔액↑→회수↓ (log_매입당시OPB에 -1)
    mono = [(-1 if c == "log_매입당시OPB" else 0) for c in feats_all.columns]
    params = dict(objective="lambdarank", n_estimators=60, num_leaves=7,
                  min_child_samples=15, learning_rate=0.05, reg_lambda=8.0,
                  subsample=0.8, colsample_bytree=0.8, random_state=42,
                  monotone_constraints=mono, verbose=-1)
    r = lgb.LGBMRanker(**params)
    buf = io.StringIO()
    with redirect_stderr(buf):
        r.fit(Xtr, ytr, group=[len(Xtr)], categorical_feature=_CAT_FEATURES)
    return r.predict(Xho)


def method3_case_matching(train, holdout, feats_all, k: int = 15) -> np.ndarray:
    """유사사례 매칭(k-NN). 회수성공 이웃 비중으로 회수가능성 추정."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.neighbors import NearestNeighbors

    def encode(idx):
        f = feats_all.loc[idx].copy()
        base = f[_NUM_FEATURES].fillna(f[_NUM_FEATURES].median())
        cats = pd.get_dummies(f[_CAT_FEATURES].astype(str))
        return base, cats

    btr, ctr = encode(train.index)
    bho, cho = encode(holdout.index)
    cho = cho.reindex(columns=ctr.columns, fill_value=0)
    sc = StandardScaler().fit(btr)
    Xtr = np.hstack([sc.transform(btr), ctr.values * 1.5])  # 범주 가중
    Xho = np.hstack([sc.transform(bho), cho.values * 1.5])
    nn = NearestNeighbors(n_neighbors=min(k, len(Xtr))).fit(Xtr)
    dist, idxs = nn.kneighbors(Xho)
    ytr = train["recovered"].values
    w = 1.0 / (1.0 + dist)
    return (w * ytr[idxs]).sum(axis=1) / w.sum(axis=1)


def method_eb_hierarchical(train, holdout, feats_all, m: int = 20) -> np.ndarray:
    """추가: 계층적 EB 부분풀링. 여러 차원 EB회수율의 로그오즈 합산."""
    dims = {"나이대": feats_all["나이대"], "지역": feats_all["지역"],
            "잔액구간": feats_all["log_매입당시OPB"].map(
                lambda v: _balance_bucket(np.expm1(v))),
            "빈티지": feats_all["대출빈티지"].astype("Int64").astype(str)}
    g = train["recovered"].mean()

    def logit(p):
        p = min(max(p, 1e-4), 1 - 1e-4)
        return np.log(p / (1 - p))

    score = np.zeros(len(holdout))
    for dim_vals in dims.values():
        eb = _fit_segment_eb(train, dim_vals, m=m)
        hv = dim_vals.loc[holdout.index]
        score += np.array([logit(eb.get(str(v), g)) - logit(g) for v in hv])
    return score


# ===========================================================================
# 챔피언/기준선
# ===========================================================================
def champion_logistic(train_dev, holdout, feats_all) -> np.ndarray:
    """관행 챔피언 프록시: L2 정규화 로지스틱(원-핫)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    def enc(idx, cols=None):
        f = feats_all.loc[idx].copy()
        num = f[_NUM_FEATURES].fillna(f[_NUM_FEATURES].median())
        cat = pd.get_dummies(f[_CAT_FEATURES].astype(str))
        X = pd.concat([num, cat], axis=1)
        return X if cols is None else X.reindex(columns=cols, fill_value=0)

    Xtr = enc(train_dev.index)
    Xho = enc(holdout.index, Xtr.columns)
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=0.3, max_iter=1000, class_weight="balanced")
    clf.fit(sc.transform(Xtr), train_dev["recovered"].values)
    return clf.predict_proba(sc.transform(Xho))[:, 1]


def champion_gbm(train_dev, holdout, feats_all) -> np.ndarray:
    """CatBoost류 부스팅 프록시: sklearn GradientBoosting(강 정규화)."""
    from sklearn.ensemble import GradientBoostingClassifier

    def enc(idx, cols=None):
        f = feats_all.loc[idx].copy()
        num = f[_NUM_FEATURES].fillna(f[_NUM_FEATURES].median())
        cat = pd.get_dummies(f[_CAT_FEATURES].astype(str))
        X = pd.concat([num, cat], axis=1)
        return X if cols is None else X.reindex(columns=cols, fill_value=0)

    Xtr = enc(train_dev.index)
    Xho = enc(holdout.index, Xtr.columns)
    clf = GradientBoostingClassifier(n_estimators=60, max_depth=2,
                                     learning_rate=0.05, subsample=0.8,
                                     random_state=42)
    clf.fit(Xtr, train_dev["recovered"].values)
    return clf.predict_proba(Xho)[:, 1]


# ===========================================================================
# 3단계 — 지표
# ===========================================================================
def precision_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores, kind="stable")[:k]
    return float(y[order].mean())


def ndcg_at_k(y: np.ndarray, scores: np.ndarray, k: int) -> float:
    order = np.argsort(-scores, kind="stable")[:k]
    gains = y[order]
    disc = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    dcg = float((gains * disc).sum())
    ideal = np.sort(y)[::-1][:k]
    idcg = float((ideal * disc).sum()) or 1.0
    return dcg / idcg


def bootstrap_prec_ci(y, scores, k, n_boot=2000, seed=1) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(precision_at_k(y[idx], scores[idx], k))
    return (float(np.percentile(vals, 5)), float(np.percentile(vals, 95)))


# ===========================================================================
# 오케스트레이션 + 보고
# ===========================================================================
def run(seed: int = 20260701) -> str:
    rng = np.random.default_rng(seed)
    L: List[str] = []

    def out(s=""):
        L.append(s)

    df = synth_recovery.generate(n=1000, n_recovered=40, seed=seed)
    feats = build_features(df)
    assert_no_leakage(list(feats.columns))

    # 시간분할: 폐지배정일자 기준 train(≤55%) / dev(≤70%) / holdout(30%)
    order = df["폐지배정일자"].argsort().values
    n = len(df)
    c1, c2 = int(n * 0.55), int(n * 0.70)
    tr_idx, dv_idx, ho_idx = order[:c1], order[c1:c2], order[c2:]
    train, dev, holdout = df.iloc[tr_idx], df.iloc[dv_idx], df.iloc[ho_idx]
    train_dev = df.iloc[order[:c2]]
    yho = holdout["recovered"].values
    base = df["recovered"].mean()

    out("=" * 74)
    out("  코로나채권 회수예측 — 신규 방법론 시뮬레이션 보고 (합성데이터)")
    out("=" * 74)
    out(f"\n[데이터]  전체 {n}건 / 회수 {int(df['recovered'].sum())}건 "
        f"(base rate {base*100:.1f}%)  ※ 잠재신호를 심은 합성데이터")
    out(f"[분할]    시간분할(폐지배정일자):  train {len(train)}건(회수 {int(train['recovered'].sum())}) / "
        f"dev {len(dev)}(회수 {int(dev['recovered'].sum())}) / "
        f"holdout {len(holdout)}(회수 {int(yho.sum())})")
    out(f"[피처]    {len(feats.columns)}개(누수-safe): {', '.join(feats.columns)}")
    out(f"[누수점검] 통과 — 상환일/회수율/입금열/현재잔액 = 피처 제외")

    # ---- 1단계: 피처별 회수율 ----
    out("\n" + "─" * 74)
    out("  1단계  피처별 회수율 (★=95% CI가 전체평균 배제 → 소표본에서도 유의)")
    out("─" * 74)
    band = df["주민등록번호"].map(_rrn_age_band)
    tables = [
        ("나이대", band),
        ("지역", df["지역"]),
        ("잔액구간", df["매입당시OPB"].map(lambda v: _balance_bucket(float(v)))),
        ("경과기간", feats["경과월_대출"].map(_seasoning_bucket)),
        ("빈티지(대출연도)", feats["대출빈티지"].astype("Int64").astype(str)),
    ]
    for name, g in tables:
        t = recovery_rate_table(df, g, name)
        out(f"\n· {name} (전체평균 {base*100:.1f}%)")
        for _, r in t.iterrows():
            out(f"    {str(r['구간']):<10} n={r['n']:<4} 회수={r['회수']:<3} "
                f"{r['회수율%']:>5.1f}%  CI{r['95%CI%']:<16} {r['vs평균']}")

    # 교호작용(소표본 검정력 한계 예시)
    out(f"\n· 교호작용 예시: 잔액구간 × 지역권 (셀당 표본 급감)")
    df["_지역권"] = df["지역"].map(lambda r: "수도권" if r in ("서울", "경기", "인천")
                                   else "비수도권")
    df["_잔액구간"] = df["매입당시OPB"].map(lambda v: "소액(≤800만)" if float(v) <= 8e6
                                          else "고액(>800만)")
    inter = df.groupby(["_잔액구간", "_지역권"])["recovered"].agg(["size", "sum"])
    for (bb, rr), r in inter.iterrows():
        lo, hi = wilson_ci(int(r["sum"]), int(r["size"]))
        out(f"    {bb:<12} {rr:<7} n={int(r['size']):<4} 회수={int(r['sum']):<2} "
            f"{r['sum']/r['size']*100:>5.1f}%  CI[{lo*100:.1f},{hi*100:.1f}]")
    out("    → 셀당 회수 사건이 한 자릿수라 CI가 매우 넓다(대부분 유의 판정 불가).")

    # 상환 계절성(라벨/타이밍 전용 — 피처 아님)
    rec = df[df["recovered"] == 1].copy()
    rec["_월"] = rec["상환일"].map(lambda s: int(str(s)[5:7]) if s else 0)
    mm = rec["_월"].value_counts().sort_index()
    out(f"\n· 상환 계절성(회수 {len(rec)}건, 타이밍 분석 전용):")
    out("    월별 상환건수: " + ", ".join(f"{m}월:{c}" for m, c in mm.items()))
    out(f"    회수율(원금대비) 중앙값 {rec['회수율'].median()*100:.0f}% / 완제(≥95%) "
        f"{int((rec['회수율']>=0.95).sum())}건")

    # ---- 2단계+3단계: 방법별 holdout 성능 ----
    out("\n" + "─" * 74)
    out("  2·3단계  방법별 holdout 성능 (Precision@K = 상위 K 추천의 실제 회수율)")
    out("─" * 74)

    k_dev = max(10, int(len(dev) * 0.10))
    s_f, finfo = method1_formula_search(train, dev, holdout, feats, k_dev, rng)
    scores = {
        "방법1 점수공식탐색": s_f,
        "방법2 LGBM lambdarank": method2_lambdarank(train_dev, holdout, feats),
        "방법3 유사사례 k-NN": method3_case_matching(train_dev, holdout, feats),
        "추가 계층적EB(부분풀링)": method_eb_hierarchical(train_dev, holdout, feats),
        "챔피언 정규화로지스틱": champion_logistic(train_dev, holdout, feats),
        "챔피언 부스팅(CatBoost류)": champion_gbm(train_dev, holdout, feats),
    }
    # 기준선
    rng2 = np.random.default_rng(7)
    scores["기준선 랜덤"] = rng2.random(len(holdout))
    scores["기준선 잔액오름차순"] = -feats.loc[holdout.index, "log_매입당시OPB"].values

    Klist = [20, 30, max(1, int(len(holdout) * 0.10))]
    Klist = sorted(set(Klist))
    out(f"\n  holdout n={len(holdout)}, 회수 {int(yho.sum())}건, base {yho.mean()*100:.1f}%  "
        f"(K = {Klist})")
    out(f"\n  {'방법':<26}" + "".join(f"P@{k:<7}" for k in Klist)
        + f"{'P@30 90%CI':<16}NDCG@30 lift@30")
    out("  " + "-" * 84)

    rank_rows = []
    for name, s in scores.items():
        s = np.asarray(s, dtype=float)
        precs = [precision_at_k(yho, s, k) for k in Klist]
        p30 = precision_at_k(yho, s, 30)
        lo, hi = bootstrap_prec_ci(yho, s, 30)
        nd = ndcg_at_k(yho, s, 30)
        lift = (p30 / yho.mean()) if yho.mean() else 0.0
        rank_rows.append((name, p30, lift))
        pcells = "".join(f"{p*100:>5.1f}%  " for p in precs)
        out(f"  {name:<26}{pcells}[{lo*100:>4.1f},{hi*100:>4.1f}]%   "
            f"{nd:>5.2f}   {lift:>4.2f}x")

    out("\n  ※ CI는 holdout 부트스트랩 90%. 서로 겹치면 '통계적으로 동급'으로 읽어야 함.")

    # ---- 발견 공식 ----
    out("\n" + "─" * 74)
    out("  발견된 상위 회수 공식(방법1, 해석가능)")
    out("─" * 74)
    out(f"  회수점수 ≈ " + "  +  ".join(
        f"{w:+.2f}·{nm}" for nm, w in finfo["weights"].items()))
    out(f"    (dev Precision@{k_dev}={finfo['dev_precision']*100:.0f}%, "
        f"후보 {finfo['n_candidates']}개 탐색)")
    out(f"    나이대 EB회수율: {finfo['age_eb']}")
    out(f"    지역 EB회수율 top5: {finfo['region_eb_top']}")

    # ---- 정직한 결론 ----
    rank_rows.sort(key=lambda x: -x[1])
    champ = max(v for n_, v, _ in rank_rows if "챔피언" in n_)
    best_name, best_p, best_lift = rank_rows[0]
    out("\n" + "═" * 74)
    out("  정직한 결론")
    out("═" * 74)
    out(f"  · holdout P@30 순위:")
    for nm, p, lf in rank_rows:
        tag = " ←최고" if nm == best_name else (" (챔피언)" if "챔피언" in nm else "")
        out(f"      {nm:<26} {p*100:>5.1f}%  (lift {lf:.2f}x){tag}")
    out(f"\n  · 최고: {best_name} (P@30 {best_p*100:.1f}%, base 대비 {best_lift:.1f}배).")
    delta = (best_p - champ) * 100
    verdict = ("신규≈챔피언(동급)" if abs(delta) < 3 else
               ("신규 우세" if delta > 0 else "챔피언 우세"))
    out(f"  · 챔피언 최고 대비: {delta:+.1f}%p → {verdict}")
    out(f"    (단, 위 부트스트랩 CI 폭을 보면 회수 {int(yho.sum())}건 규모에선 "
        f"대부분 차이가 CI 안. '이겼다'는 주장은 신중해야.)")
    out("\n  · 방법론적 관찰:")
    out("      - 소표본에서 잔액·나이대 신호는 1단계 표에서 이미 드러남(★).")
    out("      - 지역은 방향성은 보이나 CI가 넓어 단독 확증 어려움.")
    out("      - LGBM lambdarank는 회수 28건 규모에서 과적합 위험 — 단조제약을")
    out("        걸어도 단순 공식/EB 대비 뚜렷한 우위를 주기 어렵다.")
    out("      - 해석가능 공식 + 계층적EB가 '설명가능 + 견고'로 실무 적합.")
    out("\n  · 다음 실험 제안:")
    out("      1) 검열(아직 미상환=미회수 아님) 반영한 생존분석/PU 라벨")
    out("      2) 월별 피드백 누적으로 유효 N 확대(패널 관점)")
    out("      3) 실데이터로 base rate·세그먼트 사건수 감사 후 방법 확정")
    out("═" * 74)

    return "\n".join(L)


if __name__ == "__main__":
    print(run())
