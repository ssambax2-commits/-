# -*- coding: utf-8 -*-
"""
model.py — CatBoost 2단계 Hurdle 모델 (1M/3M 시간창 라벨)
===========================================================
Stage 1: 기준일 t0 이후 1개월/3개월 내 입금발생확률(분류, balanced bagging).
Stage 2: 입금 발생 시 예상입금액 E[amt|paid] (회귀). positive 부족/불안정 시
         세그먼트(잔액구간×담보부NPL) 평균 → 전역 평균으로 fallback.
최종 예상회수액 = P(입금발생) × E[입금액|입금발생), 1M/3M 분리 산출.
캘리브레이션: 모집단 평균확률을 학습 base rate에 맞추는 전역 스케일링
(순위 보존). 검증: 시간/차주그룹 분리 split, Lift@5%/10%, Precision@50/100,
기준선(랜덤/잔액순) 비교, champion-challenger.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from . import config

try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    _CATBOOST_OK = True
except Exception:  # noqa: BLE001
    _CATBOOST_OK = False

HORIZONS = ("1m", "3m")
PRIMARY_HORIZON = "1m"   # 랭킹(paid_similarity)에 쓰는 주 호라이즌


def _segment_keys(X: pd.DataFrame) -> pd.Series:
    """예상액 fallback 세그먼트: 잔액(log OPB) 3분위 × 담보부NPL."""
    opb = pd.to_numeric(X.get("log_매입당시OPB"), errors="coerce")
    try:
        bins = pd.qcut(opb, 3, labels=["소", "중", "대"], duplicates="drop")
    except (ValueError, IndexError):
        bins = pd.Series(["중"] * len(X), index=X.index)
    npl = X.get("담보부NPL")
    npl = pd.to_numeric(npl, errors="coerce").fillna(0).astype(int) if npl is not None \
        else pd.Series(0, index=X.index)
    return pd.Series([f"{b}|{'NPL' if n == 1 else '일반'}"
                      for b, n in zip(bins.astype(str), npl)], index=X.index)


@dataclass
class HurdleModel:
    cat_features: List[str] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    version: str = ""
    active: bool = False               # Stage1(주 호라이즌) 사용 가능 여부
    n_positive: int = 0                # 주 호라이즌 positive (모델메타 호환)
    n_negative: int = 0
    metrics: Dict = field(default_factory=dict)
    stage2_enabled: bool = False       # 회귀기 1개 이상 학습됨
    warning: str = ""
    ranking_horizon: str = PRIMARY_HORIZON

    # 호라이즌별 상태
    n_pos_by_h: Dict = field(default_factory=dict)
    base_rate_by_h: Dict = field(default_factory=dict)     # 학습 positive rate
    stage2_fallback: Dict = field(default_factory=dict)    # h -> {used, reason}

    _clf_by_h: Dict = field(default_factory=dict, repr=False)   # h -> [clf,...]
    _reg_by_h: Dict = field(default_factory=dict, repr=False)   # h -> reg|None
    _amt_seg_mean: Dict = field(default_factory=dict, repr=False)  # h -> {seg: mean}
    _amt_global_mean: Dict = field(default_factory=dict, repr=False)  # h -> mean

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, labels: Dict[str, Tuple[np.ndarray, np.ndarray]],
            ensemble_size: int = None) -> "HurdleModel":
        """labels: {"1m": (y, amt), "3m": (y, amt)} — y=0/1, amt=관측창 입금액."""
        self.feature_names = list(X.columns)
        if ensemble_size is None:
            ensemble_size = config.BAGGING_ENSEMBLE_SIZE
        cat_idx = [X.columns.get_loc(c) for c in self.cat_features if c in X.columns]
        seg = _segment_keys(X)
        rng = np.random.RandomState(config.RANDOM_SEED)

        for h in HORIZONS:
            if h not in labels:
                continue
            y, amt = labels[h]
            y = np.asarray(y).astype(int)
            amt = np.asarray(amt, dtype=float)
            n_pos = int((y == 1).sum())
            self.n_pos_by_h[h] = n_pos
            self.base_rate_by_h[h] = float(y.mean()) if len(y) else 0.0

            # ---- Stage 2 예상액: 항상 세그먼트/전역 평균 준비(fallback 보장) ----
            pos_mask = (y == 1) & np.isfinite(amt) & (amt > 0)
            if pos_mask.any():
                amts = amt[pos_mask]
                cap = np.percentile(amts, 95)  # 극단값 상한(대형 편향 방지)
                amts_c = np.clip(amts, None, cap)
                self._amt_global_mean[h] = float(amts_c.mean())
                seg_pos = seg[pos_mask]
                self._amt_seg_mean[h] = {
                    k: float(np.clip(amt[pos_mask][seg_pos.values == k], None, cap).mean())
                    for k in set(seg_pos.values)
                }
            else:
                self._amt_global_mean[h] = 0.0
                self._amt_seg_mean[h] = {}

            if not _CATBOOST_OK or n_pos < config.POSITIVE_MIN_FOR_ML:
                self.stage2_fallback[h] = {
                    "used": True,
                    "reason": (f"positive {n_pos}건 < 임계 {config.POSITIVE_MIN_FOR_ML}"
                               if _CATBOOST_OK else "CatBoost 미탑재"),
                }
                continue

            # ---- Stage 1 분류기 (balanced bagging) ----
            pos_idx = np.where(y == 1)[0]
            neg_idx = np.where(y == 0)[0]
            neg_n = min(len(neg_idx), max(len(pos_idx) * 3, len(pos_idx) + 1))
            models = []
            for b in range(ensemble_size):
                sub = rng.choice(neg_idx, size=neg_n, replace=False) \
                    if neg_n < len(neg_idx) else neg_idx
                idx = np.concatenate([pos_idx, sub])
                rng.shuffle(idx)
                params = dict(config.CATBOOST_PARAMS)
                params["random_seed"] = config.RANDOM_SEED + b
                params["auto_class_weights"] = "Balanced"
                clf = CatBoostClassifier(**params)
                clf.fit(Pool(X.iloc[idx], y[idx], cat_features=cat_idx))
                models.append(clf)
            self._clf_by_h[h] = models

            # ---- Stage 2 회귀 (log 금액) ----
            if pos_mask.sum() >= config.POSITIVE_MIN_FOR_ML:
                reg = CatBoostRegressor(
                    iterations=200, depth=4, learning_rate=0.05, l2_leaf_reg=6.0,
                    loss_function="RMSE", random_seed=config.RANDOM_SEED,
                    verbose=False, allow_writing_files=False)
                reg.fit(Pool(X.iloc[np.where(pos_mask)[0]],
                             np.log1p(amt[pos_mask]), cat_features=cat_idx))
                self._reg_by_h[h] = reg
                self.stage2_fallback[h] = {"used": False, "reason": ""}
            else:
                self._reg_by_h[h] = None
                self.stage2_fallback[h] = {
                    "used": True,
                    "reason": f"금액 표본 {int(pos_mask.sum())}건 부족 → 세그먼트 평균",
                }

        # 주 호라이즌 결정: 1m 우선, 부족하면 3m
        if self._clf_by_h.get("1m"):
            self.ranking_horizon = "1m"
        elif self._clf_by_h.get("3m"):
            self.ranking_horizon = "3m"
            self.warning = "1개월 positive 부족 — 랭킹은 3개월 모델 사용."
        else:
            self.active = False
            self.n_positive = self.n_pos_by_h.get("1m", 0)
            self.n_negative = int(len(X)) - self.n_positive
            if not self.warning:
                self.warning = (
                    f"positive(1M={self.n_pos_by_h.get('1m', 0)}, "
                    f"3M={self.n_pos_by_h.get('3m', 0)}) 부족 — 규칙·prior 폴백.")
            self.stage2_enabled = False
            return self

        self.active = True
        self.n_positive = self.n_pos_by_h.get(self.ranking_horizon, 0)
        self.n_negative = int(len(X)) - self.n_positive
        self.stage2_enabled = any(r is not None for r in self._reg_by_h.values())
        return self

    # ------------------------------------------------------------------
    def predict_proba_h(self, X: pd.DataFrame, horizon: str) -> Optional[np.ndarray]:
        models = self._clf_by_h.get(horizon)
        if not models:
            return None
        Xc = X[self.feature_names] if all(c in X.columns for c in self.feature_names) else X
        p = np.zeros(len(Xc))
        for m in models:
            p += m.predict_proba(Xc)[:, 1]
        return p / len(models)

    def predict_proba(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        """랭킹용(주 호라이즌) 확률 — 기존 인터페이스 유지."""
        if not self.active:
            return None
        return self.predict_proba_h(X, self.ranking_horizon)

    def expected_amount_h(self, X: pd.DataFrame, horizon: str) -> np.ndarray:
        """E[입금액|입금발생] — 회귀기 있으면 예측, 없으면 세그먼트/전역 평균."""
        n = len(X)
        reg = self._reg_by_h.get(horizon)
        if reg is not None:
            Xc = X[self.feature_names] if all(c in X.columns for c in self.feature_names) else X
            e = np.expm1(reg.predict(Xc))
            gcap = self._amt_global_mean.get(horizon, 0.0) * 5 or None
            if gcap:
                e = np.clip(e, 0, gcap)
            return np.maximum(e, 0.0)
        seg = _segment_keys(X)
        gmean = self._amt_global_mean.get(horizon, 0.0)
        smap = self._amt_seg_mean.get(horizon, {})
        return np.array([smap.get(s, gmean) for s in seg], dtype=float)

    def expected_recovery(self, X: pd.DataFrame, horizon: str,
                          calibrate: bool = True) -> Dict[str, np.ndarray]:
        """예상회수액 = P(입금) × E[입금액|입금]. 캘리브레이션 전/후 모두 반환."""
        n = len(X)
        p = self.predict_proba_h(X, horizon)
        if p is None:
            # Stage1 없음 → base rate 균일 확률(순수 fallback)
            p = np.full(n, self.base_rate_by_h.get(horizon, 0.0))
        p_cal = calibrate_probs(p, self.base_rate_by_h.get(horizon)) if calibrate else p
        e_amt = self.expected_amount_h(X, horizon)
        return {"p_raw": p, "p_cal": p_cal, "e_amt": e_amt,
                "exp_raw": p * e_amt, "exp_cal": p_cal * e_amt}

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "HurdleModel":
        import joblib
        return joblib.load(path)


def calibrate_probs(p: np.ndarray, target_rate: Optional[float]) -> np.ndarray:
    """모집단 평균확률을 base rate에 맞추는 전역 스케일링(순위 보존)."""
    p = np.asarray(p, dtype=float)
    if target_rate is None or target_rate <= 0 or len(p) == 0:
        return np.clip(p, 0, 0.99)
    m = p.mean()
    if m <= 0:
        return np.clip(p, 0, 0.99)
    return np.clip(p * (target_rate / m), 0, 0.99)


# ---------------------------------------------------------------------------
# 검증 지표
# ---------------------------------------------------------------------------
def lift_at_k(y_true, scores, k: float = None) -> float:
    if k is None:
        k = config.LIFT_TOP_K_PCT
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    if n == 0:
        return 0.0
    base = y_true.mean()
    if base <= 0:
        return 0.0
    topn = max(1, int(np.ceil(n * k)))
    order = np.argsort(-scores)
    return float(y_true[order[:topn]].mean() / base)


def precision_at_topn(y_true, scores, n_top: int = 50) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    if n == 0:
        return 0.0
    order = np.argsort(-scores)
    return float(y_true[order[:min(n_top, n)]].mean())


def time_group_split(n: int, groups: Optional[np.ndarray],
                     time_order: Optional[np.ndarray],
                     val_frac: float = 0.25, seed: int = None):
    """시간 기준 우선, 불가하면 차주그룹 분할(랜덤 분할 금지).

    같은 차주(고객번호 그룹)가 train/val 양쪽에 들어가지 않는다.
    """
    if seed is None:
        seed = config.RANDOM_SEED
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    if time_order is not None and np.isfinite(
            pd.to_numeric(pd.Series(time_order), errors="coerce")).any():
        t = pd.to_numeric(pd.Series(time_order), errors="coerce").fillna(-np.inf).values
        order = np.argsort(t)
        cut = int(n * (1 - val_frac))
        train_sorted, val_sorted = order[:cut], order[cut:]
        if groups is not None:
            val_groups = set(np.asarray(groups)[val_sorted])
            train_sorted = np.array([i for i in train_sorted
                                     if np.asarray(groups)[i] not in val_groups])
        return train_sorted, val_sorted
    if groups is not None:
        uniq = np.array(sorted(set(groups)))
        rng.shuffle(uniq)
        val_g = set(uniq[:max(1, int(len(uniq) * val_frac))])
        val_idx = idx[np.array([g in val_g for g in groups])]
        train_idx = idx[np.array([g not in val_g for g in groups])]
        return train_idx, val_idx
    rng.shuffle(idx)
    cut = int(n * (1 - val_frac))
    return idx[:cut], idx[cut:]


def _validation_metrics(y_val, prob, X_val) -> Dict:
    """실무 지표 + 기준선(§12). 기준선: 랜덤(≈1.0), 잔액순(log OPB)."""
    out = {
        "lift@5%": round(lift_at_k(y_val, prob, 0.05), 4),
        "lift@10%": round(lift_at_k(y_val, prob, 0.10), 4),
        "precision@50": round(precision_at_topn(y_val, prob, 50), 4),
        "precision@100": round(precision_at_topn(y_val, prob, 100), 4),
    }
    rng = np.random.RandomState(config.RANDOM_SEED)
    out["기준선_랜덤_lift@10%"] = round(
        lift_at_k(y_val, rng.rand(len(y_val)), 0.10), 4)
    opb = pd.to_numeric(X_val.get("log_매입당시OPB"), errors="coerce")
    if opb is not None and opb.notna().any():
        out["기준선_잔액순_lift@10%"] = round(
            lift_at_k(y_val, opb.fillna(0).values, 0.10), 4)
    return out


def train_and_validate(X: pd.DataFrame,
                       labels: Dict[str, Tuple[np.ndarray, np.ndarray]],
                       cat_features: List[str],
                       groups: Optional[np.ndarray] = None,
                       time_order: Optional[np.ndarray] = None,
                       version: Optional[str] = None,
                       champion_lift: Optional[float] = None) -> HurdleModel:
    """분할 검증(시간/그룹) → 지표 → 전체 학습. champion-challenger 판정."""
    if version is None:
        version = f"{config.MODEL_SCHEMA_VERSION}-{_dt.datetime.now():%Y%m%d%H%M%S}"
    model = HurdleModel(cat_features=cat_features, version=version)

    y1 = np.asarray(labels.get("1m", (np.zeros(len(X)), None))[0]).astype(int)
    metrics: Dict = {
        "adopted": True, "k": config.LIFT_TOP_K_PCT,
        "n_positive_1m": int(y1.sum()),
        "n_positive_3m": int(np.asarray(labels.get("3m", (np.zeros(len(X)), None))[0]).sum())
        if "3m" in labels else None,
        "lift@k": 0.0,
    }

    if _CATBOOST_OK and y1.sum() >= config.POSITIVE_MIN_FOR_ML:
        tr, va = time_group_split(len(X), groups, time_order)
        if len(va) > 0 and y1[tr].sum() > 0 and y1[va].sum() > 0:
            vm = HurdleModel(cat_features=cat_features)
            vm.fit(X.iloc[tr], {h: (labels[h][0][tr], labels[h][1][tr])
                                for h in labels})
            prob = vm.predict_proba(X.iloc[va])
            if prob is not None:
                metrics.update(_validation_metrics(y1[va], prob, X.iloc[va]))
                metrics["lift@k"] = metrics.get("lift@10%", 0.0)

    if champion_lift is not None and metrics["lift@k"] < champion_lift:
        metrics["adopted"] = False
        metrics["champion_lift"] = champion_lift

    model.fit(X, labels)
    model.metrics = metrics
    return model
