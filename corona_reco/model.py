# -*- coding: utf-8 -*-
"""
model.py — CatBoost 2단계 Hurdle 모델 (§7)
===========================================
Stage 1: 1개월 내 입금 확률(분류). balanced bagging 앙상블 → 평균확률.
Stage 2: (기본 OFF) positive에 한해 회수비율(입금액/원금잔액) 예측.
검증: out-of-time/차주그룹 분리 split, 주지표 Lift@상위K(%). champion-challenger.
표본 부족(positive<임계)이면 규칙·prior 폴백(active=False). 시드 고정.
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import config

try:
    from catboost import CatBoostClassifier, CatBoostRegressor, Pool
    _CATBOOST_OK = True
except Exception:  # noqa: BLE001  (완전 폐쇄망 방어 — 실행 중엔 이미 번들됨)
    _CATBOOST_OK = False


@dataclass
class HurdleModel:
    cat_features: List[str] = field(default_factory=list)
    feature_names: List[str] = field(default_factory=list)
    version: str = ""
    active: bool = False           # ML 사용 가능 여부(표본 충분 & 학습 성공)
    n_positive: int = 0
    n_negative: int = 0
    metrics: Dict = field(default_factory=dict)
    stage2_enabled: bool = False
    warning: str = ""

    # 학습된 앙상블(직렬화 대상)
    _clf_models: list = field(default_factory=list, repr=False)
    _reg_model: object = field(default=None, repr=False)

    # ------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray,
            groups: Optional[np.ndarray] = None,
            recovery: Optional[np.ndarray] = None,
            ensemble_size: int = None,
            stage2_enabled: bool = None) -> "HurdleModel":
        y = np.asarray(y).astype(int)
        self.feature_names = list(X.columns)
        self.n_positive = int((y == 1).sum())
        self.n_negative = int((y == 0).sum())
        if ensemble_size is None:
            ensemble_size = config.BAGGING_ENSEMBLE_SIZE
        if stage2_enabled is None:
            stage2_enabled = config.STAGE2_ENABLED_DEFAULT

        if not _CATBOOST_OK:
            self.active = False
            self.warning = "CatBoost 미탑재 — 규칙·prior 폴백."
            return self
        if self.n_positive < config.POSITIVE_MIN_FOR_ML:
            self.active = False
            self.warning = (
                f"positive={self.n_positive} < 임계 {config.POSITIVE_MIN_FOR_ML} — "
                "ML 신뢰도 낮음, 규칙·prior 위주 폴백."
            )
            return self

        cat_idx = [X.columns.get_loc(c) for c in self.cat_features if c in X.columns]
        pos_idx = np.where(y == 1)[0]
        neg_idx = np.where(y == 0)[0]
        rng = np.random.RandomState(config.RANDOM_SEED)

        # negative 서브샘플 크기: positive의 3배(불균형 완화), 상한은 전체 negative
        neg_sample_size = min(len(neg_idx), max(len(pos_idx) * 3, len(pos_idx) + 1))

        models = []
        for b in range(ensemble_size):
            sub_neg = rng.choice(neg_idx, size=neg_sample_size, replace=False) \
                if neg_sample_size < len(neg_idx) else neg_idx
            idx = np.concatenate([pos_idx, sub_neg])
            rng.shuffle(idx)
            Xb = X.iloc[idx]
            yb = y[idx]
            params = dict(config.CATBOOST_PARAMS)
            params["random_seed"] = config.RANDOM_SEED + b
            params["auto_class_weights"] = "Balanced"
            clf = CatBoostClassifier(**params)
            pool = Pool(Xb, yb, cat_features=cat_idx)
            clf.fit(pool)
            models.append(clf)

        self._clf_models = models
        self.active = True
        self.warning = ""

        # Stage 2 (옵션): positive에 한해 회수비율 회귀
        self.stage2_enabled = bool(stage2_enabled and recovery is not None
                                   and self.n_positive >= config.POSITIVE_MIN_FOR_ML)
        if self.stage2_enabled:
            rec = np.asarray(recovery, dtype=float)
            mask = (y == 1) & np.isfinite(rec)
            if mask.sum() >= config.POSITIVE_MIN_FOR_ML:
                reg = CatBoostRegressor(
                    iterations=200, depth=4, learning_rate=0.05,
                    l2_leaf_reg=6.0, loss_function="RMSE",
                    random_seed=config.RANDOM_SEED, verbose=False,
                    allow_writing_files=False,
                )
                reg.fit(Pool(X.iloc[mask.nonzero()[0]], rec[mask], cat_features=cat_idx))
                self._reg_model = reg
            else:
                self.stage2_enabled = False
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """앙상블 평균 입금확률(0~1). 비활성 시 None."""
        if not self.active or not self._clf_models:
            return None
        Xc = X[self.feature_names] if all(c in X.columns for c in self.feature_names) else X
        preds = np.zeros(len(Xc))
        for clf in self._clf_models:
            preds += clf.predict_proba(Xc)[:, 1]
        return preds / len(self._clf_models)

    def predict_recovery(self, X: pd.DataFrame) -> Optional[np.ndarray]:
        if not self.stage2_enabled or self._reg_model is None:
            return None
        Xc = X[self.feature_names] if all(c in X.columns for c in self.feature_names) else X
        return self._reg_model.predict(Xc)

    # ------------------------------------------------------------------
    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self, path)

    @staticmethod
    def load(path: str) -> "HurdleModel":
        import joblib
        return joblib.load(path)


# ---------------------------------------------------------------------------
# 검증 지표
# ---------------------------------------------------------------------------
def lift_at_k(y_true: np.ndarray, scores: np.ndarray, k: float = None) -> float:
    """Lift@상위K(%). 상위 K 비율에서의 positive율 / 전체 positive율."""
    if k is None:
        k = config.LIFT_TOP_K_PCT
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    if n == 0:
        return 0.0
    base_rate = y_true.mean()
    if base_rate <= 0:
        return 0.0
    topn = max(1, int(np.ceil(n * k)))
    order = np.argsort(-scores)
    top_rate = y_true[order[:topn]].mean()
    return float(top_rate / base_rate)


def precision_at_topn(y_true: np.ndarray, scores: np.ndarray, n_top: int = 50) -> float:
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n = len(y_true)
    if n == 0:
        return 0.0
    n_top = min(n_top, n)
    order = np.argsort(-scores)
    return float(y_true[order[:n_top]].mean())


def time_group_split(n: int, groups: Optional[np.ndarray],
                     time_order: Optional[np.ndarray],
                     val_frac: float = 0.25, seed: int = None):
    """out-of-time(time_order 있으면) 또는 그룹셔플 분할. 같은 차주 분리 보장.

    반환: (train_idx, val_idx)
    """
    if seed is None:
        seed = config.RANDOM_SEED
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    if time_order is not None and np.isfinite(pd.to_numeric(pd.Series(time_order),
                                                            errors="coerce")).any():
        # 시점 기준 뒤쪽 val_frac 를 검증셋으로
        t = pd.to_numeric(pd.Series(time_order), errors="coerce").fillna(-np.inf).values
        order = np.argsort(t)
        cut = int(n * (1 - val_frac))
        train_sorted, val_sorted = order[:cut], order[cut:]
        if groups is not None:
            # 그룹 누수 방지: val에 있는 그룹은 train에서 제거
            val_groups = set(np.asarray(groups)[val_sorted])
            train_sorted = np.array([i for i in train_sorted
                                     if np.asarray(groups)[i] not in val_groups])
        return train_sorted, val_sorted

    # 그룹셔플
    if groups is not None:
        uniq = np.array(sorted(set(groups)))
        rng.shuffle(uniq)
        n_val_g = max(1, int(len(uniq) * val_frac))
        val_g = set(uniq[:n_val_g])
        val_idx = idx[np.array([g in val_g for g in groups])]
        train_idx = idx[np.array([g not in val_g for g in groups])]
        return train_idx, val_idx

    rng.shuffle(idx)
    cut = int(n * (1 - val_frac))
    return idx[:cut], idx[cut:]


def train_and_validate(X: pd.DataFrame, y: np.ndarray, cat_features: List[str],
                       groups: Optional[np.ndarray] = None,
                       time_order: Optional[np.ndarray] = None,
                       recovery: Optional[np.ndarray] = None,
                       version: Optional[str] = None,
                       champion_lift: Optional[float] = None,
                       stage2_enabled: bool = None) -> HurdleModel:
    """분할 검증으로 지표 산출 후 전체 데이터로 최종 학습. champion-challenger 판정.

    champion_lift: 기존 챔피언의 Lift@K. challenger가 못 이기면 adopted=False로 표기
    (호출부에서 기존 모델 유지 결정).
    """
    y = np.asarray(y).astype(int)
    if version is None:
        version = f"{config.MODEL_SCHEMA_VERSION}-{_dt.datetime.now():%Y%m%d%H%M%S}"

    model = HurdleModel(cat_features=cat_features, version=version)
    metrics = {"lift@k": 0.0, "precision@topn": 0.0, "k": config.LIFT_TOP_K_PCT,
               "adopted": True, "n_positive": int((y == 1).sum())}

    # 검증(표본 충분할 때만 의미)
    if _CATBOOST_OK and (y == 1).sum() >= config.POSITIVE_MIN_FOR_ML:
        tr, va = time_group_split(len(y), groups, time_order)
        if len(va) > 0 and (y[tr] == 1).sum() > 0 and (y[va] == 1).sum() > 0:
            val_model = HurdleModel(cat_features=cat_features)
            val_model.fit(X.iloc[tr], y[tr])
            prob = val_model.predict_proba(X.iloc[va])
            if prob is not None:
                metrics["lift@k"] = round(lift_at_k(y[va], prob), 4)
                metrics["precision@topn"] = round(precision_at_topn(y[va], prob), 4)

    # champion-challenger
    if champion_lift is not None and metrics["lift@k"] < champion_lift:
        metrics["adopted"] = False
        metrics["champion_lift"] = champion_lift

    # 최종 모델은 전체 데이터로 학습(채택 시 사용)
    model.fit(X, y, groups=groups, recovery=recovery, stage2_enabled=stage2_enabled)
    model.metrics = metrics
    return model
