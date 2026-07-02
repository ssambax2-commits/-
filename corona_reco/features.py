# -*- coding: utf-8 -*-
"""
features.py — ML 피처 빌드 + 누수 blacklist assertion
======================================================
ML X 는 '매입/기표 시점 고정값'만 사용한다(§6-5). 입금/결과에 의해 기계적으로
변하는 컬럼은 전면 금지하며, assert_no_leakage() 로 빌드 게이팅한다(§9-14).
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from . import config, util
from .io_loader import get_col

# ML 피처가 읽어도 되는 '원천 컬럼' 화이트리스트 (매입/기표 시점 고정값)
FEATURE_SOURCE_COLUMNS: List[str] = [
    "최초원금", "매입당시OPB", "최초미수금", "최초원리금",
    "매입일자", "대출일자", "최초연체일", "등록사유발생일",
    "대출이율", "연체이율", "양도횟수", "법시행이후양도횟수",
    "다중계좌 활동/총건수", "차주구분", "채권구분",
    "주민등록번호", "상품명", "담보세부종류", "대출종류",
]

# 최종 피처 컬럼명(모델 입력)
NUMERIC_FEATURES = [
    "log_최초원금", "log_매입당시OPB", "log_최초미수금", "log_최초원리금",
    "경과월_매입", "경과월_대출", "경과월_최초연체", "경과월_등록사유",
    "대출이율", "연체이율", "양도횟수", "법시행이후양도횟수", "다중계좌_총건수",
    "개인사업자", "담보부NPL",
]
CATEGORICAL_FEATURES = ["나이대", "상품군"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


@dataclass
class FeatureBundle:
    X: pd.DataFrame
    feature_names: List[str]
    cat_features: List[str]
    source_columns: List[str] = field(default_factory=lambda: list(FEATURE_SOURCE_COLUMNS))


class LeakageError(AssertionError):
    """ML 피처 누수 게이팅 실패."""


def assert_no_leakage(feature_names: List[str], source_columns: List[str] = None) -> None:
    """§9-14 게이팅. 피처/원천 컬럼에 blacklist 항목이 0건이어야 한다.

    실패 시 LeakageError(빌드 중단). CI/pytest 및 features 빌드 시 호출.
    """
    bl = config.ML_FEATURE_BLACKLIST
    prefixes = config.ML_FEATURE_BLACKLIST_PREFIX

    def _hits(names):
        hits = []
        for c in names:
            cs = str(c).strip()
            if cs in bl:
                hits.append(cs)
                continue
            for p in prefixes:
                # '다중계좌 총건수'는 허용, '입금액N/입금일자N' 계열만 차단
                if cs.startswith(p) and any(ch.isdigit() for ch in cs):
                    hits.append(cs)
                    break
        return hits

    feat_hits = _hits(feature_names)
    src_hits = _hits(source_columns) if source_columns else []
    if feat_hits or src_hits:
        raise LeakageError(
            "ML 피처 누수 게이팅 실패 — blacklist 컬럼 검출: "
            f"features={feat_hits}, source={src_hits}"
        )


def _parse_multi_total(val) -> float:
    """'활동/총건수' 형태에서 총건수를 뽑는다. '3/5'->5, '5'->5."""
    if val is None:
        return np.nan
    if isinstance(val, float) and pd.isna(val):
        return np.nan
    s = str(val).strip()
    if s == "":
        return np.nan
    if "/" in s:
        parts = s.split("/")
        n = util.to_number(parts[-1])
        return np.nan if n is None else n
    n = util.to_number(s)
    return np.nan if n is None else n


def _is_biz(val) -> int:
    """차주구분: 개인사업자면 1, 아니면 0."""
    s = str(val).strip() if val is not None else ""
    return 1 if "사업자" in s else 0


def _is_collateral_npl(val) -> int:
    s = str(val).strip() if val is not None else ""
    return 1 if s == config.COLLATERAL_NPL_LABEL else 0


def _product_group(product, collateral_detail, loan_kind) -> str:
    """상품군: 담보/신용/기타."""
    for v in (collateral_detail, product, loan_kind):
        s = str(v).strip() if v is not None else ""
        if s and s.lower() not in ("nan", "none"):
            if "담보" in s:
                return "담보"
    for v in (product, loan_kind):
        s = str(v).strip() if v is not None else ""
        if "신용" in s:
            return "신용"
    # 담보세부종류가 채워져 있으면 담보로 간주
    cd = str(collateral_detail).strip() if collateral_detail is not None else ""
    if cd and cd.lower() not in ("nan", "none", ""):
        return "담보"
    return "기타"


def build_features(df: pd.DataFrame, ref_date: _dt.date,
                   age_bands: Optional[pd.Series] = None) -> FeatureBundle:
    """활동/학습 DataFrame → ML 피처 번들.

    age_bands: 사전 계산된 나이대 Series(주민번호 원본 폐기 후). 없으면 여기서 계산.
    """
    n = len(df)
    idx = df.index

    c_first_principal = get_col(df, "최초원금")
    c_opb = get_col(df, "매입당시OPB")
    c_first_unpaid = get_col(df, "최초미수금")
    c_first_pi = get_col(df, "최초원리금")
    c_buy_date = get_col(df, "매입일자")
    c_loan_date = get_col(df, "대출일자")
    c_first_delinq = get_col(df, "최초연체일")
    c_reg_reason = get_col(df, "등록사유발생일")
    c_loan_rate = get_col(df, "대출이율")
    c_delinq_rate = get_col(df, "연체이율")
    c_transfer = get_col(df, "양도횟수")
    c_transfer_law = get_col(df, "법시행이후양도횟수")
    c_multi = get_col(df, "다중계좌 활동/총건수")
    c_borrower_type = get_col(df, "차주구분")
    c_bond_type = get_col(df, "채권구분")
    c_product = get_col(df, "상품명")
    c_collateral_detail = get_col(df, "담보세부종류")
    c_loan_kind = get_col(df, "대출종류")

    data = {}
    data["log_최초원금"] = [util.safe_log1p(v) for v in c_first_principal]
    data["log_매입당시OPB"] = [util.safe_log1p(v) for v in c_opb]
    data["log_최초미수금"] = [util.safe_log1p(v) for v in c_first_unpaid]
    data["log_최초원리금"] = [util.safe_log1p(v) for v in c_first_pi]

    def elapsed(col):
        out = []
        for v in col:
            d = util.parse_date(v)
            m = util.months_between(ref_date, d)
            out.append(np.nan if m is None else round(m, 2))
        return out

    data["경과월_매입"] = elapsed(c_buy_date)
    data["경과월_대출"] = elapsed(c_loan_date)
    data["경과월_최초연체"] = elapsed(c_first_delinq)
    data["경과월_등록사유"] = elapsed(c_reg_reason)

    data["대출이율"] = [util.to_number(v) if util.to_number(v) is not None else np.nan for v in c_loan_rate]
    data["연체이율"] = [util.to_number(v) if util.to_number(v) is not None else np.nan for v in c_delinq_rate]
    data["양도횟수"] = [util.to_number_or(v, 0) for v in c_transfer]
    data["법시행이후양도횟수"] = [util.to_number_or(v, 0) for v in c_transfer_law]
    data["다중계좌_총건수"] = [_parse_multi_total(v) for v in c_multi]
    data["개인사업자"] = [_is_biz(v) for v in c_borrower_type]
    data["담보부NPL"] = [_is_collateral_npl(v) for v in c_bond_type]

    # 나이대 (categorical)
    if age_bands is not None:
        data["나이대"] = list(age_bands.reindex(idx).fillna("unknown"))
    else:
        rrn = get_col(df, "주민등록번호")
        bands = []
        for v in rrn:
            band, _gender = util.rrn_to_age_gender(v, ref_date)
            bands.append(band)  # 원본 v는 여기서 사용 후 폐기됨
        data["나이대"] = bands

    data["상품군"] = [
        _product_group(c_product.iloc[i], c_collateral_detail.iloc[i], c_loan_kind.iloc[i])
        for i in range(n)
    ]

    X = pd.DataFrame(data, index=idx)[ALL_FEATURES]
    # 카테고리 결측 방어
    for c in CATEGORICAL_FEATURES:
        X[c] = X[c].fillna("unknown").astype(str)

    # 누수 게이팅 (빌드 시점 assert)
    assert_no_leakage(list(X.columns), FEATURE_SOURCE_COLUMNS)

    return FeatureBundle(
        X=X,
        feature_names=list(X.columns),
        cat_features=list(CATEGORICAL_FEATURES),
        source_columns=list(FEATURE_SOURCE_COLUMNS),
    )


def compute_age_gender_bands(df: pd.DataFrame, ref_date: _dt.date):
    """주민등록번호 → (나이대 Series, 성별 Series). 원본은 저장하지 않는다.

    성별은 점수 미반영(공정성 진단 표시용). 이 함수 밖으로 주민번호 원본은
    절대 전달되지 않는다.
    """
    rrn = get_col(df, "주민등록번호")
    ages, genders = [], []
    for v in rrn:
        band, gender = util.rrn_to_age_gender(v, ref_date)
        ages.append(band)
        genders.append(gender)
    return (pd.Series(ages, index=df.index, name="나이대"),
            pd.Series(genders, index=df.index, name="성별"))
