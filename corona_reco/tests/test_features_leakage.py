# -*- coding: utf-8 -*-
"""누수 게이팅 + 주민번호 처리 테스트 (§9: 7, 14)."""
import datetime as _dt

import pandas as pd
import pytest

from corona_reco import config, features, util

REF = _dt.date(2026, 7, 1)


def _sample_df():
    return pd.DataFrame([{
        "최초원금": 1000000, "매입당시OPB": 900000, "최초미수금": 50000, "최초원리금": 1100000,
        "매입일자": "2020-01-01", "대출일자": "2015-01-01", "최초연체일": "2016-06-01",
        "등록사유발생일": "2016-09-01", "대출이율": 10.0, "연체이율": 18.0,
        "양도횟수": 2, "법시행이후양도횟수": 1, "다중계좌 활동/총건수": "2/4",
        "차주구분": "개인사업자", "채권구분": "담보부NPL",
        "주민등록번호": "800101-1234567", "상품명": "신용대출",
        "담보세부종류": "", "대출종류": "신용",
        # 아래는 blacklist(누수) 컬럼 — 절대 피처에 들어가면 안 됨
        "현재원금": 500000, "채권상태(중)": "정상", "회생": "정상",
        "입금액1": 100000, "입금일자1": "2026-01-01",
    }])


# 14. 누수 게이팅: ML 피처/원천 컬럼에 blacklist 0건
def test_feature_columns_no_blacklist():
    fb = features.build_features(_sample_df(), REF)
    cols = set(fb.X.columns)
    assert cols & config.ML_FEATURE_BLACKLIST == set()
    # 동적 입금열도 없어야
    for c in cols:
        assert not (c.startswith("입금") and any(ch.isdigit() for ch in c))


def test_source_columns_no_blacklist():
    assert set(features.FEATURE_SOURCE_COLUMNS) & config.ML_FEATURE_BLACKLIST == set()


def test_assert_no_leakage_raises_on_blacklist():
    with pytest.raises(features.LeakageError):
        features.assert_no_leakage(["log_최초원금", "현재원금"])  # 현재원금 = blacklist


def test_assert_no_leakage_raises_on_payment_columns():
    with pytest.raises(features.LeakageError):
        features.assert_no_leakage(["입금액1"])


def test_assert_no_leakage_passes_clean():
    features.assert_no_leakage(features.ALL_FEATURES, features.FEATURE_SOURCE_COLUMNS)


def test_multi_total_not_sum():
    """다중계좌 총건수만 사용(원금합계 아님)."""
    fb = features.build_features(_sample_df(), REF)
    assert fb.X["다중계좌_총건수"].iloc[0] == 4  # '2/4' → 4
    assert "다중계좌 원금합계" not in fb.X.columns


# 7. 주민번호 → 나이대 산출, 성별 점수 미반영, 원본 미저장
def test_rrn_age_band():
    band, gender = util.rrn_to_age_gender("800101-1234567", REF)  # 1980년생, 남
    assert band == "40대"   # 2026 - 1980 = 46
    assert gender == "남"


def test_rrn_female_2000s():
    band, gender = util.rrn_to_age_gender("050101-4234567", REF)  # 2005년생, 여
    assert gender == "여"
    assert band == "20대이하"


def test_gender_not_in_features():
    fb = features.build_features(_sample_df(), REF)
    assert "성별" not in fb.X.columns
    assert "주민등록번호" not in fb.X.columns


def test_rrn_original_not_in_features():
    df = _sample_df()
    raw = df["주민등록번호"].iloc[0]
    fb = features.build_features(df, REF)
    # 어떤 피처 값에도 원본 주민번호가 들어가면 안 됨
    for c in fb.X.columns:
        vals = [str(v) for v in fb.X[c].tolist()]
        assert raw not in vals
    # 나이대만 categorical로 존재
    assert "나이대" in fb.X.columns
    assert fb.X["나이대"].iloc[0] == "40대"


def test_age_band_boundaries():
    assert util.age_to_band(29) == "20대이하"
    assert util.age_to_band(30) == "30대"
    assert util.age_to_band(80) == "80대이상"
