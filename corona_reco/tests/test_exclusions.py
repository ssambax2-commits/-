# -*- coding: utf-8 -*-
"""강제 제외 규칙 + 차주단위 전파 테스트 (§9: 1,2,3,9,10,11)."""
import datetime as _dt

import pandas as pd

from corona_reco import exclusions

REF = _dt.date(2026, 7, 1)

_DEFAULTS = {
    "원장상태": "활동", "채권상태(중)": "정상", "채무부존재소송": "",
    "회생": "", "시효일자": "2030-01-01", "고객번호": "C1", "성명": "홍길동",
}


def make_df(rows):
    full = []
    for i, r in enumerate(rows):
        base = dict(_DEFAULTS)
        base.update(r)
        base.setdefault("대출번호", f"L{i}")
        full.append(base)
    return pd.DataFrame(full)


def _ex(df):
    return exclusions.apply_exclusions(df, REF)


# 1. 활동 · 폐지_신복/폐지_회생/폐지_파산 → 안내 가능(제외 아님)
def test_pyeji_not_excluded():
    df = make_df([
        {"채권상태(중)": "폐지_신복"},
        {"채권상태(중)": "폐지_회생"},
        {"채권상태(중)": "폐지_파산"},
    ])
    out = _ex(df)
    assert (~out["_제외여부"]).all(), out[["채권상태(중)", "_제외여부", "_제외사유"]]


# 2. 회생=인가결정 / 신용회복지원신청 → 강제 제외
def test_rehab_excluded():
    df = make_df([
        {"회생": "인가결정", "고객번호": "A", "성명": "김"},
        {"회생": "신용회복지원신청", "고객번호": "B", "성명": "이"},
    ])
    out = _ex(df)
    assert out["_제외여부"].all()
    assert "연락금지" in out["_제외사유"].iloc[0] or "회생" in out["_제외사유"].iloc[0]


# 3. 상태중=사망/완제 → 강제 제외
def test_status_death_complete_excluded():
    df = make_df([
        {"채권상태(중)": "사망", "고객번호": "A", "성명": "김"},
        {"채권상태(중)": "완제", "고객번호": "B", "성명": "이"},
    ])
    out = _ex(df)
    assert out["_제외여부"].all()


# 4(부분). 상태중=민원 → 비제외 (패널티는 scoring에서 검증)
def test_minwon_not_excluded():
    df = make_df([{"채권상태(중)": "민원"}])
    out = _ex(df)
    assert not out["_제외여부"].iloc[0]


# 9. 시효일자 경과 → 강제 제외
def test_expiry_excluded():
    df = make_df([{"시효일자": "2020-01-01"}])  # REF(2026) 이전
    out = _ex(df)
    assert out["_제외여부"].iloc[0]
    assert "소멸시효" in out["_제외사유"].iloc[0]


def test_expiry_option_off():
    df = make_df([{"시효일자": "2020-01-01"}])
    out = exclusions.apply_exclusions(df, REF, exclude_expired_prescription=False)
    assert not out["_제외여부"].iloc[0]


# 10. 부채증명원 → 무시(비제외·무패널티)
def test_bujeung_ignored():
    df = make_df([{"회생": "부채증명원"}])
    out = _ex(df)
    assert not out["_제외여부"].iloc[0]
    assert out["_제외사유"].iloc[0] == ""


# 3(추가). 채무부존재소송 → 제외
def test_nonexist_suit_excluded():
    df = make_df([{"채무부존재소송": "있음"}])
    out = _ex(df)
    assert out["_제외여부"].iloc[0]


# 11. 차주단위 전파: 동일 차주에 연락금지 계좌 있으면 전 계좌 제외
def test_borrower_propagation():
    df = make_df([
        {"고객번호": "C9", "성명": "박전파", "회생": "인가결정", "대출번호": "L1"},
        {"고객번호": "C9", "성명": "박전파", "회생": "", "채권상태(중)": "정상", "대출번호": "L2"},
        {"고객번호": "C8", "성명": "타인", "회생": "", "채권상태(중)": "정상", "대출번호": "L3"},
    ])
    out = _ex(df)
    # 같은 차주(C9) 두 계좌 모두 제외
    c9 = out[out["고객번호"] == "C9"]
    assert c9["_제외여부"].all()
    # 다른 차주는 유지
    c8 = out[out["고객번호"] == "C8"]
    assert not c8["_제외여부"].iloc[0]


def test_death_propagation():
    """사망은 사람상태 → 차주 전 계좌 전파."""
    df = make_df([
        {"고객번호": "D1", "성명": "고인", "채권상태(중)": "사망", "대출번호": "L1"},
        {"고객번호": "D1", "성명": "고인", "채권상태(중)": "정상", "대출번호": "L2"},
    ])
    out = _ex(df)
    assert out["_제외여부"].all()


def test_complete_is_account_level():
    """완제는 계좌단위(전파 아님)."""
    df = make_df([
        {"고객번호": "E1", "성명": "완제자", "채권상태(중)": "완제", "대출번호": "L1"},
        {"고객번호": "E1", "성명": "완제자", "채권상태(중)": "정상", "대출번호": "L2"},
    ])
    out = _ex(df)
    assert out[out["대출번호"] == "L1"]["_제외여부"].iloc[0]
    assert not out[out["대출번호"] == "L2"]["_제외여부"].iloc[0]
