# -*- coding: utf-8 -*-
"""입금내역 파싱/파생값 테스트."""
import datetime as _dt

import pandas as pd

from corona_reco import payments

REF = _dt.date(2026, 7, 1)


def test_named_pairs_parse():
    df = pd.DataFrame([{
        "입금일자1": "2026-06-01", "입금액1": 100000,
        "입금일자2": "2026-05-01", "입금액2": 50000,
        "입금일자3": "", "입금액3": "",
    }])
    d = payments.derive_payments(df, REF)
    assert d["입금_이력유무"].iloc[0]
    assert d["입금_총액"].iloc[0] == 150000
    assert d["입금_건수"].iloc[0] == 2
    assert d["입금_건수2이상"].iloc[0]
    assert not d["입금_건수3이상"].iloc[0]
    assert d["입금_최근180일"].iloc[0]  # 2026-06-01 은 REF 30일 전


def test_negative_amount_ignored():
    df = pd.DataFrame([{
        "입금일자1": "2026-06-01", "입금액1": -5000,  # 환급/취소 → 무시
        "입금일자2": "2026-06-02", "입금액2": 0,       # 0 → 무시
    }])
    d = payments.derive_payments(df, REF)
    assert not d["입금_이력유무"].iloc[0]
    assert d["입금_총액"].iloc[0] == 0


def test_recency_bands_by_ref_date():
    # 400일 전 입금 → 365 초과, 730 이내
    old = (REF - _dt.timedelta(days=400)).strftime("%Y-%m-%d")
    df = pd.DataFrame([{"입금일자1": old, "입금액1": 100000}])
    d = payments.derive_payments(df, REF)
    assert not d["입금_최근365일"].iloc[0]
    assert d["입금_최근730일"].iloc[0]


def test_excel_serial_date():
    # 엑셀 serial: 2026-06-01 ≈ 46174
    serial = (_dt.date(2026, 6, 1) - _dt.date(1899, 12, 30)).days
    df = pd.DataFrame([{"입금일자1": serial, "입금액1": 100000}])
    d = payments.derive_payments(df, REF)
    assert d["입금_이력유무"].iloc[0]
    assert d["입금_최근180일"].iloc[0]


def test_amount_only_no_date():
    df = pd.DataFrame([{"입금일자1": "", "입금액1": 200000}])
    d = payments.derive_payments(df, REF)
    assert d["입금_이력유무"].iloc[0]
    assert d["입금_일자불명금액존재"].iloc[0]
    assert d["입금_최근일"].iloc[0] is None


def test_positional_fallback():
    # named 헤더 없이 CC~CV(80~99) 위치 fallback
    cols = [f"col{i}" for i in range(80)]
    row = {c: "" for c in cols}
    row["col_date"] = ""  # 자리 채우기용 아님
    # 80,81 = (일자,금액) 첫 쌍
    data = {c: [""] for c in cols}
    data["p80_date"] = ["2026-06-01"]
    data["p81_amt"] = [100000]
    df = pd.DataFrame(data)
    # 컬럼 순서 보장: 앞 80개 + 2개
    assert df.shape[1] == 82
    d = payments.derive_payments(df, REF)
    assert d["입금_이력유무"].iloc[0]


def test_positional_garbage_guard():
    """위치 fallback: 일자칸에 비일자 텍스트 → 입금으로 오인하지 않음."""
    data = {f"c{i}": [""] for i in range(80)}
    data["p80"] = ["서울지점"]   # 날짜 아님
    data["p81"] = [12345]        # 숫자(코드값)
    df = pd.DataFrame(data)
    d = payments.derive_payments(df, REF)
    assert not d["입금_이력유무"].iloc[0]


def test_positional_implausible_date_guard():
    """위치 fallback: serial 오인으로 1990년 이전 날짜가 나오면 무시."""
    data = {f"c{i}": [""] for i in range(80)}
    data["p80"] = [12345]        # serial → 1933년(비현실)
    data["p81"] = [100000]
    df = pd.DataFrame(data)
    d = payments.derive_payments(df, REF)
    assert not d["입금_이력유무"].iloc[0]


def test_positional_amount_only_still_allowed():
    """위치 fallback: 일자칸 빈 값 + 금액 → 일자불명 입금으로 인정."""
    data = {f"c{i}": [""] for i in range(80)}
    data["p80"] = [""]
    data["p81"] = [150000]
    df = pd.DataFrame(data)
    d = payments.derive_payments(df, REF)
    assert d["입금_이력유무"].iloc[0]
    assert d["입금_일자불명금액존재"].iloc[0]
    assert d["입금_최근액"].iloc[0] == 150000  # 확인된 금액 중 최대값 근사


def test_amount_bonuses():
    df = pd.DataFrame([{
        "입금일자1": "2026-06-01", "입금액1": 300000,
        "입금일자2": "2026-05-01", "입금액2": 300000,
        "입금일자3": "2026-04-01", "입금액3": 100000,
    }])
    d = payments.derive_payments(df, REF)
    assert d["입금_건수3이상"].iloc[0]
    assert d["입금_총액50만이상"].iloc[0]  # 700000
