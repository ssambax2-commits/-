# -*- coding: utf-8 -*-
"""점수 컴포넌트 테스트 (§9: 4 민원, 5 담보부NPL, 6 개인사업자)."""
import datetime as _dt

import pandas as pd

from corona_reco import scoring, config

REF = _dt.date(2026, 7, 1)


def make_pay(has=False, within365=False, within180=False):
    return pd.DataFrame([{
        "입금_이력유무": has, "입금_최근180일": within180, "입금_최근365일": within365,
        "입금_최근730일": has, "입금_최근일": (REF if has else None),
        "입금_건수2이상": False, "입금_건수3이상": False,
        "입금_총액10만이상": False, "입금_총액50만이상": False,
        "입금_일자불명금액존재": False, "입금_총액": 0.0, "입금_건수": 0, "입금_최근액": 0.0,
    }])


# 4. 민원 → -25, 중복이어도 1회
def test_minwon_penalty_once():
    df = pd.DataFrame([{"민원여부": "있음", "채권상태(중)": "민원"}])  # 둘 다 민원
    pen = scoring.compute_sensitive_penalty(df)
    assert pen.iloc[0] == config.PENALTY_MINWON  # 25 (50 아님)


def test_minwon_single_source():
    df = pd.DataFrame([{"민원여부": "있음", "채권상태(중)": "정상"}])
    assert scoring.compute_sensitive_penalty(df).iloc[0] == 25


def test_legal_penalty():
    df = pd.DataFrame([{"민원여부": "", "채권상태(중)": "법조치"}])
    assert scoring.compute_sensitive_penalty(df).iloc[0] == config.PENALTY_LEGAL


def test_minwon_and_legal_stack():
    df = pd.DataFrame([{"민원여부": "있음", "채권상태(중)": "법조치"}])
    # 민원(25) + 법조치(20)
    assert scoring.compute_sensitive_penalty(df).iloc[0] == 45


# 5. 담보부NPL 배수/등급상한
def test_collateral_npl_no_payment():
    df = pd.DataFrame([{"채권구분": "담보부NPL"}])
    pay = make_pay(has=False)
    mult, cap, key = scoring.compute_collateral(df, pay)
    assert abs(mult.iloc[0] - 0.55) < 1e-9
    assert cap.iloc[0] == "B"
    assert key.iloc[0] == "none"


def test_collateral_npl_recent365():
    df = pd.DataFrame([{"채권구분": "담보부NPL"}])
    pay = make_pay(has=True, within365=True)
    mult, cap, key = scoring.compute_collateral(df, pay)
    assert abs(mult.iloc[0] - 0.80) < 1e-9
    assert cap.iloc[0] == "S"


def test_collateral_npl_old_payment():
    df = pd.DataFrame([{"채권구분": "담보부NPL"}])
    pay = make_pay(has=True, within365=False)  # 과거입금
    mult, cap, key = scoring.compute_collateral(df, pay)
    assert abs(mult.iloc[0] - 0.70) < 1e-9
    assert cap.iloc[0] == "A"


def test_collateral_normal():
    df = pd.DataFrame([{"채권구분": "일반무담보"}])
    pay = make_pay(has=False)
    mult, cap, key = scoring.compute_collateral(df, pay)
    assert abs(mult.iloc[0] - 1.00) < 1e-9
    assert cap.iloc[0] == "S"


# 6. 개인사업자 biz_adj (external_prior 한 곳에서만)
def test_biz_no_payment():
    age = pd.Series(["40대"])
    biz = pd.Series([True])
    pay = make_pay(has=False)
    adj = scoring.compute_external_prior_adj(age, biz, pay)
    # 40대(+3) + 개인사업자 무입금(-5) = -2
    assert adj.iloc[0] == -2


def test_biz_recent365_zero():
    age = pd.Series(["40대"])
    biz = pd.Series([True])
    pay = make_pay(has=True, within365=True)
    adj = scoring.compute_external_prior_adj(age, biz, pay)
    # 40대(+3) + 개인사업자 recent365(0) = +3
    assert adj.iloc[0] == 3


def test_individual_no_biz_penalty():
    age = pd.Series(["40대"])
    biz = pd.Series([False])  # 개인
    pay = make_pay(has=False)
    adj = scoring.compute_external_prior_adj(age, biz, pay)
    assert adj.iloc[0] == 3  # 40대만 반영


def test_prior_clamp():
    age = pd.Series(["80대이상"])  # -3
    biz = pd.Series([True])       # 무입금 -5 => -8
    pay = make_pay(has=False)
    adj = scoring.compute_external_prior_adj(age, biz, pay)
    assert adj.iloc[0] == -8  # 하한 clamp


# payment_history_score 스펙 확인
def test_payment_history_bands():
    pay = make_pay(has=True, within180=True, within365=True)
    s, limited = scoring.compute_payment_history_score(pay)
    assert s.iloc[0] == 100
    assert not limited.iloc[0]

    pay2 = make_pay(has=False)
    s2, _ = scoring.compute_payment_history_score(pay2)
    assert s2.iloc[0] == 0


def test_small_single_payment_limited():
    """§7 소액(10만↓) 단건 입금 → 가점 제한 배수 적용."""
    import pandas as pd
    pay = pd.DataFrame([{
        "입금_이력유무": True, "입금_최근180일": True, "입금_최근365일": True,
        "입금_최근730일": True, "입금_최근일": REF, "입금_건수2이상": False,
        "입금_건수3이상": False, "입금_총액10만이상": False, "입금_총액50만이상": False,
        "입금_일자불명금액존재": False, "입금_총액": 50000, "입금_건수": 1, "입금_최근액": 50000,
    }])
    s, limited = scoring.compute_payment_history_score(pay)
    assert limited.iloc[0]
    assert s.iloc[0] == 100 * config.RECENT_PAY_SMALL_MULT
