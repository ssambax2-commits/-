# -*- coding: utf-8 -*-
"""
payments.py — 입금내역 파싱 및 파생값
======================================
학습데이터의 오른쪽에 가로형으로 붙는 입금일자N/입금액N 쌍을 파싱한다.
헤더에 입금일자N/입금액N 이 있으면 그걸 쓰고, 없으면 CC~CV(0-index 80~99)
위치로 fallback한다. 모든 최근성 판정은 '평가 기준일' 기준이며 시스템
날짜를 쓰지 않는다. 입금액 ≤ 0 은 환급/취소 추정으로 무시한다.
"""
from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from . import util

# 입금 관련 파생 컬럼명(내부 사용)
PAY_COLS = [
    "입금_이력유무", "입금_총액", "입금_건수", "입금_최근일", "입금_최근액",
    "입금_최근180일", "입금_최근365일", "입금_최근730일",
    "입금_건수2이상", "입금_건수3이상", "입금_총액10만이상", "입금_총액50만이상",
    "입금_일자불명금액존재",
]

_DATE_PAT = re.compile(r"^입금일자\s*(\d+)$")
_AMT_PAT = re.compile(r"^입금액\s*(\d+)$")


def find_payment_pairs(columns: List[str]) -> List[Tuple[str, str]]:
    """컬럼 목록에서 (입금일자컬럼, 입금액컬럼) 쌍 리스트를 반환.

    named 헤더(입금일자N/입금액N)를 우선 사용. 정규화된 컬럼명 기준.
    """
    dates = {}
    amts = {}
    for c in columns:
        cs = str(c).strip()
        md = _DATE_PAT.match(cs.replace(" ", ""))
        ma = _AMT_PAT.match(cs.replace(" ", ""))
        if md:
            dates[int(md.group(1))] = c
        elif ma:
            amts[int(ma.group(1))] = c
    pairs = []
    for n in sorted(set(dates) | set(amts)):
        d = dates.get(n)
        a = amts.get(n)
        if a is not None:  # 금액 컬럼은 최소 있어야 의미
            pairs.append((d, a))
    return pairs


def positional_pairs(df: pd.DataFrame, start: int = 80, end: int = 100) -> List[Tuple[Optional[int], int]]:
    """named 헤더가 없을 때 CC~CV(0-index 80~99) 위치 기반 쌍.

    (일자 위치, 금액 위치) 인덱스 쌍. 일자=짝수offset, 금액=홀수offset.
    """
    ncols = df.shape[1]
    pairs = []
    hi = min(end, ncols)
    i = start
    while i + 1 < hi:
        pairs.append((i, i + 1))
        i += 2
    return pairs


@dataclass
class PaymentDerived:
    has_payment: bool = False
    total_paid: float = 0.0
    paid_count: int = 0
    last_paid_date: Optional[_dt.date] = None
    last_paid_amount: float = 0.0
    within_180: bool = False
    within_365: bool = False
    within_730: bool = False
    count_ge_2: bool = False
    count_ge_3: bool = False
    total_ge_100k: bool = False
    total_ge_500k: bool = False
    amount_only_no_date: bool = False  # 입금일자 불명확하지만 금액 존재

    def as_row(self) -> dict:
        return {
            "입금_이력유무": self.has_payment,
            "입금_총액": self.total_paid,
            "입금_건수": self.paid_count,
            "입금_최근일": self.last_paid_date,
            "입금_최근액": self.last_paid_amount,
            "입금_최근180일": self.within_180,
            "입금_최근365일": self.within_365,
            "입금_최근730일": self.within_730,
            "입금_건수2이상": self.count_ge_2,
            "입금_건수3이상": self.count_ge_3,
            "입금_총액10만이상": self.total_ge_100k,
            "입금_총액50만이상": self.total_ge_500k,
            "입금_일자불명금액존재": self.amount_only_no_date,
        }


def _derive_one(values: List[Tuple[Optional[object], object]],
                ref_date: _dt.date, strict: bool = False) -> PaymentDerived:
    """(일자값, 금액값) 목록 → 파생값. 금액 ≤ 0 은 무시.

    strict=True(위치 기반 fallback 전용): 비표준 파일에서 임의 컬럼을 입금으로
    오인하지 않도록, 일자칸에 비일자 텍스트가 있거나 파싱된 날짜가 비현실
    구간(1990년 이전/평가 기준일+400일 이후)이면 해당 쌍을 무시한다.
    (일자칸이 빈 값 + 금액만 있는 '일자불명' 입금은 계속 인정)
    """
    d = PaymentDerived()
    dated_dates: List[_dt.date] = []
    dated_amounts: List[float] = []
    all_amounts: List[float] = []
    undated_amount_exists = False
    total = 0.0
    count = 0

    for date_val, amt_val in values:
        amt = util.to_number(amt_val)
        if amt is None or amt <= 0:
            continue  # 환급/취소 추정 또는 빈값
        pdate = util.parse_date(date_val) if date_val is not None else None
        if strict:
            ds = util.clean_str(date_val)
            if ds and pdate is None:
                continue  # 일자칸에 날짜 아닌 값 → 오인 방지
            if pdate is not None and not (
                    _dt.date(1990, 1, 1) <= pdate <= ref_date + _dt.timedelta(days=400)):
                continue  # 코드값 등이 serial 날짜로 오인된 경우
        count += 1
        total += amt
        all_amounts.append(amt)
        if pdate is not None:
            dated_dates.append(pdate)
            dated_amounts.append(amt)
        else:
            undated_amount_exists = True

    if count == 0:
        return d  # 입금 없음

    d.has_payment = True
    d.total_paid = total
    d.paid_count = count
    d.count_ge_2 = count >= 2
    d.count_ge_3 = count >= 3
    d.total_ge_100k = total >= 100_000   # 10만
    d.total_ge_500k = total >= 500_000   # 50만

    if dated_dates:
        # 최근일 = 가장 최신 날짜, 그 시점 금액
        idx = max(range(len(dated_dates)), key=lambda i: dated_dates[i])
        d.last_paid_date = dated_dates[idx]
        d.last_paid_amount = dated_amounts[idx]
        days = util.days_between(ref_date, d.last_paid_date)
        if days is not None and days >= 0:
            d.within_180 = days <= 180
            d.within_365 = days <= 365
            d.within_730 = days <= 730
        elif days is not None and days < 0:
            # 평가 기준일보다 미래 입금(데이터 오류) → 최근으로 간주하지 않음
            pass
    else:
        # 금액만 있고 일자 전부 불명 — 최근액은 확인된 입금액 중 최대값으로 근사
        d.amount_only_no_date = True
        d.last_paid_amount = max(all_amounts) if all_amounts else 0.0

    if undated_amount_exists and dated_dates:
        # 일부는 일자 있고 일부는 없음 → 이력은 dated 기준, 금액존재 플래그도 표시
        d.amount_only_no_date = True

    return d


def derive_payments(df: pd.DataFrame, ref_date: _dt.date) -> pd.DataFrame:
    """DataFrame 전체에 대해 입금 파생값 테이블을 만든다(index 정렬)."""
    columns = list(df.columns)
    named = find_payment_pairs(columns)

    rows: List[dict] = []
    if named:
        # named 컬럼 사용
        for _, r in df.iterrows():
            vals = []
            for dcol, acol in named:
                dv = r[dcol] if (dcol is not None and dcol in r) else None
                av = r[acol] if (acol is not None and acol in r) else None
                vals.append((dv, av))
            rows.append(_derive_one(vals, ref_date).as_row())
    else:
        # 위치 기반 fallback (strict: 임의 컬럼 오인 방지 가드)
        ppairs = positional_pairs(df)
        arr = df.values
        for i in range(len(df)):
            vals = []
            for dpos, apos in ppairs:
                dv = arr[i][dpos] if (dpos is not None and dpos < arr.shape[1]) else None
                av = arr[i][apos] if (apos is not None and apos < arr.shape[1]) else None
                vals.append((dv, av))
            rows.append(_derive_one(vals, ref_date, strict=True).as_row())

    return pd.DataFrame(rows, index=df.index)


def has_any_payment_columns(df: pd.DataFrame) -> bool:
    """입금 컬럼(named 또는 위치)이 존재하는지."""
    if find_payment_pairs(list(df.columns)):
        return True
    return df.shape[1] > 82  # 위치 fallback 가능성
