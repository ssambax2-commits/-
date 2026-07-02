# -*- coding: utf-8 -*-
"""
tests/synth.py — 합성 데이터 생성기
====================================
활동/학습 데이터를 재현 가능하게 생성한다(시드 고정). 입금열은 named
(입금일자N/입금액N) 형태로 붙인다. e2e·단위 테스트가 공유한다.
"""
from __future__ import annotations

import datetime as _dt
import random
from typing import List

import pandas as pd

_ROSTER_1 = ["민홍기", "김승한", "김영성", "최병인", "조은지"]
_ROSTER_2 = ["김종일", "서보문", "정재민"]
_ETC = ["박기타", "이미등록", ""]

_STATUS_MID_ALLOW = ["정상", "연체", "약속자", "재통화", "폐지_신복", "폐지_회생",
                     "폐지_파산", "약불자", "연락처", "기타"]
_REHAB_OK = ["", "정상", "부채증명원", "기각/취하/폐지", "심사반송"]
_BOND_TYPES = ["일반무담보", "일반무담보", "일반무담보", "담보부NPL"]
_BORROWER_TYPES = ["개인", "개인", "개인", "개인사업자"]
_PRODUCTS = ["신용대출", "담보대출", "신용카드", "할부금융"]


def _make_rrn(age: int, gender: str, ref_year: int = 2026) -> str:
    birth_year = ref_year - age
    yy = birth_year % 100
    if birth_year >= 2000:
        gcode = 3 if gender == "남" else 4
    else:
        gcode = 1 if gender == "남" else 2
    mm = random.randint(1, 12)
    dd = random.randint(1, 28)
    tail = random.randint(100000, 999999)
    return f"{yy:02d}{mm:02d}{dd:02d}-{gcode}{tail:06d}"


def _rand_date(start: _dt.date, end: _dt.date) -> _dt.date:
    delta = (end - start).days
    return start + _dt.timedelta(days=random.randint(0, max(1, delta)))


def generate(n: int = 2000, positive_rate: float = 0.06,
             ref_date: _dt.date = None, seed: int = 42,
             with_payments: bool = True, max_pairs: int = 5) -> pd.DataFrame:
    """합성 활동/학습 데이터. with_payments=True면 입금열 부착(학습데이터)."""
    if ref_date is None:
        ref_date = _dt.date(2026, 7, 1)
    random.seed(seed)

    all_names = _ROSTER_1 + _ROSTER_2 + _ETC
    rows = []
    n_customers = int(n * 0.8)  # 일부 다계좌 차주
    for i in range(n):
        cust_id = f"C{random.randint(1, n_customers):06d}"
        name = f"고객{cust_id[-4:]}"
        age = random.choice([25, 35, 45, 55, 65, 75])
        gender = random.choice(["남", "여"])
        bu_dam = random.choice(all_names)
        dam = random.choice(_ROSTER_1 + _ROSTER_2)
        principal = random.choice([80, 250, 800, 2500, 4500, 7000]) * 10000
        is_positive = random.random() < positive_rate

        # 대부분 활동, 일부 종결
        ledger = "활동" if random.random() < 0.85 else "해지"
        status_mid = random.choice(_STATUS_MID_ALLOW)
        rehab = random.choice(_REHAB_OK)

        buy_date = _rand_date(_dt.date(2018, 1, 1), _dt.date(2023, 12, 31))
        loan_date = _rand_date(_dt.date(2012, 1, 1), _dt.date(2018, 1, 1))
        delinq_date = _rand_date(loan_date, _dt.date(2019, 1, 1))
        reg_date = delinq_date + _dt.timedelta(days=90)
        expiry = _rand_date(_dt.date(2027, 1, 1), _dt.date(2032, 1, 1))

        row = {
            "순번": i + 1,
            "고객번호": cust_id,
            "대출번호": f"L{i:07d}",
            "성명": name,
            "주민등록번호": _make_rrn(age, gender),
            "팀": random.choice(["1팀", "2팀", "기타"]),
            "담당자": dam,
            "부담당자": bu_dam,
            "채권상태(대)": random.choice(["정상", "특수채권", "개인회생"]),
            "채권상태(중)": status_mid,
            "민원여부": "있음" if random.random() < 0.05 else "",
            "채무부존재소송": "",
            "회생": rehab,
            "원장상태": ledger,
            "현재원금": principal,
            "현재OPB": principal,
            "최초원금": principal + random.randint(0, 500000),
            "최초미수금": random.randint(0, 300000),
            "최초원리금": principal + random.randint(0, 800000),
            "매입당시OPB": principal + random.randint(0, 200000),
            "대출이율": round(random.uniform(3, 20), 2),
            "연체이율": round(random.uniform(6, 24), 2),
            "양도횟수": random.randint(0, 3),
            "법시행이후양도횟수": random.randint(0, 2),
            "다중계좌 활동/총건수": f"{random.randint(1,3)}/{random.randint(3,6)}",
            "다중계좌 원금합계": principal * random.randint(1, 4),
            "차주구분": random.choice(_BORROWER_TYPES),
            "채권구분": random.choice(_BOND_TYPES),
            "상품명": random.choice(_PRODUCTS),
            "담보세부종류": "부동산" if random.random() < 0.2 else "",
            "대출종류": random.choice(["신용", "담보"]),
            "매입일자": buy_date.strftime("%Y-%m-%d"),
            "대출일자": loan_date.strftime("%Y-%m-%d"),
            "최초연체일": delinq_date.strftime("%Y-%m-%d"),
            "등록사유발생일": reg_date.strftime("%Y-%m-%d"),
            "시효일자": expiry.strftime("%Y-%m-%d"),
            "최종이자수입일": _rand_date(_dt.date(2024, 1, 1), ref_date).strftime("%Y-%m-%d"),
            "최종갱신금액": principal,  # 입금신호로 쓰면 안 됨(진단용)
        }

        # 입금열 (positive만 채움)
        if with_payments:
            n_pairs = random.randint(1, 3) if is_positive else 0
            for p in range(1, max_pairs + 1):
                if p <= n_pairs:
                    pd_date = _rand_date(ref_date - _dt.timedelta(days=500), ref_date)
                    row[f"입금일자{p}"] = pd_date.strftime("%Y-%m-%d")
                    row[f"입금액{p}"] = random.choice([50000, 100000, 300000, 600000])
                else:
                    row[f"입금일자{p}"] = ""
                    row[f"입금액{p}"] = ""
            # positive인데 해지면: 그 입금으로 종결된 건(라벨=1 유지)
            if is_positive and random.random() < 0.3:
                row["원장상태"] = "해지"

        rows.append(row)

    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: str, encoding: str = "utf-8-sig"):
    df.to_csv(path, index=False, encoding=encoding)
    return path
