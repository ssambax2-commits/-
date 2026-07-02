# -*- coding: utf-8 -*-
"""
exclusions.py — 강제 제외 규칙 + 차주 단위 전파
=================================================
§3 규칙을 구현한다. 계좌 단위 제외와 차주(고객번호+성명) 단위 전파를 구분한다.
제외 시 반드시 사유를 기록한다. 폐지_신복/폐지_회생/폐지_파산은 절대 제외하지 않는다.
"""
from __future__ import annotations

import datetime as _dt

import pandas as pd

from . import config, util
from .io_loader import get_col

# util 공용 헬퍼 alias (하위 호환)
_is_present = util.is_present
_clean_str = util.clean_str


def borrower_key(gonum, name) -> str:
    """차주 키: 고객번호 + 성명."""
    return f"{_clean_str(gonum)}|{_clean_str(name)}"


def apply_exclusions(df: pd.DataFrame, ref_date: _dt.date,
                     exclude_expired_prescription: bool = None) -> pd.DataFrame:
    """제외 판정 컬럼을 붙인 DataFrame 반환.

    추가 컬럼:
      _제외여부(bool), _제외사유(str), _제외단위(계좌/차주/''), _차주키(str)
    """
    if exclude_expired_prescription is None:
        exclude_expired_prescription = config.EXCLUDE_EXPIRED_PRESCRIPTION_DEFAULT

    n = len(df)
    ledger = get_col(df, "원장상태")
    status_mid = get_col(df, "채권상태(중)")
    nonexist_suit = get_col(df, "채무부존재소송")
    rehab = get_col(df, "회생")
    expiry = get_col(df, "시효일자")
    gonum = get_col(df, "고객번호")
    name = get_col(df, "성명")

    excluded = [False] * n
    reasons = [[] for _ in range(n)]
    units = [""] * n
    bkeys = [""] * n

    # 차주단위 전파를 위해 먼저 계좌별 판정 + 사람상태 표시
    borrower_flag = {}  # borrower_key -> 전파사유(set)

    for i in range(n):
        bk = borrower_key(gonum.iloc[i], name.iloc[i])
        bkeys[i] = bk
        acct_reasons = []
        propagate = set()

        # 규칙 1: 원장상태 ≠ 활동
        led = _clean_str(ledger.iloc[i])
        st_mid = _clean_str(status_mid.iloc[i])

        # 폐지_* 는 절대 제외 금지 (안내 가능) — 다른 규칙과 무관히 상태중 제외에서 보호
        is_never_exclude_status = st_mid in config.STATUS_MID_NEVER_EXCLUDE

        if led and led != config.LEDGER_ACTIVE:
            acct_reasons.append(f"원장상태 비활동({led})")

        # 규칙 2: 채권상태(중) 강제제외
        if st_mid in config.STATUS_MID_EXCLUDE and not is_never_exclude_status:
            acct_reasons.append(f"채권상태(중)={st_mid}")
            if st_mid in config.STATUS_MID_EXCLUDE_BORROWER:
                propagate.add(f"차주단위 전파(채권상태중={st_mid})")

        # 규칙 3: 채무부존재소송 있음
        if _is_present(nonexist_suit.iloc[i]):
            acct_reasons.append("채무부존재소송")

        # 규칙 4: 회생 연락금지 → 차주단위 전파
        rh = _clean_str(rehab.iloc[i])
        if rh in config.REHAB_EXCLUDE:
            acct_reasons.append(f"회생={rh}(연락금지)")
            propagate.add(f"차주단위 전파(회생={rh})")
        # 부채증명원 등 REHAB_IGNORE 는 무시 (아무 것도 안 함)

        # 규칙 5: 시효일자 경과(옵션)
        if exclude_expired_prescription:
            exp_date = util.parse_date(expiry.iloc[i])
            if exp_date is not None and exp_date < ref_date:
                acct_reasons.append(f"소멸시효완성({exp_date.isoformat()})")

        if acct_reasons:
            excluded[i] = True
            reasons[i] = acct_reasons
            units[i] = "계좌"

        if propagate:
            borrower_flag.setdefault(bk, set()).update(propagate)

    # 차주단위 전파: 사람상태(회생/사망/면책) 계좌가 있으면 그 차주 전 계좌 제외
    for i in range(n):
        bk = bkeys[i]
        if bk in borrower_flag:
            prop = borrower_flag[bk]
            if not excluded[i]:
                excluded[i] = True
                units[i] = "차주"
                reasons[i] = list(prop)
            else:
                # 이미 계좌사유 있으면 전파사유도 병기
                for p in prop:
                    if p not in reasons[i]:
                        reasons[i].append(p)
                if units[i] != "차주":
                    units[i] = "계좌+차주"

    out = df.copy()
    out["_제외여부"] = excluded
    out["_제외사유"] = ["; ".join(r) for r in reasons]
    out["_제외단위"] = units
    out["_차주키"] = bkeys
    return out


def special_bond_diagnostic(df: pd.DataFrame) -> pd.Series:
    """채권상태(대)=특수채권 & 원장상태=활동 → 진단 메시지(제외 아님)."""
    big = get_col(df, "채권상태(대)")
    ledger = get_col(df, "원장상태")
    msgs = []
    for i in range(len(df)):
        b = _clean_str(big.iloc[i])
        led = _clean_str(ledger.iloc[i])
        if b == config.STATUS_BIG_SPECIAL and led == config.LEDGER_ACTIVE:
            msgs.append("특수채권 상태값 확인 필요")
        else:
            msgs.append("")
    return pd.Series(msgs, index=df.index)
