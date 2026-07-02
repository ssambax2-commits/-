# -*- coding: utf-8 -*-
"""
util.py — 공용 유틸리티
========================
날짜 파싱(엑셀 serial + 문자열), 숫자 강제변환, 주민등록번호 → 나이대/성별
계산(원본 즉시 폐기)을 담는다. 특정 모듈에 종속되지 않는 순수 함수들만 둔다.
"""
from __future__ import annotations

import datetime as _dt
import math
import re
from typing import Optional, Tuple

import numpy as np
import pandas as pd

# 엑셀 serial date epoch (1900 윤년 버그 보정: 1899-12-30 기준)
_EXCEL_EPOCH = _dt.datetime(1899, 12, 30)


# 값이 "있음/참"으로 해석되지 않는 부정 토큰
_NEGATIVE_TOKENS = {
    "", "없음", "무", "n", "no", "0", "false", "해당없음",
    "nan", "none", "null", "-",
}


def clean_str(val) -> str:
    """None/NaN 안전 문자열 변환(strip)."""
    if val is None:
        return ""
    if isinstance(val, float) and math.isnan(val):
        return ""
    return str(val).strip()


def is_present(val) -> bool:
    """'있음' 계열 판정(민원여부/채무부존재소송 등)."""
    if val is None:
        return False
    if isinstance(val, float) and math.isnan(val):
        return False
    s = str(val).strip().lower()
    return s not in _NEGATIVE_TOKENS


def to_number(value) -> Optional[float]:
    """콤마/원화기호/공백 등이 섞인 값을 float로. 실패 시 None."""
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return None
        return float(value)
    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "-"):
        return None
    # 숫자/부호/소수점만 남기기
    s = s.replace(",", "").replace("₩", "").replace("원", "").strip()
    s = re.sub(r"[^0-9.\-+eE]", "", s)
    if s in ("", "-", "+", ".", "-.", "+."):
        return None
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def to_number_or(value, default: float = 0.0) -> float:
    """to_number 결과가 None이면 default."""
    n = to_number(value)
    return default if n is None else n


def parse_date(value) -> Optional[_dt.date]:
    """엑셀 serial number 및 다양한 문자열 날짜를 date로 파싱.

    지원: 엑셀 serial(정수/실수), 'YYYY-MM-DD', 'YYYY.MM.DD', 'YYYY/MM/DD',
          'YYYYMMDD', datetime/date/Timestamp 객체.
    파싱 실패 시 None.
    """
    if value is None:
        return None
    # pandas/파이썬 날짜 객체
    if isinstance(value, (pd.Timestamp,)):
        if pd.isna(value):
            return None
        return value.date()
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value

    # NaN 방어
    if isinstance(value, float) and math.isnan(value):
        return None

    # 엑셀 serial (숫자형)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _excel_serial_to_date(float(value))

    s = str(value).strip()
    if s == "" or s.lower() in ("nan", "none", "null", "-"):
        return None

    # 숫자만 이루어진 문자열 → serial 또는 YYYYMMDD
    if re.fullmatch(r"\d+(\.\d+)?", s):
        num = float(s)
        # 8자리 정수는 YYYYMMDD 로 우선 해석
        if s.isdigit() and len(s) == 8:
            d = _try_yyyymmdd(s)
            if d is not None:
                return d
        # 그 외 숫자는 엑셀 serial 로 해석 (합리적 범위)
        if 1 <= num <= 100000:
            return _excel_serial_to_date(num)

    # 구분자 있는 문자열
    m = re.match(r"^(\d{4})[.\-/년]\s*(\d{1,2})[.\-/월]\s*(\d{1,2})", s)
    if m:
        y, mo, da = int(m.group(1)), int(m.group(2)), int(m.group(3))
        return _safe_date(y, mo, da)

    # pandas 최종 시도
    try:
        ts = pd.to_datetime(s, errors="coerce")
        if pd.isna(ts):
            return None
        return ts.date()
    except (ValueError, TypeError):
        return None


def _excel_serial_to_date(num: float) -> Optional[_dt.date]:
    try:
        return (_EXCEL_EPOCH + _dt.timedelta(days=num)).date()
    except (OverflowError, ValueError):
        return None


def _try_yyyymmdd(s: str) -> Optional[_dt.date]:
    try:
        y, mo, da = int(s[0:4]), int(s[4:6]), int(s[6:8])
        return _safe_date(y, mo, da)
    except (ValueError, TypeError):
        return None


def _safe_date(y: int, mo: int, da: int) -> Optional[_dt.date]:
    try:
        return _dt.date(y, mo, da)
    except ValueError:
        return None


def days_between(later: Optional[_dt.date], earlier: Optional[_dt.date]) -> Optional[int]:
    """later - earlier (일수). 하나라도 None이면 None."""
    if later is None or earlier is None:
        return None
    return (later - earlier).days


def months_between(later: Optional[_dt.date], earlier: Optional[_dt.date]) -> Optional[float]:
    """later - earlier (개월수, 근사). 하나라도 None이면 None."""
    d = days_between(later, earlier)
    if d is None:
        return None
    return d / 30.4375


def safe_log1p(value) -> float:
    """log(1+x). 음수/None은 0으로 방어."""
    n = to_number(value)
    if n is None or n < 0:
        return 0.0
    return float(np.log1p(n))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


# ---------------------------------------------------------------------------
# 주민등록번호 → 나이대/성별
#   원본은 절대 저장/출력하지 않는다. 앞자리로만 계산 후 즉시 폐기.
# ---------------------------------------------------------------------------
def rrn_to_age_gender(rrn, ref_date: Optional[_dt.date] = None) -> Tuple[str, str]:
    """주민등록번호 → (나이대, 성별). 원본은 반환하지 않는다.

    반환 나이대: '20대이하','30대','40대','50대','60대','70대','80대이상','unknown'
    반환 성별:   '남','여','unknown'  (성별은 점수 미반영, 공정성 진단 표시용)
    """
    if rrn is None:
        return ("unknown", "unknown")
    s = re.sub(r"[^0-9]", "", str(rrn))
    if len(s) < 7:
        # 성별코드(7번째 자리)가 없으면 생년만이라도 시도
        if len(s) < 6:
            return ("unknown", "unknown")
        gender_code = None
    else:
        gender_code = s[6]

    yy = s[0:2]
    mm = s[2:4]
    dd = s[4:6]
    try:
        yy_i = int(yy)
        mm_i = int(mm)
        dd_i = int(dd)
    except ValueError:
        return ("unknown", "unknown")

    # 세기/성별 판정 (성별코드 기준)
    century = None
    gender = "unknown"
    if gender_code is not None and gender_code.isdigit():
        gc = int(gender_code)
        if gc in (1, 2, 5, 6):
            century = 1900
        elif gc in (3, 4, 7, 8):
            century = 2000
        elif gc in (9, 0):
            century = 1800
        if gc in (1, 3, 5, 7, 9):
            gender = "남"
        elif gc in (2, 4, 6, 8, 0):
            gender = "여"

    if century is None:
        # 성별코드 없거나 불명 → 2자리 연도 휴리스틱(현재<=25면 2000년대)
        century = 2000 if yy_i <= 25 else 1900

    birth_year = century + yy_i
    birth = _safe_date(birth_year, mm_i if 1 <= mm_i <= 12 else 1,
                       dd_i if 1 <= dd_i <= 31 else 1)
    if birth is None:
        return ("unknown", gender)

    ref = ref_date or _dt.date.today()
    age = ref.year - birth.year - ((ref.month, ref.day) < (birth.month, birth.day))
    return (age_to_band(age), gender)


def age_to_band(age: int) -> str:
    if age < 0 or age > 130:
        return "unknown"
    if age < 30:
        return "20대이하"
    if age < 40:
        return "30대"
    if age < 50:
        return "40대"
    if age < 60:
        return "50대"
    if age < 70:
        return "60대"
    if age < 80:
        return "70대"
    return "80대이상"
