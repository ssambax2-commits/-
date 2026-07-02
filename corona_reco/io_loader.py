# -*- coding: utf-8 -*-
"""
io_loader.py — 데이터 로더
===========================
csv/xlsx 파일을 UTF-8/EUC-KR(CP949) 자동 판별로 읽고, 컬럼명을 정규화한다.
활동데이터/학습데이터/피드백파일 모두 이 로더를 통과한다.
"""
from __future__ import annotations

import io
import os
import re
from typing import Dict, List, Optional

import pandas as pd

# 인코딩 후보 (한국 실무 파일 대응)
_ENCODINGS = ["utf-8-sig", "utf-8", "cp949", "euc-kr"]

# 컬럼명 정규화용 별칭 맵 (canonical <- 여러 표기)
# key: 정규화된(공백제거) 별칭, value: canonical 컬럼명
COLUMN_ALIASES: Dict[str, str] = {}


def _register(canonical: str, *aliases: str):
    COLUMN_ALIASES[_norm_key(canonical)] = canonical
    for a in aliases:
        COLUMN_ALIASES[_norm_key(a)] = canonical


def _norm_key(name: str) -> str:
    """비교용 정규화: 공백/특수문자 제거, 소문자."""
    s = str(name).strip()
    s = re.sub(r"\s+", "", s)
    return s


# --- 활동/학습 데이터 주요 컬럼 별칭 등록 ---
_register("고객번호", "고객 번호", "고객No", "customer_no")
_register("대출번호", "대출 번호", "계좌번호", "loan_no")
_register("성명", "이름", "고객명", "name")
_register("주민등록번호", "주민번호", "주민등록 번호", "rrn")
_register("팀", "team")
_register("담당자", "담당")
_register("부담당자", "부담당", "실담당자")
_register("채권상태(대)", "채권상태대", "채권상태_대")
_register("채권상태(중)", "채권상태중", "채권상태_중", "상태중")
_register("민원여부", "민원")
_register("채무부존재소송", "채무부존재")
_register("세분류")
_register("회생", "회생상태", "채무조정상태")
_register("최종갱신일")
_register("최종갱신금액")
_register("현재원금", "현재 원금")
_register("현재OPB", "현재opb")
_register("현재연체이자")
_register("현재미수금")
_register("현재부족금")
_register("현총액", "현재총액")
_register("최초원금")
_register("최초미수금")
_register("최초원리금")
_register("매입당시OPB", "매입당시opb")
_register("현재OPB", "현재opb")
_register("대출일자")
_register("만기일자")
_register("최종이자수입일")
_register("최초연체일")
_register("대출이율")
_register("연체이율")
_register("약정일자")
_register("연체일수")
_register("원장상태", "원장 상태")
_register("다중계좌 활동/총건수", "다중계좌활동/총건수", "다중계좌건수", "다중계좌 활동총건수")
_register("다중계좌 원금합계", "다중계좌원금합계", "다중계좌 원금 합계", "다중계좌원금잔액")
_register("시효일자", "소멸시효일자")
_register("해제일자")
_register("등록사유발생일")
_register("매입일자")
_register("매입금액")
_register("매입회차")
_register("매입구분")
_register("채권구분", "채권 구분")
_register("차주구분", "차주 구분")
_register("상품명")
_register("상품명 세분류", "상품명세분류")
_register("양도횟수")
_register("법시행이후양도횟수")
_register("순번", "no")
_register("법인폰")
_register("내선번호")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """컬럼명을 canonical 표기로 정규화. 미등록 컬럼은 strip만."""
    rename = {}
    for col in df.columns:
        key = _norm_key(col)
        if key in COLUMN_ALIASES:
            rename[col] = COLUMN_ALIASES[key]
        else:
            rename[col] = str(col).strip()
    out = df.rename(columns=rename)
    # 중복 컬럼명 방어: 첫 컬럼만 유지
    out = out.loc[:, ~out.columns.duplicated()]
    return out


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    last_err: Optional[Exception] = None
    for enc in _ENCODINGS:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, dtype=str,
                               keep_default_na=True, na_values=[""])
        except (UnicodeDecodeError, Exception) as e:  # noqa: BLE001
            last_err = e
            continue
    raise ValueError(f"CSV 인코딩 판별 실패: {last_err}")


def load_table(path: str, sheet_name=0) -> pd.DataFrame:
    """파일 경로에서 표를 읽어 컬럼 정규화한 DataFrame 반환.

    - .csv/.txt : 인코딩 자동 판별
    - .xlsx/.xls: openpyxl 엔진
    문자열 컬럼은 dtype=str 로 읽어 고객번호/날짜 훼손을 막는다(숫자화는 사용처에서).
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".txt"):
        with open(path, "rb") as f:
            raw = f.read()
        df = _read_csv_bytes(raw)
    elif ext in (".xlsx", ".xlsm", ".xls"):
        # dtype=str 로 읽되, 엑셀 serial 날짜/숫자 보존을 위해 원본 유지
        df = pd.read_excel(path, sheet_name=sheet_name, dtype=object)
    else:
        raise ValueError(f"지원하지 않는 확장자입니다: {ext}")

    if isinstance(df, dict):  # sheet_name=None 등
        # 첫 시트 사용
        df = next(iter(df.values()))
    df = normalize_columns(df)
    return df


def load_excel_all_sheets(path: str) -> Dict[str, pd.DataFrame]:
    """엑셀의 전체 시트를 dict로. 피드백 파일(사용자시트+숨김시트) 처리용."""
    sheets = pd.read_excel(path, sheet_name=None, dtype=object)
    return {name: normalize_columns(df) for name, df in sheets.items()}


def get_col(df: pd.DataFrame, name: str, default=None) -> pd.Series:
    """정규화된 컬럼을 안전하게 가져온다. 없으면 default로 채운 Series."""
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def has_col(df: pd.DataFrame, name: str) -> bool:
    return name in df.columns


def list_missing_columns(df: pd.DataFrame, required: List[str]) -> List[str]:
    return [c for c in required if c not in df.columns]
