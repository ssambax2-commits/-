# -*- coding: utf-8 -*-
"""
store.py — SQLite 스키마·누적·dedup·모델메타 (§6-2)
====================================================
로컬 SQLite 파일 1개에 추천/피드백/모델메타/세그먼트통계를 자동 누적한다.
사용자가 파일을 수동으로 들고 다니지 않는다.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sqlite3
from typing import Dict, List, Optional

import pandas as pd

from . import config


SCHEMA = {
    "recommendations": """
        CREATE TABLE IF NOT EXISTS recommendations (
            추천월 TEXT NOT NULL,
            고객번호 TEXT NOT NULL,
            성명 TEXT,
            대출번호 TEXT NOT NULL,
            부담당자 TEXT,
            팀 TEXT,
            등급 TEXT,
            final_score REAL,
            base_score REAL,
            paid_similarity_score REAL,
            payment_history_score REAL,
            burden_score REAL,
            external_prior_adj REAL,
            sensitive_penalty REAL,
            feedback_adj REAL,
            원금잔액 REAL,
            다중계좌원금잔액 REAL,
            추천사유 TEXT,
            피처벡터 TEXT,
            모델버전 TEXT,
            등록시각 TEXT,
            회생 TEXT,
            채권상태중 TEXT,
            나이대 TEXT,
            개인사업자 INTEGER,
            담보부NPL INTEGER,
            PRIMARY KEY (추천월, 고객번호, 대출번호)
        )
    """,
    "feedback": """
        CREATE TABLE IF NOT EXISTS feedback (
            추천월 TEXT NOT NULL,
            고객번호 TEXT NOT NULL,
            대출번호 TEXT NOT NULL,
            실제입금여부 INTEGER,
            실제입금액 REAL,
            입금일자 TEXT,
            원금잔액 REAL,
            회수비율 REAL,
            성공여부 INTEGER,
            등록시각 TEXT,
            PRIMARY KEY (추천월, 고객번호, 대출번호)
        )
    """,
    "model_meta": """
        CREATE TABLE IF NOT EXISTS model_meta (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            모델버전 TEXT,
            학습일 TEXT,
            positive수 INTEGER,
            negative수 INTEGER,
            지표 TEXT,
            파라미터경로 TEXT,
            채택여부 INTEGER
        )
    """,
    "segment_stats": """
        CREATE TABLE IF NOT EXISTS segment_stats (
            갱신월 TEXT NOT NULL,
            세그먼트키 TEXT NOT NULL,
            관측수 INTEGER,
            입금성공수 INTEGER,
            EB보정율 REAL,
            EB회수비율 REAL,
            PRIMARY KEY (갱신월, 세그먼트키)
        )
    """,
}


class Store:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self):
        cur = self.conn.cursor()
        for ddl in SCHEMA.values():
            cur.execute(ddl)
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass

    # ------------------------------------------------------------------
    # recommendations
    # ------------------------------------------------------------------
    def save_recommendations(self, month: str, borrowers: pd.DataFrame,
                             feature_json_by_key: Dict[str, str],
                             model_version: str) -> int:
        """추천 결과 저장(dedup: 추천월+고객번호+대출번호 최신유지).

        feature_json_by_key: '대표대출번호' -> 피처벡터 JSON 문자열.
        """
        now = _dt.datetime.now().isoformat(timespec="seconds")
        cur = self.conn.cursor()
        cnt = 0
        for _, r in borrowers.iterrows():
            loan_no = str(r.get("대표대출번호", ""))
            fv = feature_json_by_key.get(loan_no, "")
            cur.execute("""
                INSERT OR REPLACE INTO recommendations
                (추천월, 고객번호, 성명, 대출번호, 부담당자, 팀, 등급, final_score,
                 base_score, paid_similarity_score, payment_history_score, burden_score,
                 external_prior_adj, sensitive_penalty, feedback_adj,
                 원금잔액, 다중계좌원금잔액, 추천사유, 피처벡터, 모델버전, 등록시각,
                 회생, 채권상태중, 나이대, 개인사업자, 담보부NPL)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                month, str(r.get("고객번호", "")), str(r.get("성명", "")), loan_no,
                str(r.get("부담당자", "")), str(r.get("팀", "")), str(r.get("등급", "")),
                float(r.get("borrower_score", 0) or 0),
                float(r.get("base_score", 0) or 0),
                float(r.get("paid_similarity_score", 0) or 0),
                float(r.get("payment_history_score", 0) or 0),
                float(r.get("burden_score", 0) or 0),
                float(r.get("external_prior_adj", 0) or 0),
                float(r.get("sensitive_penalty", 0) or 0),
                float(r.get("feedback_adj", 0) or 0),
                float(r.get("원금잔액", 0) or 0),
                float(r.get("다중계좌원금잔액", 0) or 0),
                str(r.get("추천사유", "")), fv, model_version, now,
                str(r.get("회생", "")), str(r.get("채권상태(중)", "")),
                str(r.get("나이대", "")),
                int(r.get("개인사업자", 0) or 0),
                int(1 if str(r.get("collateral_key", "normal")) != "normal" else 0),
            ))
            cnt += 1
        self.conn.commit()
        return cnt

    def get_recommendations(self, month: Optional[str] = None) -> pd.DataFrame:
        q = "SELECT * FROM recommendations"
        params = ()
        if month:
            q += " WHERE 추천월 = ?"
            params = (month,)
        return pd.read_sql_query(q, self.conn, params=params)

    # ------------------------------------------------------------------
    # feedback
    # ------------------------------------------------------------------
    def save_feedback(self, rows: pd.DataFrame) -> int:
        """피드백 저장. rows 컬럼: 추천월,고객번호,대출번호,실제입금여부,실제입금액,
        입금일자,원금잔액,회수비율,성공여부. dedup 최신유지."""
        now = _dt.datetime.now().isoformat(timespec="seconds")
        cur = self.conn.cursor()
        cnt = 0
        for _, r in rows.iterrows():
            cur.execute("""
                INSERT OR REPLACE INTO feedback
                (추천월, 고객번호, 대출번호, 실제입금여부, 실제입금액, 입금일자,
                 원금잔액, 회수비율, 성공여부, 등록시각)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            """, (
                str(r.get("추천월", "")), str(r.get("고객번호", "")), str(r.get("대출번호", "")),
                _to_int_or_none(r.get("실제입금여부")),
                _to_float_or_none(r.get("실제입금액")),
                (str(r.get("입금일자")) if r.get("입금일자") not in (None, "") else None),
                _to_float_or_none(r.get("원금잔액")),
                _to_float_or_none(r.get("회수비율")),
                _to_int_or_none(r.get("성공여부")),
                now,
            ))
            cnt += 1
        self.conn.commit()
        return cnt

    def get_feedback(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM feedback", self.conn)

    def get_feedback_with_segments(self) -> pd.DataFrame:
        """feedback + recommendations(세그먼트 변수) 조인 → EB 캘리브레이션용."""
        q = """
            SELECT f.추천월, f.고객번호, f.대출번호, f.실제입금액, f.원금잔액,
                   f.회수비율, f.성공여부,
                   r.부담당자, r.팀, r.회생, r.채권상태중, r.나이대,
                   r.개인사업자, r.담보부NPL, r.원금잔액 AS 추천원금잔액
            FROM feedback f
            JOIN recommendations r
              ON f.추천월 = r.추천월 AND f.고객번호 = r.고객번호 AND f.대출번호 = r.대출번호
            WHERE f.성공여부 IS NOT NULL
        """
        return pd.read_sql_query(q, self.conn)

    def get_training_pool(self) -> pd.DataFrame:
        """recommendations(피처벡터) + feedback(성공여부) 조인 → ML 재학습 풀."""
        q = """
            SELECT r.추천월, r.고객번호, r.대출번호, r.피처벡터, f.성공여부, f.회수비율
            FROM recommendations r
            JOIN feedback f
              ON r.추천월 = f.추천월 AND r.고객번호 = f.고객번호 AND r.대출번호 = f.대출번호
            WHERE r.피처벡터 IS NOT NULL AND r.피처벡터 <> '' AND f.성공여부 IS NOT NULL
        """
        return pd.read_sql_query(q, self.conn)

    # ------------------------------------------------------------------
    # model_meta
    # ------------------------------------------------------------------
    def save_model_meta(self, model_version: str, positives: int, negatives: int,
                        metrics: Dict, params_path: str, adopted: bool) -> None:
        cur = self.conn.cursor()
        cur.execute("""
            INSERT INTO model_meta
            (모델버전, 학습일, positive수, negative수, 지표, 파라미터경로, 채택여부)
            VALUES (?,?,?,?,?,?,?)
        """, (
            model_version, _dt.datetime.now().isoformat(timespec="seconds"),
            int(positives), int(negatives), json.dumps(metrics, ensure_ascii=False),
            params_path, 1 if adopted else 0,
        ))
        self.conn.commit()

    def get_latest_model_meta(self) -> Optional[dict]:
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM model_meta WHERE 채택여부=1 ORDER BY id DESC LIMIT 1")
        row = cur.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------
    # segment_stats
    # ------------------------------------------------------------------
    def save_segment_stats(self, month: str, stats: List[dict]) -> int:
        cur = self.conn.cursor()
        cnt = 0
        for s in stats:
            cur.execute("""
                INSERT OR REPLACE INTO segment_stats
                (갱신월, 세그먼트키, 관측수, 입금성공수, EB보정율, EB회수비율)
                VALUES (?,?,?,?,?,?)
            """, (
                month, str(s.get("세그먼트키", "")), int(s.get("관측수", 0)),
                int(s.get("입금성공수", 0)), float(s.get("EB보정율", 0) or 0),
                float(s.get("EB회수비율", 0) or 0),
            ))
            cnt += 1
        self.conn.commit()
        return cnt

    def get_segment_stats(self) -> pd.DataFrame:
        return pd.read_sql_query("SELECT * FROM segment_stats", self.conn)

    def total_feedback_observations(self) -> int:
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM feedback WHERE 성공여부 IS NOT NULL")
        return int(cur.fetchone()[0])


def _to_int_or_none(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return int(v)
    except (ValueError, TypeError):
        return None


def _to_float_or_none(v):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    try:
        return float(v)
    except (ValueError, TypeError):
        return None


def default_data_dir(base_dir: Optional[str] = None) -> str:
    """exe 옆 사용자쓰기 가능 폴더(data/). 없으면 생성."""
    if base_dir is None:
        base_dir = os.getcwd()
    d = os.path.join(base_dir, config.DATA_DIRNAME)
    os.makedirs(d, exist_ok=True)
    return d
