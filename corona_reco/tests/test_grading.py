# -*- coding: utf-8 -*-
"""차주통합/등급/상한/최소보장/팀배정 테스트 (§9: 8, 12)."""
import datetime as _dt

import numpy as np
import pandas as pd

from corona_reco import config, grading

REF = _dt.date(2026, 7, 1)


# 8. 부담당자 공란 → 담당자 fallback → 없으면 미지정
def test_manager_fallback():
    assert grading.manager_of_row("김종일", "홍길동") == "김종일"
    assert grading.manager_of_row("", "민홍기") == "민홍기"
    assert grading.manager_of_row("", "") == config.UNASSIGNED_LABEL
    assert grading.manager_of_row(None, None) == config.UNASSIGNED_LABEL


def test_assign_managers_series():
    df = pd.DataFrame([
        {"부담당자": "김종일", "담당자": "누구"},
        {"부담당자": "", "담당자": "민홍기"},
        {"부담당자": "", "담당자": ""},
    ])
    m = grading.assign_managers(df)
    assert list(m) == ["김종일", "민홍기", config.UNASSIGNED_LABEL]


# 12. 팀 매핑: roster 우선, 미등록 → 기타
def test_team_mapping():
    assert grading.team_of_manager("김종일") == "2팀"
    assert grading.team_of_manager("서보문") == "2팀"
    assert grading.team_of_manager("민홍기") == "1팀"
    assert grading.team_of_manager("조은지") == "1팀"
    assert grading.team_of_manager("박미등록") == "기타"
    assert grading.team_of_manager("") == "기타"


def test_team_roster_beats_data():
    """데이터 팀 컬럼과 충돌해도 roster가 이긴다(불일치 카운트)."""
    df = pd.DataFrame([
        {"부담당자": "김종일", "담당자": "", "팀": "1팀"},  # roster=2팀, 데이터=1팀 → 충돌
        {"부담당자": "민홍기", "담당자": "", "팀": "1팀"},  # 일치
    ])
    m = grading.assign_managers(df)
    assert grading.team_mismatch_count(df, m) == 1


def test_grade_from_score():
    cuts = config.GRADE_CUTS
    assert grading.grade_from_score(75, cuts) == "S"
    assert grading.grade_from_score(60, cuts) == "A"
    assert grading.grade_from_score(45, cuts) == "B"
    assert grading.grade_from_score(30, cuts) == "C"
    assert grading.grade_from_score(10, cuts) == "D"


def _borrowers(n, score, manager="민홍기", cap="S"):
    return pd.DataFrame([{
        "부담당자": manager, "borrower_score": score, "등급상한": cap,
        "고객번호": f"C{i}", "성명": f"N{i}",
    } for i in range(n)])


def test_assignee_cap_s():
    b = _borrowers(25, 80)  # 전원 S 자격
    b = grading.assign_grades(b)
    assert (b["등급"] == "S").sum() == 25
    b = grading.apply_assignee_caps(b, cap_s=20, cap_a=100)
    assert (b["등급"] == "S").sum() == 20  # 초과 5명 하향
    assert (b["등급"] == "A").sum() == 5


def test_assignee_cap_a():
    b = _borrowers(130, 60)  # 전원 A 자격
    b = grading.assign_grades(b)
    b = grading.apply_assignee_caps(b, cap_s=20, cap_a=100)
    assert (b["등급"] == "A").sum() == 100
    assert (b["등급"] == "B").sum() == 30


def test_collateral_grade_cap():
    b = _borrowers(1, 80, cap="B")  # 점수는 S지만 상한 B
    b = grading.assign_grades(b)
    assert b["등급"].iloc[0] == "B"


def test_min_guarantee_fills_to_20():
    # 5명 전원 D → 최소 20 보장이지만 인원 부족 → 전원 추천
    b = _borrowers(5, 10)
    b = grading.assign_grades(b)
    b = grading.apply_assignee_caps(b)
    b = grading.apply_min_guarantee(b, min_reco=20)
    assert b["추천여부"].all()  # 전원(5명)


def test_min_guarantee_promotes_top_d():
    # 25명 전원 D → 상위 20명만 추천여부 True (등급은 D 유지)
    b = pd.DataFrame([{
        "부담당자": "민홍기", "borrower_score": 24 - i * 0.5, "등급상한": "S",
        "고객번호": f"C{i}", "성명": f"N{i}",
    } for i in range(25)])
    b = grading.assign_grades(b)
    b = grading.apply_assignee_caps(b)
    b = grading.apply_min_guarantee(b, min_reco=20)
    assert b["추천여부"].sum() == 20
    assert (b["등급"] == "D").all()  # 억지 승급 없음


def test_borrower_score_formula():
    """borrower_score = 0.7*max + 0.3*mean - 다중계좌패널티."""
    df = pd.DataFrame([
        {"_차주키": "K1", "고객번호": "C1", "성명": "김", "대출번호": "L1",
         "현재원금": 1000000, "회생": "", "채권상태(중)": "정상"},
        {"_차주키": "K1", "고객번호": "C1", "성명": "김", "대출번호": "L2",
         "현재원금": 2000000, "회생": "", "채권상태(중)": "정상"},
    ])
    scores = pd.DataFrame({
        "final_score": [80.0, 40.0], "collateral_cap": ["S", "S"],
        "collateral_key": ["normal", "normal"], "base_score": [50, 50],
        "paid_similarity_score": [50, 50], "payment_history_score": [0, 0],
        "burden_score": [90, 90], "external_prior_adj": [0, 0],
        "feedback_adj": [0, 0], "sensitive_penalty": [0, 0],
    }, index=df.index)
    pay = pd.DataFrame({
        "입금_이력유무": [False, False], "입금_최근365일": [False, False],
        "입금_최근180일": [False, False],
    }, index=df.index)
    m = grading.assign_managers(df)
    t = grading.assign_teams(m)
    b = grading.aggregate_borrowers(df, scores, pay, m, t)
    # 2계좌 → 패널티 없음(3건 미만). 0.7*80 + 0.3*60 = 56+18 = 74
    assert abs(b["borrower_score"].iloc[0] - 74.0) < 1e-6
