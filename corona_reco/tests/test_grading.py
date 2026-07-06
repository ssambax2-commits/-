# -*- coding: utf-8 -*-
"""v4 3단계 등급(즉시/당월/보류) · 150/200 상한 · 500만 필터 · 랜덤박스 테스트."""
import datetime as _dt

import numpy as np
import pandas as pd

from corona_reco import config, grading

TI, TM, TH = config.TIER_IMMEDIATE, config.TIER_MONTH, config.TIER_HOLD


# --- 담당자/팀 (불변) ---
def test_manager_fallback():
    assert grading.manager_of_row("김종일", "홍") == "김종일"
    assert grading.manager_of_row("", "민홍기") == "민홍기"
    assert grading.manager_of_row("", "") == config.UNASSIGNED_LABEL


def test_team_mapping():
    assert grading.team_of_manager("김종일") == "2팀"
    assert grading.team_of_manager("민홍기") == "1팀"
    assert grading.team_of_manager("박미등록") == "기타"


# --- 픽스처 ---
def _borrowers(specs, manager="민홍기"):
    """specs: (score, 합산잔액) 또는 (score, 합산잔액, dep, base, sim)."""
    rows = []
    for i, sp in enumerate(specs):
        score, multi = sp[0], sp[1]
        dep = sp[2] if len(sp) > 2 else 0.0
        base = sp[3] if len(sp) > 3 else 70
        sim = sp[4] if len(sp) > 4 else 70
        rows.append({
            "부담당자": manager, "borrower_score": score,
            "동일차주계좌합산원금잔액": multi, "예상회수1M": 0, "예상회수3M": 0,
            "최근입금일": None, "최근입금의존도": dep,
            "base_score": base, "paid_similarity_score": sim,
            "고객번호": f"C{i:03d}", "성명": f"N{i}", "대표대출번호": f"L{i:03d}",
        })
    return pd.DataFrame(rows)


# --- 즉시 경계(데이터 주도) ---
def test_immediate_boundary_gap():
    scores = np.array([95, 94, 93, 60, 59, 58, 57, 56, 55, 54], dtype=float)
    b = grading._immediate_boundary(scores)
    assert b == 93.0   # 93→60 갭에서 분할 → 상위 3명 즉시


def test_original_tier_immediate_month_hold():
    # 갭(93→60)이 상위 50% 안에 들도록 10명 분포 + floor 미만 1명
    specs = [(95, 20_000_000), (94, 20_000_000), (93, 20_000_000)]   # 즉시
    specs += [(60 - i, 20_000_000) for i in range(7)]                # 당월(60..54)
    specs += [(30, 20_000_000)]                                      # floor 미만 → 보류
    out = grading.assign_original_tier(_borrowers(specs))
    assert (out["원등급"] == TI).sum() == 3
    assert (out["원등급"] == TM).sum() == 7
    assert (out["원등급"] == TH).sum() == 1
    assert "보류" in out[out["borrower_score"] == 30].iloc[0]["등급하향사유"]


# --- §2 500만 하한 ---
def test_small_borrower_excluded():
    b = _borrowers([(90, 4_000_000), (90, 6_000_000)])  # 1명은 400만(<500만)
    out = grading.assign_original_tier(b)
    small = out[out["동일차주계좌합산원금잔액"] == 4_000_000].iloc[0]
    big = out[out["동일차주계좌합산원금잔액"] == 6_000_000].iloc[0]
    assert small["소액제외여부"] and small["원등급"] == TH
    assert "500만원 이하" in small["등급하향사유"]
    assert not big["소액제외여부"] and big["원등급"] != TH


# --- §1 150/200 상한 + 초과편입 ---
def _run(b, base_cap=3, hard_cap=5):
    b = grading.assign_original_tier(b)
    b = grading.apply_assignee_caps(b, base_cap=base_cap, hard_cap=hard_cap)
    b = grading.apply_recommendation_flags(b)
    b = grading.assign_ranks(b)
    return b


def test_cap_normal_overflow_hold():
    # 전원 동점 80점, 잔액만 상이 → 정상3 + 초과편입2(추천) + 보류1
    balances = [90, 80, 70, 60, 50, 40]  # 백만
    b = _borrowers([(80, x * 1_000_000) for x in balances])
    out = _run(b, base_cap=3, hard_cap=5)
    assert (out["추천여부"]).sum() == 5            # 3 정상 + 2 초과편입
    assert (out["최종등급"] == TH).sum() == 1      # 하드캡 밖 1
    assert (out["정원초과편입여부"]).sum() == 2
    # 초과편입은 잔액 4·5순위(6000만·5000만), 보류는 최저 잔액(4000만)
    held = out[out["최종등급"] == TH].iloc[0]
    assert held["동일차주계좌합산원금잔액"] == 40_000_000
    assert all("초과" in s for s in out.loc[out["정원초과편입여부"], "상한하향사유"])
    assert all("보류" in s for s in out.loc[out["최종등급"] == TH, "상한하향사유"])


def test_overflow_threshold_blocks_low_score():
    # 정상편입 점수 높고(90..), 151~200 구간 점수 낮으면(50) 초과편입 임계 미달 → 보류
    b = _borrowers([(90, 10_000_000), (89, 10_000_000), (88, 10_000_000),
                    (50, 10_000_000), (49, 10_000_000)])
    out = _run(b, base_cap=3, hard_cap=5)
    assert (out["정원초과편입여부"]).sum() == 0
    assert (out["최종등급"] == TH).sum() == 2
    assert all("임계" in s or "보류" in s
               for s in out.loc[out["최종등급"] == TH, "상한하향사유"])


def test_hard_cap_absolute():
    b = _borrowers([(80, 10_000_000)] * 8)
    b["고객번호"] = [f"C{i:03d}" for i in range(8)]
    out = _run(b, base_cap=3, hard_cap=5)
    # 하드캡 5 초과 3명은 무조건 보류
    assert (out["최종등급"] == TH).sum() == 3
    assert (out["추천여부"]).sum() == 5


def test_tie_break_balance_at_boundary():
    # 경계(정원3)에서 동점 → 차주합산 잔액 큰 3명이 유지
    b = _borrowers([(80, b_) for b_ in
                    (9_000_000, 8_000_000, 7_000_000, 6_000_000, 5_000_000)])
    out = _run(b, base_cap=3, hard_cap=3)  # 하드캡=정원=3 → 초과편입 없음
    kept = out[out["추천여부"]]["동일차주계좌합산원금잔액"].tolist()
    assert sorted(kept, reverse=True) == [9_000_000, 8_000_000, 7_000_000]
    held = out[out["최종등급"] == TH]["동일차주계좌합산원금잔액"].tolist()
    assert sorted(held, reverse=True) == [6_000_000, 5_000_000]


def test_cap_is_per_manager():
    b1 = _borrowers([(80, 10_000_000)] * 5, manager="김종일")
    b2 = _borrowers([(80, 10_000_000)] * 2, manager="민홍기")
    b = pd.concat([b1, b2], ignore_index=True)
    b["고객번호"] = [f"C{i:03d}" for i in range(len(b))]
    out = _run(b, base_cap=3, hard_cap=3)
    assert (out[out["부담당자"] == "김종일"]["추천여부"]).sum() == 3
    assert (out[out["부담당자"] == "민홍기"]["추천여부"]).sum() == 2


# --- §1-4 추천여부: 즉시/당월만, 보류 False ---
def test_recommend_only_immediate_month():
    b = _borrowers([(95, 20_000_000), (60, 20_000_000), (30, 20_000_000)])
    out = _run(b, base_cap=10, hard_cap=20)
    by = dict(zip(out["최종등급"], out["추천여부"]))
    assert by[TI] and by[TM]
    assert not by.get(TH, False)


def test_원등급_최종등급_분리():
    b = _borrowers([(80, 10_000_000)] * 6)
    b["고객번호"] = [f"C{i:03d}" for i in range(6)]
    out = _run(b, base_cap=3, hard_cap=5)
    assert (out["원등급"] != TH).all()           # 원등급은 전원 추천대상(즉시/당월)
    assert (out["최종등급"] == TH).sum() == 1     # 최종은 1명 보류(정원 초과)
    assert set(out.columns) >= {"원등급", "최종등급", "정원초과편입여부",
                                "상한적용여부", "상한전_담당자내순위",
                                "상한후_담당자내순위", "동일차주계좌합산원금잔액",
                                "상한하향사유", "추천순위_차주기준"}


# --- §7 최근입금 단독요인 → 즉시에서 당월 강등 ---
def test_recent_pay_only_demotes_immediate():
    # 즉시 후보군(상위 3) 중 dep 0.7 + base/sim 약함 → 당월로 강등, 나머지 즉시 유지
    specs = [(95, 20_000_000, 0.7, 50, 50),   # 강등 대상
             (94, 20_000_000, 0.0, 80, 80),
             (93, 20_000_000, 0.0, 80, 80)]
    specs += [(60 - i, 20_000_000) for i in range(7)]   # 당월군
    out = grading.assign_original_tier(_borrowers(specs))
    demoted = out[out["최근입금의존도"] == 0.7].iloc[0]
    assert demoted["원등급"] == TM and demoted["최근입금단독추천여부"]
    # 강등 안 된 상위 2명은 즉시 유지
    strong = out[(out["borrower_score"].isin([94, 93]))]
    assert (strong["원등급"] == TI).all()


# --- §3 랜덤박스 ---
def test_randombox_min_and_topn():
    # 합산 1,000만 미만은 풀 제외, 상위 20 중 5명 무작위
    specs = [(90 - i, (30 - i) * 1_000_000) for i in range(25)]  # 잔액 30M..6M
    specs += [(50, 3_000_000)]  # 300만 → 제외
    b = _borrowers(specs)
    pick = grading.randombox_pick(b, seed=1)
    assert len(pick) == config.RANDOMBOX_PICK
    assert (pick["동일차주계좌합산원금잔액"] >= config.RANDOMBOX_MIN_MULTI_PRINCIPAL).all()
    assert list(pick.columns) == ["고객번호", "부담당자", "동일차주계좌합산원금잔액"]


def test_randombox_varies_by_seed():
    specs = [(90 - i, (30 - i) * 1_000_000) for i in range(25)]
    b = _borrowers(specs)
    p1 = set(grading.randombox_pick(b, seed=1)["고객번호"])
    p2 = set(grading.randombox_pick(b, seed=2)["고객번호"])
    assert p1 != p2                    # 시드 다르면 다른 5명
    # 상위 20 풀 안에서만 뽑힘
    top20 = set(b.sort_values("borrower_score", ascending=False)
                .head(20)["고객번호"])
    assert p1 <= top20 and p2 <= top20


# --- §9 차주 통합 공식(0.7max+0.3mean) 유지 ---
def test_borrower_score_formula():
    df = pd.DataFrame([
        {"_차주키": "K1", "고객번호": "C1", "성명": "김", "대출번호": "L1",
         "현재원금": 3000000, "최초원금": 4000000, "회생": "", "채권상태(중)": "정상"},
        {"_차주키": "K1", "고객번호": "C1", "성명": "김", "대출번호": "L2",
         "현재원금": 5000000, "최초원금": 5000000, "회생": "", "채권상태(중)": "정상"},
    ])
    scores = pd.DataFrame({
        "final_score": [80.0, 40.0], "collateral_cap": ["S", "S"],
        "collateral_key": ["normal", "normal"], "base_score": [50, 50],
        "paid_similarity_score": [50, 50], "payment_history_score": [0, 0],
        "burden_score": [90, 90], "external_prior_adj": [0, 0],
        "feedback_adj": [0, 0], "sensitive_penalty": [0, 0],
        "stage2_adj": [0, 0], "최근입금의존도": [0.0, 0.0],
        "최근입금가점제한여부": [False, False],
    }, index=df.index)
    pay = pd.DataFrame({
        "입금_이력유무": [False, False], "입금_최근365일": [False, False],
        "입금_최근180일": [False, False], "입금_최근일": [None, None],
        "입금_최근액": [0, 0],
    }, index=df.index)
    m = grading.assign_managers(df)
    t = grading.assign_teams(m)
    b = grading.aggregate_borrowers(df, scores, pay, m, t)
    assert abs(b["borrower_score"].iloc[0] - 74.0) < 1e-6   # 0.7*80+0.3*60
    assert b["동일차주계좌합산원금잔액"].iloc[0] == 8000000
    assert b["동일차주계좌수"].iloc[0] == 2
