# -*- coding: utf-8 -*-
"""피드백 파싱/별칭/EB 캘리브레이션 테스트 (§9: 13)."""
import pandas as pd

from corona_reco import feedback, config


# 13. 성공판정: 금액>0=성공, 여부 토큰 인식
def test_success_by_amount():
    assert feedback.success_of("", 100000) == 1
    assert feedback.success_of(None, 50000) == 1


def test_success_by_flag():
    assert feedback.success_of("Y", None) == 1
    assert feedback.success_of("예", "") == 1
    assert feedback.success_of("입금", None) == 1
    assert feedback.success_of("성공", 0) == 1  # 여부=성공이면 금액0이어도 성공? amount<=0 → flag 우선
    # 실패 토큰 + 금액 없음/0
    assert feedback.success_of("N", 0) == 0
    assert feedback.success_of("미입금", None) == 0
    assert feedback.success_of("아니오", 0) == 0


def test_success_unknown():
    assert feedback.success_of(None, None) is None
    assert feedback.success_of("", "") is None


# 별칭: 납입금액/입금액/실제입금액 동일 처리
def test_alias_recognition_csv(tmp_path):
    p = tmp_path / "fb.csv"
    df = pd.DataFrame([
        {"추천월": "2026-07", "고객번호": "C1", "성명": "김", "대출번호": "L1",
         "납입금액": 200000, "입금여부": "Y", "현재원금": 1000000},
        {"추천월": "2026-07", "고객번호": "C2", "성명": "이", "대출번호": "L2",
         "납입금액": 0, "입금여부": "N", "현재원금": 500000},
    ])
    df.to_csv(p, index=False, encoding="utf-8-sig")
    rows, warns = feedback.parse_feedback_file(str(p))
    assert rows["성공여부"].iloc[0] == 1
    assert rows["성공여부"].iloc[1] == 0
    # 회수비율 = 실제입금액/원금잔액 (현재원금 별칭)
    assert abs(rows["회수비율"].iloc[0] - 0.2) < 1e-6


def test_alias_siljeimgeum(tmp_path):
    p = tmp_path / "fb2.csv"
    df = pd.DataFrame([
        {"추천월": "2026-07", "고객번호": "C1", "성명": "김", "대출번호": "L1",
         "실제입금액": 300000, "실제입금여부": "입금", "원금잔액": 600000},
    ])
    df.to_csv(p, index=False, encoding="utf-8-sig")
    rows, _ = feedback.parse_feedback_file(str(p))
    assert rows["성공여부"].iloc[0] == 1
    assert abs(rows["회수비율"].iloc[0] - 0.5) < 1e-6


def test_payment_date_before_month_invalidated(tmp_path):
    p = tmp_path / "fb3.csv"
    df = pd.DataFrame([
        {"추천월": "2026-07", "고객번호": "C1", "성명": "김", "대출번호": "L1",
         "실제입금액": 100000, "입금일자": "2026-05-01"},  # 추천월 이전
    ])
    df.to_csv(p, index=False, encoding="utf-8-sig")
    rows, warns = feedback.parse_feedback_file(str(p))
    assert rows["입금일자"].iloc[0] is None
    assert any("추천월 이전" in w for w in warns)


def test_mixing_weight():
    assert feedback.mixing_weight(10) == 0.0
    assert feedback.mixing_weight(50) == 0.15
    assert feedback.mixing_weight(200) == 0.30
    assert feedback.mixing_weight(5000) == 0.70


def test_eb_calibration():
    seg = pd.DataFrame([
        {"성공여부": 1, "회수비율": 0.5, "부담당자": "민홍기", "팀": "1팀",
         "회생": "", "채권상태중": "정상", "나이대": "40대", "개인사업자": 0,
         "담보부NPL": 0, "추천원금잔액": 1000000},
        {"성공여부": 0, "회수비율": 0.0, "부담당자": "민홍기", "팀": "1팀",
         "회생": "", "채권상태중": "정상", "나이대": "40대", "개인사업자": 0,
         "담보부NPL": 0, "추천원금잔액": 1000000},
    ])
    calib = feedback.build_segment_calibration(seg, m=20)
    assert calib["global"]["n"] == 2
    assert abs(calib["global"]["p"] - 0.5) < 1e-9
    # EB shrink: 세그먼트 성공률이 전역으로 수축
    dims = calib["dims"]
    assert "부담당자" in dims


def test_feedback_adj_bounded():
    seg = pd.DataFrame([
        {"성공여부": 1, "회수비율": 1.0, "부담당자": "민홍기", "팀": "1팀",
         "회생": "", "채권상태중": "정상", "나이대": "40대", "개인사업자": 0,
         "담보부NPL": 0, "추천원금잔액": 1000000},
    ] * 30)
    calib = feedback.build_segment_calibration(seg, m=20)
    frame = pd.DataFrame([{
        "회생": "", "원금잔액": 1000000, "팀": "1팀", "부담당자": "민홍기",
        "나이대": "40대", "담보부NPL": 0, "개인사업자": 0, "채권상태중": "정상",
    }])
    adj = feedback.compute_feedback_adj(frame, calib, mix_weight=0.5)
    assert -config.FEEDBACK_ADJ_CAP <= adj.iloc[0] <= config.FEEDBACK_ADJ_CAP


def test_feedback_adj_zero_when_no_mix():
    frame = pd.DataFrame([{
        "회생": "", "원금잔액": 1000000, "팀": "1팀", "부담당자": "민홍기",
        "나이대": "40대", "담보부NPL": 0, "개인사업자": 0, "채권상태중": "정상",
    }])
    adj = feedback.compute_feedback_adj(frame, {"global": {"p": 0.5, "r": 0.5, "n": 10}, "dims": {}},
                                        mix_weight=0.0)
    assert adj.iloc[0] == 0.0
