# -*- coding: utf-8 -*-
"""
synth_recovery.py — 회수 라벨이 있는 합성 코로나채권 데이터 생성기
=================================================================
목적: 신규 회수예측 방법론을 검증하기 위한, '알려진 잠재신호(ground truth)'를
심은 합성데이터. 실데이터가 없을 때 파이프라인을 미리 짜고 시뮬레이션한다.

실제 입력 스키마(io_loader 별칭)와 동일한 컬럼 + 지역(신규) + 상환일/회수율.

라벨 정의 (사용자 지시): '상환되어 대분류가 해지된 채권' = recovered=1.
  1,000건 중 정확히 40건(4%)을 회수로 심는다 → 소표본(N_positive=40) 시뮬레이션.

⚠️ 잠재신호는 이 파일에만 존재한다. 모델은 이를 모른 채 '피처만 보고' 복원해야
   한다. 심어둔 신호 강도(로그오즈 기여):
     - 강함(검출 기대):   매입당시 잔액(↑→회수 확률↓), 나이대(40대 최고 U자형)
     - 중간:              지역(수도권↑, 일부 지방↓)
     - 약함(소표본에선 노이즈에 묻힘): 대출 경과기간, 빈티지, 담보부NPL
   + 큰 가우시안 잡음(회수엔 '운'도 작용) → 완벽 예측 불가.

⚠️ 상환일/회수율/입금열은 '회수된 채권에만' 존재한다 → 예측 피처로 쓰면 누수.
   여기서는 라벨/타이밍 분석 용도로만 생성한다(모델 피처는 recovery_lab에서
   매입·기표 시점 고정값만 선택).
"""
from __future__ import annotations

import datetime as _dt
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 지역(신규 컬럼) — 17개 시도 + 표본 가중 + 잠재 회수효과(로그오즈)
# ---------------------------------------------------------------------------
REGIONS = ["서울", "경기", "인천", "부산", "대구", "광주", "대전", "울산", "세종",
           "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주"]
# 인구 비례 대략 가중(수도권 과대) — 합=1 로 정규화
_REGION_W = np.array([18, 26, 6, 7, 5, 3, 3, 2, 1, 3, 3, 4, 3, 3, 4, 5, 2], float)
_REGION_W = _REGION_W / _REGION_W.sum()
# 잠재 지역효과: 수도권/광역시 상환성향↑, 일부 지방↓ (중간 강도 신호)
REGION_EFFECT = {
    "서울": 0.70, "경기": 0.55, "인천": 0.35, "대전": 0.30, "세종": 0.30,
    "울산": 0.20, "부산": 0.10, "대구": 0.00, "광주": 0.00, "경남": -0.10,
    "제주": -0.20, "충북": -0.20, "충남": -0.20, "경북": -0.40, "강원": -0.40,
    "전북": -0.45, "전남": -0.50,
}

# 나이대 라벨(config.EXTERNAL_PRIORS age_band_adj 와 동일 키)
AGE_BANDS = ["20대이하", "30대", "40대", "50대", "60대", "70대", "80대이상"]

# 매입당시 OPB 후보(원) — 소액~고액
_OPB_CHOICES = np.array([800_000, 2_500_000, 5_000_000, 8_000_000,
                         15_000_000, 25_000_000, 45_000_000, 70_000_000], float)
_OPB_W = np.array([10, 16, 18, 16, 14, 12, 8, 6], float)
_OPB_W = _OPB_W / _OPB_W.sum()

_PRODUCTS = ["신용대출", "담보대출", "신용카드", "할부금융"]
_BORROWER_TYPES = ["개인", "개인", "개인", "개인사업자"]


def _age_effect(age: int) -> float:
    """나이대 잠재효과(로그오즈). 40대 최고, 양끝 낮은 U자형."""
    if age < 30:
        return -0.25
    if age < 40:
        return 0.45
    if age < 50:
        return 0.90        # 40대 강한 +
    if age < 60:
        return 0.45
    if age < 70:
        return -0.20
    return -0.60


def _age_band(age: int) -> str:
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


def _make_rrn(age: int, gender: str, rng: np.random.Generator,
              ref_year: int = 2026) -> str:
    birth_year = ref_year - age
    yy = birth_year % 100
    if birth_year >= 2000:
        gcode = 3 if gender == "남" else 4
    else:
        gcode = 1 if gender == "남" else 2
    mm = int(rng.integers(1, 13))
    dd = int(rng.integers(1, 29))
    tail = int(rng.integers(100000, 1000000))
    return f"{yy:02d}{mm:02d}{dd:02d}-{gcode}{tail:06d}"


def _repay_month_weights() -> np.ndarray:
    """상환 계절성(월별 가중): 1월(설 상여)·6~7월·12월(연말정산) 소폭 몰림."""
    w = np.array([1.6, 1.0, 1.0, 1.0, 1.1, 1.4, 1.3, 1.0, 1.0, 1.1, 1.0, 1.5], float)
    return w / w.sum()


def generate(n: int = 1000, n_recovered: int = 40,
             ref_date: Optional[_dt.date] = None,
             seed: int = 20260701) -> pd.DataFrame:
    """합성 코로나채권 학습데이터 생성.

    반환 DataFrame: 실제 입력 스키마 컬럼 + 지역 + (회수건에 한해) 상환일/회수율/
    입금열 + 라벨(recovered) + 진단용 latent 컬럼(_latent_logit, 모델 사용 금지).
    """
    if ref_date is None:
        ref_date = _dt.date(2026, 7, 1)
    rng = np.random.default_rng(seed)

    # --- 원천 변수 표본 ---
    ages = rng.integers(23, 82, n)
    genders = rng.choice(["남", "여"], n)
    regions = rng.choice(REGIONS, size=n, p=_REGION_W)
    opb = rng.choice(_OPB_CHOICES, size=n, p=_OPB_W)
    loan_year = rng.integers(2011, 2022, n)            # 대출 빈티지
    collateral = (rng.random(n) < 0.18).astype(int)    # 담보부NPL
    biz = np.array([1 if bt == "개인사업자" else 0
                    for bt in rng.choice(_BORROWER_TYPES, n)])
    loan_rate = np.round(rng.uniform(3, 20, n), 2)
    delinq_rate = np.round(rng.uniform(6, 24, n), 2)
    transfer = rng.integers(0, 4, n)
    transfer_law = rng.integers(0, 3, n)
    multi_total = rng.integers(1, 7, n)

    # 폐지배정일자(시간축) — 2023-01 ~ 2025-06 균등 → 시간분할 검증에 사용
    t_start = _dt.date(2023, 1, 1)
    t_end = _dt.date(2025, 6, 30)
    assign_offsets = rng.integers(0, (t_end - t_start).days + 1, n)
    assign_dates = [t_start + _dt.timedelta(days=int(o)) for o in assign_offsets]

    # --- 잠재 회수 로그오즈(ground truth) ---
    log_opb = np.log10(opb)
    balance_eff = -0.60 * (log_opb - log_opb.mean())            # 잔액↑ → 회수↓ (강)
    age_eff = np.array([_age_effect(int(a)) for a in ages])     # 40대↑ (강)
    region_eff = np.array([REGION_EFFECT[r] for r in regions])  # 수도권↑ (중)
    season_eff = -0.06 * (ref_date.year - loan_year)            # 오래될수록 소폭↓ (약)
    vintage_eff = np.where((loan_year >= 2019) & (loan_year <= 2021),
                           -0.12, 0.0)                          # 코로나 빈티지 소폭↓ (약)
    collat_eff = 0.30 * collateral                              # 담보 소폭↑ (약)
    noise = rng.normal(0, 1.05, n)                             # 운(큰 잡음)

    latent = (balance_eff + age_eff + region_eff + season_eff
              + vintage_eff + collat_eff + noise)

    # 정확히 n_recovered 건을 상위 latent로 회수 처리
    recovered = np.zeros(n, dtype=int)
    top_idx = np.argsort(-latent)[:n_recovered]
    recovered[top_idx] = 1

    # --- 회수건 부가정보(상환일/회수율/입금열) ---
    month_w = _repay_month_weights()
    repay_dates = [""] * n
    recovery_ratio = [np.nan] * n
    pay_cols = {f"입금일자{p}": [""] * n for p in range(1, 6)}
    pay_cols.update({f"입금액{p}": [""] * n for p in range(1, 6)})

    for i in np.where(recovered == 1)[0]:
        a_date = assign_dates[i]
        # 상환까지 소요: 배정 후 1~14개월, 계절성 반영한 월 선택
        base = a_date + _dt.timedelta(days=int(rng.integers(25, 60)))
        # 이후 몇 달 뒤의 '몰리는 월'로 상환일 배치
        add_months = int(rng.integers(0, 12))
        yy = base.year + (base.month - 1 + add_months) // 12
        mm = (base.month - 1 + add_months) % 12 + 1
        # 계절성: 그 달을 살짝 흔들어 몰림 재현
        mm = int(rng.choice(np.arange(1, 13), p=month_w)) if rng.random() < 0.5 else mm
        dd = int(rng.integers(1, 28))
        try:
            rdate = _dt.date(yy, mm, dd)
        except ValueError:
            rdate = _dt.date(yy, mm, 28)
        if rdate <= a_date:
            rdate = a_date + _dt.timedelta(days=40)
        repay_dates[i] = rdate.isoformat()

        # 회수율: 40% 완제(≈1.0), 나머지 부분회수. 소액일수록 완제 확률↑
        p_full = 0.55 if opb[i] <= 5_000_000 else 0.30
        if rng.random() < p_full:
            ratio = float(rng.uniform(0.95, 1.0))
        else:
            ratio = float(np.clip(rng.beta(2, 3), 0.05, 0.9))
        recovery_ratio[i] = round(ratio, 3)

        # 입금열: 회수금액을 1~3회로 분할, 상환일 이전에 배치
        total_amt = ratio * opb[i]
        k = int(rng.integers(1, 4))
        splits = rng.dirichlet(np.ones(k)) * total_amt
        for j in range(k):
            dd_off = int(rng.integers(5, 400))
            pdate = rdate - _dt.timedelta(days=dd_off * (k - j))
            if pdate <= a_date:
                pdate = a_date + _dt.timedelta(days=10 + j * 20)
            pay_cols[f"입금일자{j+1}"][i] = pdate.isoformat()
            pay_cols[f"입금액{j+1}"][i] = int(round(splits[j] / 1000) * 1000)

    # --- 날짜 파생: 대출일자/최초연체일/매입일자 ---
    loan_dates, delinq_dates, buy_dates = [], [], []
    for i in range(n):
        ly = int(loan_year[i])
        ld = _dt.date(ly, int(rng.integers(1, 13)), int(rng.integers(1, 28)))
        dq = ld + _dt.timedelta(days=int(rng.integers(180, 900)))
        bd = _dt.date(int(rng.integers(2018, 2024)), int(rng.integers(1, 13)),
                      int(rng.integers(1, 28)))
        loan_dates.append(ld.isoformat())
        delinq_dates.append(dq.isoformat())
        buy_dates.append(bd.isoformat())

    # 대분류(원장상태): 회수건 = 해지, 그 외 = 활동
    ledger = np.where(recovered == 1, "해지", "활동")

    n_customers = int(n * 0.85)
    rows = []
    for i in range(n):
        cust = f"C{int(rng.integers(1, n_customers + 1)):06d}"
        row = {
            "순번": i + 1,
            "고객번호": cust,
            "대출번호": f"L{i:07d}",
            "성명": f"고객{cust[-4:]}",
            "주민등록번호": _make_rrn(int(ages[i]), genders[i], rng),
            "지역": regions[i],                       # ★ 신규 컬럼
            "팀": rng.choice(["1팀", "2팀", "기타"]),
            "담당자": rng.choice(["민홍기", "김승한", "김종일", "서보문", "정재민"]),
            "부담당자": rng.choice(["민홍기", "김승한", "김영성", "김종일", "정재민"]),
            "채권상태(대)": ledger[i],                # 대분류(해지/활동)
            "채권상태(중)": rng.choice(["폐지_신복", "폐지_회생", "폐지_파산",
                                        "연체", "약속자", "재통화"]),
            "회생": rng.choice(["", "부채증명원", "기각/취하/폐지"]),
            "원장상태": ledger[i],
            "현재원금": int(opb[i]),                  # (누수 피처 — 모델 제외)
            "현재OPB": int(opb[i]),
            "최초원금": int(opb[i] + rng.integers(0, 600000)),
            "최초미수금": int(rng.integers(0, 400000)),
            "최초원리금": int(opb[i] + rng.integers(0, 900000)),
            "매입당시OPB": int(opb[i]),               # ★ 피처 잔액(고정값)
            "대출이율": loan_rate[i],
            "연체이율": delinq_rate[i],
            "양도횟수": int(transfer[i]),
            "법시행이후양도횟수": int(transfer_law[i]),
            "다중계좌 활동/총건수": f"{int(rng.integers(1, multi_total[i]+1))}/{int(multi_total[i])}",
            "다중계좌 원금합계": int(opb[i] * int(rng.integers(1, 5))),
            "차주구분": "개인사업자" if biz[i] else "개인",
            "채권구분": "담보부NPL" if collateral[i] else "일반무담보",
            "상품명": rng.choice(_PRODUCTS),
            "담보세부종류": "부동산" if collateral[i] else "",
            "대출종류": "담보" if collateral[i] else "신용",
            "대출일자": loan_dates[i],
            "최초연체일": delinq_dates[i],
            "등록사유발생일": (_dt.date.fromisoformat(delinq_dates[i])
                              + _dt.timedelta(days=90)).isoformat(),
            "매입일자": buy_dates[i],
            "폐지배정일자": assign_dates[i].isoformat(),   # 시간축(검증 분할용)
            "시효일자": _dt.date(int(rng.integers(2027, 2033)),
                                int(rng.integers(1, 13)), 1).isoformat(),
            # --- 라벨/타이밍(회수건만) ---
            "상환일": repay_dates[i],                 # (누수 — 라벨/타이밍 전용)
            "회수율": recovery_ratio[i],              # (누수 — 분석 전용)
            "recovered": int(recovered[i]),           # ★ 라벨
            # 진단용(모델 사용 금지)
            "_latent_logit": round(float(latent[i]), 4),
        }
        for p in range(1, 6):
            row[f"입금일자{p}"] = pay_cols[f"입금일자{p}"][i]
            row[f"입금액{p}"] = pay_cols[f"입금액{p}"][i]
        rows.append(row)

    return pd.DataFrame(rows)


def write_csv(df: pd.DataFrame, path: str, encoding: str = "utf-8-sig") -> str:
    df.to_csv(path, index=False, encoding=encoding)
    return path


if __name__ == "__main__":
    d = generate()
    print("shape:", d.shape)
    print("recovered:", int(d["recovered"].sum()), "/", len(d),
          f"({d['recovered'].mean()*100:.1f}%)")
    print("지역 분포 top:", d["지역"].value_counts().head().to_dict())
