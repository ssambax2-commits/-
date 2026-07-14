# 사양서 (SPECIFICATION) — corona_reco v4

정식 참조 사양. 값·계약·알고리즘의 **단일 진실 원천**. 빌드 지시는 `BUILD_PROMPT.md` 참조.

- 앱: 코로나채권/NPL·PL 우선관리 추천 데스크톱(Windows exe, 내부망·오프라인)
- 버전: v4 (등급 3단계 · 누적학습 무결성 · 랜덤박스 · CustomTkinter)
- 스택: Python 3.11, CatBoost, pandas/numpy, openpyxl, joblib, xlrd, customtkinter, PyInstaller

---

## 1. 모듈 맵 (책임)
| 모듈 | 책임 |
|---|---|
| `config.py` | 모든 상수(가중치·정원·컷·roster·외부prior·blacklist) |
| `util.py` | 날짜/숫자 파싱, add_months, 주민번호→나이대·성별, UserFacingError |
| `io_loader.py` | csv/xlsx(.xls=xlrd) 로더, 컬럼 별칭, 필수컬럼 검증 |
| `payments.py` | 입금 파생값, **label_windows(t0,1·3M)** 시간창 라벨 |
| `exclusions.py` | 강제 제외 + 차주 전파 |
| `features.py` | ML 피처(기표 시점 고정값만) + **누수 blacklist assertion** |
| `scoring.py` | base/paid_similarity/payment/burden/prior/penalty/collateral + 최종식 |
| `model.py` | CatBoost 2단계 Hurdle(1M/3M), calibration, Lift@K 검증, champion-challenger |
| `feedback.py` | 피드백 별칭 파싱, EB 세그먼트 캘리브레이션, 혼합비중 |
| `grading.py` | 차주 통합, **3단계 등급·150/200 상한·랜덤박스** |
| `store.py` | SQLite 스키마·누적·dedup·모델메타·**store_state 무결성·백업** |
| `report_excel.py` / `feedback_excel.py` | 실무 보고서 / 피드백 파일 |
| `reasons.py` | 추천사유·하향사유 문장 |
| `pipeline.py` | 오케스트레이션(최초/월별) |
| `main.py` | CustomTkinter GUI |

---

## 2. 입력 데이터 계약
**활동데이터(월별, 추천 대상)** 필수 컬럼: `고객번호, 대출번호, 성명, 현재원금`. 주요 참조:
순번·주민등록번호·팀·담당자·부담당자·채권상태(대)·채권상태(중)·민원여부·채무부존재소송·회생·
원장상태·최초원금·매입당시OPB·대출이율·연체이율·양도횟수·법시행이후양도횟수·다중계좌 활동/총건수·
다중계좌 원금합계·차주구분·채권구분·상품명·상품명 세분류·담보세부종류·대출종류·매입일자·대출일자·
최초연체일·등록사유발생일·시효일자·최종이자수입일.

**학습데이터(최초 1회)** = 활동데이터 구조 + 오른쪽에 입금내역 가로형(`입금일자N/입금액N`,
헤더 없으면 CC~CV=0-index 80~99). 입금액 ≤ 0 무시. 엑셀 serial·문자열 날짜 모두 파싱.

**피드백 파일** = 지난달 생성분에 실적 기입. 별칭 흡수: 입금액↔실제입금액↔납입금액,
입금여부↔실제입금여부, 원금잔액↔현재원금, 활동여부↔활동↔컨택여부. 성공판정: 여부∈{Y,예,입금,
성공,1}이거나 금액>0 → 성공; {N,아니오,미입금,0}+금액0/없음 → 실패. 입금일자는 추천월 이후만 유효.

---

## 3. 시간축 라벨 & 누수 방지 (최우선)
- **t0 = 평가기준일 − 3개월**(`LABEL_TRAIN_OFFSET_MONTHS=3`), 관측창 `(1,3)`개월.
- 라벨: `y_1m/y_3m`(t0 이후 관측창 내 입금여부), `amt_1m/amt_3m`(입금액 합).
- 피처는 t0 이전 **기표/매입 시점 고정값만**. **ML 피처 blacklist**(현재원금·현재OPB·현재
  연체이자·현재미수금·현재부족금·현총액·최종이자수입일·최종갱신일/금액·차기이자수입일·해제일자·
  세분류·채권상태(대/중)·회생·원장상태·시효일자·약정일자·입금일자N/입금액N·다중계좌 원금합계·
  피드백 결과 전부). `features.assert_no_leakage()`가 위반 시 `LeakageError`로 빌드 중단.
- **허용 피처(화이트리스트):** log(최초원금·매입당시OPB·최초미수금·최초원리금), 경과월(매입·
  대출·최초연체·등록사유), 대출이율, 연체이율, 양도횟수, 법시행이후양도횟수, 다중계좌_총건수,
  개인사업자, 담보부NPL, 나이대(cat), 상품군(cat). 학습 라벨 제외: 해지 & 관측창 무입금 &
  과거입금 없음.

---

## 4. 모델 (CatBoost 2단계 Hurdle)
- **Stage1**(호라이즌별 1M·3M 분류): balanced bagging 앙상블 21개(positive 전량+negative
  3배 서브샘플), auto_class_weights, depth 4, l2 6.0, iters 300, early stop. positive <
  `POSITIVE_MIN_FOR_ML=30`이면 해당 호라이즌 규칙/prior fallback. 랭킹 호라이즌=1M(부족 시 3M).
- **Stage2**(입금 발생 시 예상입금액, log 회귀): 금액표본<30이면 세그먼트(잔액 3분위×담보부NPL)
  평균→전역 평균 fallback. 극단값 95퍼센타일 상한.
- **최종 예상회수액 = P(입금) × E(입금액|입금)**, 1M/3M 분리.
- **캘리브레이션**: 모집단 평균확률을 학습 base rate에 맞추는 전역 스케일링(순위 보존).
  진단에 calibration 전/후 합계 표시.
- **검증**: `time_group_split`(시간 우선, 불가 시 고객번호 그룹; 동일 차주 train/val 분리) →
  Lift@5%/10%, Precision@50/100, 기준선(랜덤·log잔액순). champion-challenger는 Lift@10%로 비교.

---

## 5. 점수식 (0~100)
```
pre  = w.base·base + w.sim·paid_similarity + w.pay·payment_history + w.burden·burden
final = clamp(pre × collateral_mult − sensitive_penalty + external_prior_adj
             + feedback_adj + stage2_adj, 0, 100)
```
- 기본 가중치 `DEFAULT_WEIGHTS = base 0.30 / paid_similarity 0.35 / payment_history 0.20 /
  burden 0.15`. ML 비활성 시 sim 가중치 base/pay/burden으로 자동 재분배.
- **burden 곡선**: ≤500만 55 / ≤1,000만 88 / ≤3,000만 **100** / ≤5,000만 85 / ≤1억 70 /
  >1억 55. 다중계좌 합산 >1억 −15, >5,000만 −8.
- **collateral(담보부NPL)**: none (0.55,B) / hasPay (0.70,A) / recent365 (0.80,S) /
  normal (1.00,S).
- **external_prior**: 나이대(20대이하−1·30대+1·40대+3·50대+1·60대−1·70대−2·80대이상−3, ±3) +
  개인사업자(무입금−5·입금有−2·최근365일0), 합계 clamp[−8,+5].
- **sensitive_penalty**: 민원(민원여부 또는 상태중=민원) −25(중복1회), 법조치 −20.
- **payment_history**: 입금없음0/최근180일100/365일80/730일60/그이전40/일자불명50, 건수·총액
  보정, **10만↓ 단건 ×0.6**(`RECENT_PAY_SMALL_MULT`).
- **차주 통합점수** = 0.70·max + 0.30·mean − 다중계좌부담(3건↑−3, 5건↑−6).

---

## 6. 등급 3단계 알고리즘 (핵심)
```
# 부담당자별, 차주 기준
적격 = 강제제외 통과 & 동일차주계좌합산원금잔액 ≥ 5,000,000
# 원등급(상한 전): 부담당자별 적격 차주 점수분포
boundary = 상위갭(점수 desc, 탐색 5~50%, 실패 시 상위30% 분위수)
원등급 = 즉시   if score ≥ boundary
        당월   if score ≥ MONTH_SCORE_FLOOR(40)
        보류   else
# 최종등급(150/200 상한): 원등급∈{즉시,당월}만 다중키 정렬
정렬키 = (점수, 차주합산잔액, 예상1M, 예상3M, 최근입금일, 고객번호, 대출번호)
θ = max(median(순위≤150 점수), OVERFLOW_ABS_FLOOR=45)
순위 ≤ 150            → 최종=원등급(즉시/당월)
150 < 순위 ≤ 200      → score ≥ θ ? 당월(정원초과편입) : 보류
순위 > 200            → 보류(하드캡)
추천여부 = 최종등급 ∈ {즉시, 당월}    # 보류=False, 최소보장 없음
```
상수: `ASSIGNEE_BASE_CAP=150, ASSIGNEE_HARD_CAP=200, OVERFLOW_ABS_FLOOR=45,
MONTH_SCORE_FLOOR=40, IMMEDIATE_GAP_SEARCH_LO=0.05, HI=0.50, QUANTILE_FALLBACK=0.30`.
한 차주의 모든 계좌에 동일 최종등급. 보류 하향분은 `상한하향사유` 기록(동점 경계는 "차주합산
원금잔액 기준 보류").

## 7. 랜덤박스
`grading.randombox_pick(borrowers, seed)`: 동일차주계좌합산원금잔액 ≥ `RANDOMBOX_MIN=10,000,000`
차주 → 점수순 상위 `TOP_POOL=20` → 무작위 `PICK=5`. 반환: 고객번호·부담당자·합산잔액. 매 클릭
새 시드(다른 5명). 앱 표시 전용.

## 8. 출력 시트
**추천 보고서**: 표지 / 1팀·2팀 추천리스트 / 팀별 요약 / 제외채권 / 법조치추천(참고) /
화해·정상·약속자 관리현황 / 진단·검증 / (옵션)차주상세.
팀 리스트 컬럼: 팀·부담당자·고객번호·대출번호·성명·채권구분·대분류·중분류·상품명·채권상태·
원금잔액·동일차주계좌합산원금잔액·동일차주계좌수·최근입금일·최근입금액·누적변제율·모델점수·
원등급·최종등급·추천순위_차주기준·추천순위_계좌기준·1개월예상회수액·3개월예상회수액·추천사유·
등급하향사유·정원초과편입여부·상한적용여부·상한하향사유·추천여부·피드백입금일·피드백입금액·
피드백결과. 스타일: 즉시 진한강조(F8CBAD)/당월 연한강조(FFF2CC)/정원초과 파랑(DDEBF7), 금액
#,##0(원), 틀고정·필터·가로 폭맞춤.
**피드백 파일**: 사용자 시트(전체 채점 차주; 입력칸=활동여부·실제입금여부·실제입금액·입금일자·
비고) + 숨김 시트(_모델메타: 추천월·고객번호·성명·대출번호·최종등급·모델버전·피처벡터).

## 9. SQLite 스키마
`recommendations`(추천월·고객번호·대출번호 PK, 세부점수·피처벡터·원등급·최종등급·추천여부·
정원초과편입여부·세그먼트변수) · `feedback`(추천월·고객번호·대출번호 PK, 실제입금·회수비율·
성공여부·활동여부) · `model_meta` · `segment_stats` · **`store_state`**(store_id·schema_version·
first/last_train_date·feedback_months·cum_samples·cum_positive·last_clean_exit·dirty·checksum) ·
`training_rows`(초기 학습데이터 원천 보존).

## 10. 무결성 상태 판정 (`store.check_integrity`)
- 🔴 위험: store_state 없음 또는 누적표본 0 & training_rows 0.
- 🟡 주의: 스키마버전 불일치 / dirty=1(비정상 종료) / 체크섬 불일치 / 모델 파일 누락.
- 🟢 정상: 위 해당 없음(최초학습일·반영피드백월수·누적표본·store_id 표시).
자동 백업 `data/backup/CoronaReco_YYYYMMDD_HHMM.db`(10개 회전) + `data/store_audit.log` +
dirty 플래그(시작1/정상종료0).

## 11. 상수 요약
| 이름 | 값 |
|---|---|
| DEFAULT_WEIGHTS | base .30 / sim .35 / pay .20 / burden .15 |
| ASSIGNEE_BASE_CAP / HARD_CAP | 150 / 200 |
| MONTH_SCORE_FLOOR / OVERFLOW_ABS_FLOOR | 40 / 45 |
| IMMEDIATE gap 탐색 / 분위수 | 0.05~0.50 / 상위 0.30 |
| BORROWER_MIN_MULTI_PRINCIPAL | 5,000,000 |
| RANDOMBOX min / pool / pick | 10,000,000 / 20 / 5 |
| LABEL offset / windows | 3개월 / (1,3) |
| POSITIVE_MIN_FOR_ML / bagging | 30 / 21 |
| RECENT_PAY small / mult / dep컷 | 100,000 / 0.6 / 0.60 |
| 차주통합 max/mean | 0.70 / 0.30 |
| 팀 roster | 2팀: 김종일·서보문·정재민 / 1팀: 민홍기·김승한·김영성·최병인·조은지 |

## 12. 테스트 목록 (pytest, 83건)
exclusions(폐지 안내가능·회생/사망/완제 제외·시효·부채증명원·차주 전파) · scoring(민원1회·
담보부NPL 배수·개인사업자·소액입금 제한·burden) · features(누수 게이팅·주민번호) · payments(시간창
라벨·serial·위치 가드) · grading(3단계·150/200·초과편입·동점 차주합산잔액순·500만 제외·랜덤박스·
원등급/최종등급 분리) · feedback(별칭·EB) · store_integrity(🟢🟡🔴·백업·체크섬·training_rows) ·
pipeline_e2e(최초→월별→랜덤박스). **랜덤분할 금지, 시간/그룹 분할.**
