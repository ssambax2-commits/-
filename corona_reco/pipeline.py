# -*- coding: utf-8 -*-
"""
pipeline.py — 실행 오케스트레이션 (월별 업무용)
================================================
최초/월별 두 모드를 조율한다. 순서:
로드(+필수컬럼/행 오류 점검) → 입금파생 → 강제제외(컴플라이언스) →
업무제외 분리(100만↓/부동산담보/정상·화해·약속자) → 피처(누수 게이팅) →
모델(1M/3M Hurdle+fallback) → 예상회수액(캘리브레이션) → 피드백 EB →
점수(최근입금 의존 진단) → 차주통합·원등급→S상한→최종등급 → 계좌 리스트 →
법조치추천 추출 → DB누적(선택편향 진단) → 보고서/피드백 엑셀.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

from . import (config, exclusions, feedback, feedback_excel, features,
               grading, io_loader, model as model_mod, payments, reasons,
               report_excel, scoring, store as store_mod, util)
from .util import UserFacingError

REQUIRED_ACTIVITY_COLUMNS = ["고객번호", "대출번호", "성명", "현재원금"]


@dataclass
class RunOptions:
    mode: str                          # "first" | "monthly"
    activity_path: str
    ref_date: _dt.date
    month: str                         # 추천월 "YYYY-MM"
    training_path: Optional[str] = None
    prev_feedback_path: Optional[str] = None
    data_dir: Optional[str] = None
    out_report_path: Optional[str] = None
    out_feedback_path: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    cuts: Optional[Dict[str, float]] = None
    cap_s: int = None
    cap_a: int = None
    min_reco: int = None
    exclude_expired_prescription: bool = None
    include_detail: bool = False
    stage2_enabled: bool = None        # (호환 유지 — 현재는 항상 시도+fallback)


@dataclass
class RunResult:
    borrowers: pd.DataFrame
    report_path: str
    feedback_path: str
    accounts: pd.DataFrame = None
    diagnostics: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    model_active: bool = False
    error_list_path: str = ""


def _log(progress: Optional[Callable], msg: str):
    if progress:
        progress(msg)


def _model_path(d): return os.path.join(d, config.MODEL_FILENAME)
def _snapshot_path(d): return os.path.join(d, "train_snapshot.joblib")
def _db_path(d): return os.path.join(d, config.SQLITE_FILENAME)


# ---------------------------------------------------------------------------
# 학습 데이터: 시간축 라벨(§4)
# ---------------------------------------------------------------------------
def build_training_data(train_df: pd.DataFrame, ref_date: _dt.date):
    """학습 스냅샷 → (X, labels{1m,3m}, groups, time_order, cat_features, diag).

    기준일 t0 = ref_date − LABEL_TRAIN_OFFSET_MONTHS(3개월 관측창 확보).
    피처는 t0 기준 경과값(기표/매입 시점 고정값)만 사용 — 입금·잔액류는
    blacklist 게이팅으로 원천 차단(§6-5). 라벨 = t0 이후 1M/3M 입금여부/입금액.
    """
    t0 = util.add_months(ref_date, -config.LABEL_TRAIN_OFFSET_MONTHS)
    lab = payments.label_windows(train_df, t0, config.LABEL_WINDOWS_MONTHS)

    age_bands, _g = features.compute_age_gender_bands(train_df, t0)
    fb = features.build_features(train_df, t0, age_bands=age_bands)  # 경과월 기준 t0
    X = fb.X

    # 라벨 오염 방지: 해지(비활동) & 관측창 내 무입금 & 과거입금도 없음 → 학습 제외
    ledger = io_loader.get_col(train_df, "원장상태").map(util.clean_str)
    any_pos = (lab["y_1m"] == 1) | (lab["y_3m"] == 1)
    keep = (any_pos | (ledger == config.LEDGER_ACTIVE) | lab["had_pay_before_t0"]).values

    gonum = io_loader.get_col(train_df, "고객번호").map(util.clean_str)
    name = io_loader.get_col(train_df, "성명").map(util.clean_str)
    groups = (gonum + "|" + name).values

    buy = io_loader.get_col(train_df, "매입일자")
    time_order = np.array([(util.parse_date(v).toordinal() if util.parse_date(v)
                            else np.nan) for v in buy], dtype=float)

    labels = {}
    for m in config.LABEL_WINDOWS_MONTHS:
        labels[f"{m}m"] = (lab[f"y_{m}m"].values[keep],
                           lab[f"amt_{m}m"].values[keep])

    present_blacklist = sorted(
        set(str(c).strip() for c in train_df.columns) & config.ML_FEATURE_BLACKLIST)
    n_keep = int(keep.sum())
    diag = {
        "시간축 기준일(t0)": t0.isoformat(),
        "피처 기준기간": f"~ {t0.isoformat()} (t0 이전, 기표/매입 시점 고정값만)",
        "라벨 관측기간": (f"1M: {t0.isoformat()}~{util.add_months(t0,1).isoformat()} / "
                     f"3M: {t0.isoformat()}~{util.add_months(t0,3).isoformat()}"),
        "학습 표본 수": n_keep,
        "positive 1M 수": int(labels["1m"][0].sum()),
        "positive 3M 수": int(labels["3m"][0].sum()),
        "1M positive rate": round(float(labels["1m"][0].mean()), 4) if n_keep else 0,
        "3M positive rate": round(float(labels["3m"][0].mean()), 4) if n_keep else 0,
        "누수 의심(blacklist) 컬럼 존재 수": len(present_blacklist),
        "제거된 누수 의심 컬럼": ", ".join(present_blacklist[:15]) + ("…" if len(present_blacklist) > 15 else ""),
    }
    return (X[keep], labels, groups[keep], time_order[keep], fb.cat_features, diag)


def _reconstruct_pool_X(pool_df, feature_names, cat_features):
    rows, ys, amts = [], [], []
    for _, r in pool_df.iterrows():
        try:
            fv = json.loads(r["피처벡터"]) if r["피처벡터"] else None
        except (ValueError, TypeError):
            fv = None
        if not fv:
            continue
        rows.append(fv)
        ys.append(int(r["성공여부"]))
        amt = r.get("실제입금액")
        amts.append(float(amt) if pd.notna(amt) else 0.0)
    if not rows:
        return None, None, None
    X = pd.DataFrame(rows)
    for c in feature_names:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_names]
    for c in cat_features:
        X[c] = X[c].fillna("unknown").astype(str)
    return X, np.array(ys, dtype=int), np.array(amts, dtype=float)


# ---------------------------------------------------------------------------
# 행 단위 입력 점검(§14) — 오류리스트 엑셀
# ---------------------------------------------------------------------------
def _scan_row_issues(act: pd.DataFrame) -> pd.DataFrame:
    issues = []
    gonum = io_loader.get_col(act, "고객번호")
    loan = io_loader.get_col(act, "대출번호")
    prin = io_loader.get_col(act, "현재원금")
    for i in range(len(act)):
        rowno = i + 2  # 엑셀 기준(헤더 1행)
        if not util.clean_str(gonum.iloc[i]):
            issues.append({"행번호": rowno, "컬럼": "고객번호", "문제": "값 없음"})
        if not util.clean_str(loan.iloc[i]):
            issues.append({"행번호": rowno, "컬럼": "대출번호", "문제": "값 없음"})
        pv = prin.iloc[i]
        if util.clean_str(pv) and util.to_number(pv) is None:
            issues.append({"행번호": rowno, "컬럼": "현재원금",
                           "문제": f"숫자 변환 실패: {str(pv)[:30]}"})
    return pd.DataFrame(issues)


# ---------------------------------------------------------------------------
# 업무 제외 분리(§5 제외채권/관리현황) — 컴플라이언스 통과분 대상
# ---------------------------------------------------------------------------
def _business_split(act_keep: pd.DataFrame, X_keep: pd.DataFrame):
    """반환: (scored_mask, biz_reason Series, managed_mask)."""
    n = len(act_keep)
    st_mid = io_loader.get_col(act_keep, "채권상태(중)").map(util.clean_str)
    st_big = io_loader.get_col(act_keep, "채권상태(대)").map(util.clean_str)
    coll_detail = io_loader.get_col(act_keep, "담보세부종류").map(util.clean_str)
    prod_group = X_keep["상품군"].astype(str)

    # §2 소액 하한은 차주합산 500만 기준으로 grading 단계에서 처리(여기선 계좌 부동산/관리만)
    managed = st_mid.isin(config.MANAGED_STATUS_MID) | st_big.isin(config.MANAGED_STATUS_BIG)
    realestate = coll_detail.str.contains("부동산", na=False) | (prod_group == "담보")

    reason = pd.Series([""] * n, index=act_keep.index)
    reason[realestate] = "부동산 담보성 상품 제외"
    reason[managed] = "정상/화해/약속자 관리대상으로 추천 제외"
    scored_mask = ~(managed | realestate)
    return scored_mask, reason, managed


# ---------------------------------------------------------------------------
# 법조치추천 추출(§5-6, 참고용)
# ---------------------------------------------------------------------------
def _legal_candidates(act: pd.DataFrame, pay: pd.DataFrame,
                      ref_date: _dt.date) -> pd.DataFrame:
    rows = []
    st_mid = io_loader.get_col(act, "채권상태(중)").map(util.clean_str)
    rehab = io_loader.get_col(act, "회생").map(util.clean_str)
    prin = io_loader.get_col(act, "현재원금")
    expiry = io_loader.get_col(act, "시효일자")
    gonum = io_loader.get_col(act, "고객번호")
    loan = io_loader.get_col(act, "대출번호")
    name = io_loader.get_col(act, "성명")
    bu = io_loader.get_col(act, "부담당자")
    da = io_loader.get_col(act, "담당자")
    for i in range(len(act)):
        p = util.to_number_or(prin.iloc[i], 0.0)
        if p < config.LEGAL_MIN_PRINCIPAL:
            continue
        if st_mid.iloc[i] in config.MANAGED_STATUS_MID:
            continue
        rh = rehab.iloc[i]
        if rh in config.REHAB_EXCLUDE:
            continue  # 조정사건 진행 — 법조치 부적절
        if bool(pay["입금_최근365일"].iloc[i]):
            continue  # 최근 1년 입금 있으면 회수추천 영역
        reasons_l = ["장기 무입금 · 잔액 상위"]
        exp_d = util.parse_date(expiry.iloc[i])
        if exp_d is not None and exp_d <= util.add_months(ref_date, config.LEGAL_EXPIRY_MONTHS):
            reasons_l.insert(0, f"시효 임박({exp_d.isoformat()})")
        # 조정사건 키워드(회생/신복/파산)는 사유 문구에서 배제
        mgr = grading.manager_of_row(bu.iloc[i], da.iloc[i])
        rows.append({
            "팀": grading.team_of_manager(mgr), "부담당자": mgr,
            "고객번호": util.clean_str(gonum.iloc[i]),
            "대출번호": util.clean_str(loan.iloc[i]),
            "성명": util.clean_str(name.iloc[i]),
            "원금잔액": p,
            "시효일자": exp_d.isoformat() if exp_d else "",
            "법조치검토사유": " · ".join(reasons_l),
        })
    df = pd.DataFrame(rows)
    if len(df):
        df = df.sort_values("원금잔액", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------------------------
def run_pipeline(opt: RunOptions, progress: Optional[Callable] = None) -> RunResult:
    try:
        return _run_pipeline_inner(opt, progress)
    except UserFacingError:
        raise
    except PermissionError as e:
        raise UserFacingError(
            "저장 경로에 쓸 수 없습니다. 결과 파일이 열려 있으면 닫고, "
            "쓰기 가능한 폴더(문서/바탕화면)에서 다시 실행하십시오.",
            detail=str(e)) from e


def _run_pipeline_inner(opt: RunOptions, progress) -> RunResult:
    warnings: List[str] = []
    diagnostics: Dict = {}

    data_dir = opt.data_dir or store_mod.default_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    store = store_mod.Store(_db_path(data_dir))

    try:
        # 1) 로드 + 필수컬럼/행 점검
        _log(progress, "활동데이터 로딩...")
        act = io_loader.load_table(opt.activity_path)
        io_loader.require_columns(act, REQUIRED_ACTIVITY_COLUMNS, "활동데이터")
        diagnostics["활동데이터 행수"] = len(act)
        diagnostics["입력파일(활동)"] = os.path.basename(opt.activity_path)

        issues = _scan_row_issues(act)
        error_list_path = ""
        if len(issues):
            diagnostics["행 단위 경고 건수"] = int(len(issues))
            warnings.append(f"입력 행 경고 {len(issues)}건 — 오류리스트 파일 확인.")
            error_list_path = os.path.join(data_dir, f"오류리스트_{opt.month}.xlsx")
            try:
                issues.to_excel(error_list_path, index=False)
            except Exception:  # noqa: BLE001
                error_list_path = ""

        # 2) 입금 파생
        _log(progress, "입금내역 파싱...")
        pay = payments.derive_payments(act, opt.ref_date)

        # 3) 강제 제외(컴플라이언스) + 차주 전파
        _log(progress, "강제 제외 규칙 적용...")
        act_ex = exclusions.apply_exclusions(act, opt.ref_date,
                                             opt.exclude_expired_prescription)
        excluded_df = act_ex[act_ex["_제외여부"]].copy()
        diagnostics["제외 계좌 수(컴플라이언스)"] = int(len(excluded_df))
        special_diag = exclusions.special_bond_diagnostic(act_ex)
        diagnostics["특수채권 상태값 확인 필요(계좌)"] = int((special_diag != "").sum())

        # 4) 나이/성별(원본 즉시 폐기) + 피처(누수 게이팅)
        _log(progress, "피처 빌드 + 누수 게이팅...")
        age_bands, gender = features.compute_age_gender_bands(act, opt.ref_date)
        fb_bundle = features.build_features(act, opt.ref_date, age_bands=age_bands)
        X_all = fb_bundle.X

        # 5) 모델 준비(최초=학습 / 월별=로드+증분)
        the_model, diag_m = _prepare_model(opt, store, data_dir, progress)
        diagnostics.update(diag_m)

        # 6) 컴플라이언스 통과 모집단
        keep = ~act_ex["_제외여부"].values
        idx_keep = act.index[keep]
        act_keep = act.loc[idx_keep].copy()
        act_keep["_차주키"] = act_ex.loc[idx_keep, "_차주키"].values
        pay_keep = pay.loc[idx_keep]
        X_keep = X_all.loc[idx_keep]
        age_keep = age_bands.loc[idx_keep]
        biz_keep = X_all.loc[idx_keep, "개인사업자"]

        # 7) 업무 제외 분리(§5): 100만↓ / 부동산담보 / 정상·화해·약속자
        scored_mask, biz_reason, managed_mask = _business_split(act_keep, X_keep)
        managed_df = act_keep[managed_mask].copy()
        managed_pay = pay_keep[managed_mask]
        biz_excluded_df = act_keep[(~scored_mask) & (~managed_mask)].copy()
        biz_excluded_df["_업무제외사유"] = biz_reason[(~scored_mask) & (~managed_mask)]
        diagnostics["업무제외 계좌 수(100만↓/부동산담보)"] = int(len(biz_excluded_df))
        diagnostics["관리현황 계좌 수(정상·화해·약속자)"] = int(len(managed_df))

        idx_sc = act_keep.index[scored_mask]
        act_sc = act_keep.loc[idx_sc]
        pay_sc = pay_keep.loc[idx_sc]
        X_sc = X_keep.loc[idx_sc]
        age_sc = age_keep.loc[idx_sc]
        biz_sc = biz_keep.loc[idx_sc]
        managers = grading.assign_managers(act_sc)
        teams = grading.assign_teams(managers)
        diagnostics["팀 불일치 건수(roster 우선)"] = grading.team_mismatch_count(act_sc, managers)
        diagnostics["채점 대상 계좌 수"] = int(len(act_sc))

        # 8) Stage1 확률 + 예상회수액(1M/3M, 캘리브레이션)
        paid_prob = None
        exp_recovery = pd.DataFrame(index=idx_sc,
                                    data={"e1m": 0.0, "e3m": 0.0})
        stage_diag = {"Stage1 사용": False, "Stage2 사용": False,
                      "Stage2 fallback": True, "fallback 사유": "모델 없음",
                      "calibration 적용": False}
        if the_model is not None:
            prob_arr = the_model.predict_proba(X_sc)
            if prob_arr is not None:
                paid_prob = pd.Series(prob_arr, index=idx_sc)
            stage_diag["Stage1 사용"] = bool(the_model.active)
            stage_diag["Stage2 사용"] = bool(the_model.stage2_enabled)
            fb1 = the_model.stage2_fallback.get("1m", {})
            fb3 = the_model.stage2_fallback.get("3m", {})
            stage_diag["Stage2 fallback"] = bool(fb1.get("used", True) or fb3.get("used", True))
            stage_diag["fallback 사유"] = "; ".join(
                x.get("reason", "") for x in (fb1, fb3) if x.get("reason")) or "-"
            rec1 = the_model.expected_recovery(X_sc, "1m")
            rec3 = the_model.expected_recovery(X_sc, "3m")
            exp_recovery["e1m"] = rec1["exp_cal"]
            exp_recovery["e3m"] = rec3["exp_cal"]
            stage_diag["calibration 적용"] = True
            stage_diag["calibration 전 예상회수액(1M합)"] = round(float(rec1["exp_raw"].sum()))
            stage_diag["calibration 후 예상회수액(1M합)"] = round(float(rec1["exp_cal"].sum()))
            stage_diag["calibration 전 예상회수액(3M합)"] = round(float(rec3["exp_raw"].sum()))
            stage_diag["calibration 후 예상회수액(3M합)"] = round(float(rec3["exp_cal"].sum()))
            if the_model.warning:
                warnings.append(the_model.warning)
        diagnostics.update(stage_diag)
        ml_active = paid_prob is not None
        diagnostics["ML 활성"] = ml_active

        # 9) Stage2 소폭 점수 반영(순위 보조, 상한 3점)
        stage2_adj = None
        if float(exp_recovery["e1m"].max()) > 0:
            stage2_adj = (exp_recovery["e1m"] / exp_recovery["e1m"].max()
                          ) * config.STAGE2_BONUS_CAP

        # 10) 피드백 EB 보정
        calib, mix, fb_warn, fb_diag = _obtain_calibration(opt, store, progress)
        warnings.extend(fb_warn)
        diagnostics.update(fb_diag)
        feedback_adj = None
        if calib is not None and mix > 0 and len(act_sc) > 0:
            seg_frame = feedback.build_segment_frame(
                act_sc, managers, teams, age_sc, biz_sc, X_sc["담보부NPL"])
            feedback_adj = feedback.compute_feedback_adj(seg_frame, calib, mix)

        # 11) 점수(최근입금 의존 진단 포함)
        _log(progress, "점수 계산...")
        scores = scoring.compute_scores(
            act_sc, pay_sc, opt.ref_date, age_bands=age_sc, biz_series=biz_sc,
            paid_prob=paid_prob, feedback_adj=feedback_adj, stage2_adj=stage2_adj,
            weights=opt.weights, ml_active=ml_active)

        # 12) 차주 통합 → 원등급 → S/A 상한 → 최종등급 → 순위
        _log(progress, "차주 통합 · 원등급/최종등급 · S상한(동점처리)...")
        borrowers = grading.full_grading_pipeline(
            act_sc, scores, pay_sc, managers, teams,
            cuts=opt.cuts, cap_s=opt.cap_s, cap_a=opt.cap_a, min_reco=opt.min_reco,
            age_bands=age_sc, biz_series=biz_sc, exp_recovery=exp_recovery)
        borrowers = reasons.add_reasons(borrowers)

        if len(borrowers):
            diagnostics["추천 차주 수(즉시+당월)"] = int(borrowers["추천여부"].sum())
            for t in (config.TIER_IMMEDIATE, config.TIER_MONTH, config.TIER_HOLD):
                diagnostics[f"{t} 차주 수(최종)"] = int((borrowers["최종등급"] == t).sum())
            diagnostics["정원초과 편입 차주 수(151~200)"] = int(borrowers["정원초과편입여부"].sum())
            diagnostics["소액(차주합산500만↓) 제외 차주 수"] = int(borrowers["소액제외여부"].sum())
            diagnostics["최근입금 단독요인 제한 수"] = int(borrowers["최근입금단독추천여부"].sum())
            # 부담당자별 「정상(≤150)/초과편입/보류」 한 줄 요약(§1-3)
            for mgr, grp in borrowers.groupby("부담당자", sort=True):
                if mgr == config.UNASSIGNED_LABEL:
                    continue
                normal = int(((grp["추천여부"]) & (~grp["정원초과편입여부"])).sum())
                over = int(grp["정원초과편입여부"].sum())
                hold = int((grp["최종등급"] == config.TIER_HOLD).sum())
                diagnostics[f"[{mgr}] 정상/초과편입/보류"] = f"{normal} / {over} / {hold}"

        # 13) 계좌 단위 리스트(보고서용) + 전월 피드백 결과 조인
        _log(progress, "계좌 리스트 구성...")
        accounts = _build_account_table(act_sc, scores, pay_sc, managers, teams,
                                        exp_recovery, borrowers, store)

        # 14) 법조치추천(참고) — 컴플라이언스 통과 & 관리대상 제외 모집단
        legal_df = _legal_candidates(act_keep[~managed_mask], pay_keep[~managed_mask],
                                     opt.ref_date)
        diagnostics["법조치 검토 후보 수"] = int(len(legal_df))

        # 15) 피처 JSON + DB 저장
        feature_json_by_key = _build_feature_json(borrowers, X_all)
        model_version = the_model.version if the_model else config.MODEL_SCHEMA_VERSION
        _log(progress, "SQLite 누적 저장...")
        store.save_recommendations(opt.month, borrowers, feature_json_by_key, model_version)

        # 16) 성과/선택편향 진단(누적 피드백 기반, §12·13)
        diagnostics.update(_performance_diag(store))
        diagnostics.update(_fairness_diag(borrowers, act_sc, gender))

        # 17) 보고서/피드백 엑셀
        _log(progress, "엑셀 보고서 생성...")
        report_path = opt.out_report_path or os.path.join(
            data_dir, f"추천보고서_{opt.month}.xlsx")
        feedback_path = opt.out_feedback_path or os.path.join(
            data_dir, f"피드백_{opt.month}.xlsx")
        managed_view = _managed_table(managed_df, managed_pay)
        small_view = borrowers[borrowers["소액제외여부"]].copy() if len(borrowers) else borrowers
        report_excel.write_report(
            report_path,
            month=opt.month,
            run_date=_dt.date.today().isoformat(),
            input_files={"활동데이터": os.path.basename(opt.activity_path),
                         "학습데이터": os.path.basename(opt.training_path) if opt.training_path else "-",
                         "전월 피드백": os.path.basename(opt.prev_feedback_path) if opt.prev_feedback_path else "-"},
            accounts=accounts, borrowers=borrowers,
            excluded_compliance=excluded_df, excluded_business=biz_excluded_df,
            excluded_small=small_view,
            managed=managed_view, legal=legal_df,
            diagnostics=diagnostics, include_detail=opt.include_detail)
        feedback_excel.write_feedback_file(
            feedback_path, borrowers, feature_json_by_key, model_version, opt.month)

        _log(progress, "완료.")
        return RunResult(borrowers=borrowers, accounts=accounts,
                         report_path=report_path, feedback_path=feedback_path,
                         diagnostics=diagnostics, warnings=warnings,
                         model_active=ml_active, error_list_path=error_list_path)
    finally:
        store.close()


# ---------------------------------------------------------------------------
def _prepare_model(opt, store, data_dir, progress):
    diag = {}
    mpath = _model_path(data_dir)
    spath = _snapshot_path(data_dir)

    if opt.mode == "first":
        if not opt.training_path:
            diag["모델"] = "학습데이터 없음 — 규칙/prior 폴백"
            return None, diag
        _log(progress, "학습데이터 로딩 및 모델 학습(1M/3M 라벨)...")
        train_df = io_loader.load_table(opt.training_path)
        X, labels, groups, time_order, cat_features, lab_diag = \
            build_training_data(train_df, opt.ref_date)
        diag.update(lab_diag)
        m = model_mod.train_and_validate(X, labels, cat_features,
                                         groups=groups, time_order=time_order)
        m.save(mpath)
        _save_snapshot(spath, X, labels, groups, time_order, cat_features)
        store.save_model_meta(m.version, m.n_positive, m.n_negative, m.metrics,
                              mpath, adopted=m.metrics.get("adopted", True))
        diag["모델버전"] = m.version
        diag["랭킹 호라이즌"] = m.ranking_horizon
        for k in ("lift@5%", "lift@10%", "precision@50", "precision@100",
                  "기준선_랜덤_lift@10%", "기준선_잔액순_lift@10%"):
            if k in m.metrics:
                diag[f"검증 {k}"] = m.metrics[k]
        return m, diag

    # monthly
    if not os.path.exists(mpath):
        diag["모델"] = "저장된 모델 없음 — 규칙/prior 폴백(최초 실행을 먼저 하세요)"
        return None, diag
    _log(progress, "저장된 모델 로드...")
    champion = model_mod.HurdleModel.load(mpath)
    diag["모델버전"] = champion.version

    challenger = _try_incremental(store, champion, spath, progress)
    if challenger is not None:
        champ_meta = store.get_latest_model_meta()
        champ_lift = None
        if champ_meta and champ_meta.get("지표"):
            try:
                champ_lift = json.loads(champ_meta["지표"]).get("lift@k")
            except (ValueError, TypeError):
                champ_lift = None
        if challenger.active and challenger.metrics.get("lift@k", 0) >= (champ_lift or 0):
            challenger.save(mpath)
            store.save_model_meta(challenger.version, challenger.n_positive,
                                  challenger.n_negative, challenger.metrics, mpath, True)
            diag["모델"] = f"challenger 채택(Lift@10% {challenger.metrics.get('lift@k')})"
            return challenger, diag
        diag["모델"] = "challenger 미채택 — 기존 챔피언 유지"
    return champion, diag


def _try_incremental(store, champion, spath, progress):
    """스냅샷 + 피드백 풀(익월 입금 라벨=1M)로 challenger 학습(§13).

    피드백 라벨은 1M 호라이즌에만 주입 — 3M은 원 스냅샷 유지(관측기간 불일치 방지).
    """
    pool = store.get_training_pool()
    if pool is None or len(pool) < 10 or not os.path.exists(spath):
        return None
    try:
        import joblib
        snap = joblib.load(spath)
    except Exception:  # noqa: BLE001
        return None
    if "labels" not in snap:
        return None  # 구버전 스냅샷 — 증분 생략
    Xp, yp, amtp = _reconstruct_pool_X(pool, snap["feature_names"], snap["cat_features"])
    if Xp is None:
        return None
    _log(progress, "피드백 증분 학습(challenger)...")
    X = pd.concat([snap["X"], Xp], ignore_index=True)
    n0 = len(snap["X"])
    labels = {}
    y1, a1 = snap["labels"]["1m"]
    labels["1m"] = (np.concatenate([y1, yp]), np.concatenate([a1, amtp]))
    if "3m" in snap["labels"]:
        y3, a3 = snap["labels"]["3m"]
        labels["3m"] = (np.concatenate([y3, yp]),   # 3M ⊇ 1M 근사(보수적)
                        np.concatenate([a3, amtp]))
    groups = np.concatenate([snap["groups"],
                             np.array([f"fb_{i}" for i in range(len(yp))])])
    tmax = (np.nanmax(snap["time_order"])
            if np.isfinite(snap["time_order"]).any() else 0)
    time_order = np.concatenate([snap["time_order"], np.full(len(yp), tmax + 1)])
    return model_mod.train_and_validate(X, labels, snap["cat_features"],
                                        groups=groups, time_order=time_order)


def _save_snapshot(spath, X, labels, groups, time_order, cat_features):
    import joblib
    joblib.dump({"X": X, "labels": labels, "groups": groups,
                 "time_order": time_order, "cat_features": cat_features,
                 "feature_names": list(X.columns)}, spath)


def _obtain_calibration(opt, store, progress):
    warns: List[str] = []
    diag: Dict = {}
    if opt.mode == "monthly" and opt.prev_feedback_path:
        _log(progress, "전월 피드백 파싱 및 누적...")
        fb_rows, warns = feedback.parse_feedback_file(opt.prev_feedback_path, store=store)
        saved = store.save_feedback(fb_rows)
        diag["피드백 누적 건수"] = saved
    fseg = store.get_feedback_with_segments()
    n_obs = len(fseg)
    diag["EB 관측수"] = int(n_obs)
    diag["총 피드백 관측수"] = store.total_feedback_observations()
    if n_obs == 0:
        return None, 0.0, warns, diag
    calib = feedback.build_segment_calibration(fseg)
    mw = feedback.mixing_weight(n_obs)
    diag["피드백 혼합비중"] = mw
    try:
        store.save_segment_stats(opt.month, feedback.segment_stats_for_store(calib))
    except Exception:  # noqa: BLE001
        pass
    return calib, mw, warns, diag


# ---------------------------------------------------------------------------
# 계좌 단위 결과표(§5 팀 리스트 컬럼)
# ---------------------------------------------------------------------------
def _build_account_table(act_sc, scores, pay_sc, managers, teams,
                         exp_recovery, borrowers, store) -> pd.DataFrame:
    bmap = {r["_차주키"]: r for _, r in borrowers.iterrows()} if len(borrowers) else {}

    # 전월 피드백 최신값 조인용
    fb_lut = {}
    try:
        fbdf = store.get_feedback()
        if len(fbdf):
            fbdf = fbdf.sort_values("추천월")
            for _, r in fbdf.iterrows():
                fb_lut[(str(r["고객번호"]), str(r["대출번호"]))] = r
    except Exception:  # noqa: BLE001
        pass

    prin = io_loader.get_col(act_sc, "현재원금")
    first_p = io_loader.get_col(act_sc, "최초원금")
    rows = []
    for i in act_sc.index:
        bkey = act_sc.at[i, "_차주키"]
        b = bmap.get(bkey, {})
        cp = util.to_number_or(prin.loc[i], 0.0)
        fp = util.to_number(first_p.loc[i])
        repay = util.clamp(1 - cp / fp, 0, 1) if (fp and fp > 0) else None
        gonum = util.clean_str(act_sc.at[i, "고객번호"]) if "고객번호" in act_sc.columns else ""
        loan = util.clean_str(act_sc.at[i, "대출번호"]) if "대출번호" in act_sc.columns else ""
        fbr = fb_lut.get((gonum, loan))
        rows.append({
            "팀": teams.loc[i], "부담당자": managers.loc[i],
            "고객번호": gonum, "대출번호": loan,
            "성명": util.clean_str(act_sc.at[i, "성명"]) if "성명" in act_sc.columns else "",
            "채권구분": util.clean_str(act_sc.at[i, "채권구분"]) if "채권구분" in act_sc.columns else "",
            "대분류": util.clean_str(act_sc.at[i, "채권상태(대)"]) if "채권상태(대)" in act_sc.columns else "",
            "중분류": util.clean_str(act_sc.at[i, "상품명 세분류"]) if "상품명 세분류" in act_sc.columns else "",
            "상품명": util.clean_str(act_sc.at[i, "상품명"]) if "상품명" in act_sc.columns else "",
            "채권상태": util.clean_str(act_sc.at[i, "채권상태(중)"]) if "채권상태(중)" in act_sc.columns else "",
            "원금잔액": cp,
            "동일차주계좌합산원금잔액": float(b.get("동일차주계좌합산원금잔액", cp) or cp),
            "동일차주계좌수": int(b.get("동일차주계좌수", 1) or 1),
            "최근입금일": (pay_sc.at[i, "입금_최근일"].isoformat()
                      if pay_sc.at[i, "입금_최근일"] else ""),
            "최근입금액": float(util.to_number(pay_sc.at[i, "입금_최근액"]) or 0),
            "누적변제율": round(repay, 4) if repay is not None else None,
            "모델점수": round(float(scores.at[i, "final_score"]), 2),
            "차주점수": round(float(b.get("borrower_score", 0) or 0), 2),
            "원등급": str(b.get("원등급", "")),
            "최종등급": str(b.get("최종등급", "")),
            "추천여부": bool(b.get("추천여부", False)),
            "정원초과편입여부": bool(b.get("정원초과편입여부", False)),
            "소액제외여부": bool(b.get("소액제외여부", False)),
            "추천순위_차주기준": b.get("추천순위_차주기준"),
            "1개월예상회수액": round(float(exp_recovery.at[i, "e1m"])),
            "3개월예상회수액": round(float(exp_recovery.at[i, "e3m"])),
            "추천사유": str(b.get("추천사유", "")),
            "등급하향사유": str(b.get("등급하향사유", "")),
            "상한적용여부": bool(b.get("상한적용여부", False)),
            "상한하향사유": str(b.get("상한하향사유", "")),
            "최근입금의존도": float(scores.at[i, "최근입금의존도"]),
            "최근입금단독추천여부": bool(b.get("최근입금단독추천여부", False)),
            "최근입금가점제한여부": bool(scores.at[i, "최근입금가점제한여부"]),
            "피드백입금일": (str(fbr["입금일자"]) if fbr is not None and pd.notna(fbr.get("입금일자")) else ""),
            "피드백입금액": (float(fbr["실제입금액"]) if fbr is not None and pd.notna(fbr.get("실제입금액")) else None),
            "피드백결과": ("입금" if (fbr is not None and fbr.get("성공여부") == 1)
                      else ("미입금" if (fbr is not None and fbr.get("성공여부") == 0) else "")),
            "_차주키": bkey,
        })
    acc = pd.DataFrame(rows)
    if len(acc) == 0:
        return acc
    # 추천순위_계좌기준: 부담당자 내 계좌 모델점수 순
    acc["추천순위_계좌기준"] = None
    for mgr, grp in acc.groupby("부담당자", sort=False):
        order = grp.sort_values(["모델점수", "동일차주계좌합산원금잔액"],
                                ascending=[False, False]).index
        for rank, j in enumerate(order, start=1):
            acc.at[j, "추천순위_계좌기준"] = rank
    return acc


def _managed_table(managed_df, managed_pay) -> pd.DataFrame:
    """화해·정상·약속자 관리현황(§5-7)."""
    if len(managed_df) == 0:
        return pd.DataFrame()
    rows = []
    prin = io_loader.get_col(managed_df, "현재원금")
    delin = io_loader.get_col(managed_df, "연체일수")
    for i in managed_df.index:
        mgr = grading.manager_of_row(
            managed_df.at[i, "부담당자"] if "부담당자" in managed_df.columns else "",
            managed_df.at[i, "담당자"] if "담당자" in managed_df.columns else "")
        rows.append({
            "팀": grading.team_of_manager(mgr), "부담당자": mgr,
            "고객번호": util.clean_str(managed_df.at[i, "고객번호"]),
            "대출번호": util.clean_str(managed_df.at[i, "대출번호"]) if "대출번호" in managed_df.columns else "",
            "성명": util.clean_str(managed_df.at[i, "성명"]),
            "상태(대)": util.clean_str(managed_df.at[i, "채권상태(대)"]) if "채권상태(대)" in managed_df.columns else "",
            "상태(중)": util.clean_str(managed_df.at[i, "채권상태(중)"]) if "채권상태(중)" in managed_df.columns else "",
            "원금잔액": util.to_number_or(prin.loc[i], 0.0),
            "연체일수": util.to_number(delin.loc[i]),
            "최근입금일": (managed_pay.at[i, "입금_최근일"].isoformat()
                      if managed_pay.at[i, "입금_최근일"] else ""),
            "최근입금액": float(util.to_number(managed_pay.at[i, "입금_최근액"]) or 0),
        })
    return pd.DataFrame(rows)


def _performance_diag(store) -> Dict:
    """등급/담당자별 실제 입금률 + 선택편향(§12·13)."""
    diag: Dict = {}
    try:
        gp = store.grade_performance()
        for _, r in gp.iterrows():
            if int(r["관측수"]) >= config.GRADE_PERF_MIN_N:
                diag[f"{r['등급']}등급 실제 입금률"] = round(float(r["입금률"]), 4)
        sa = gp[gp["등급"].isin([config.TIER_IMMEDIATE, config.TIER_MONTH])]
        if len(sa) and sa["관측수"].sum() >= config.GRADE_PERF_MIN_N:
            diag["즉시+당월 실제 입금률"] = round(
                float(sa["입금수"].sum() / sa["관측수"].sum()), 4)
    except Exception:  # noqa: BLE001
        pass
    try:
        bias = store.selection_bias_stats()
        for k, v in bias.items():
            if v is not None and v != "":
                diag[k] = v
    except Exception:  # noqa: BLE001
        pass
    return diag


def _build_feature_json(borrowers, X_all):
    out = {}
    for _, r in borrowers.iterrows():
        rep_idx = r.get("대표index")
        loan_no = str(r.get("대표대출번호", ""))
        if rep_idx is not None and rep_idx in X_all.index:
            fv = X_all.loc[rep_idx].to_dict()
            fv = {k: (None if (isinstance(v, float) and pd.isna(v)) else _py(v))
                  for k, v in fv.items()}
            out[loan_no] = json.dumps(fv, ensure_ascii=False)
    return out


def _py(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    return v


def _fairness_diag(borrowers, act_keep, gender):
    diag = {}
    try:
        g = gender.reindex(act_keep.index)
        vc = g.value_counts()
        for k in ("남", "여", "unknown"):
            diag[f"성별 분포({k})"] = int(vc.get(k, 0))
    except Exception:  # noqa: BLE001
        pass
    return diag
