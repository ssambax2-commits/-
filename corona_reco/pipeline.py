# -*- coding: utf-8 -*-
"""
pipeline.py — 실행 오케스트레이션
==================================
최초/월별 두 모드를 조율한다. GUI(main.py)와 테스트가 이 진입점을 공유한다.
데이터 로드 → 입금파생 → 제외 → 나이/성별(원본폐기) → 피처 → 모델 →
피드백 EB → 점수 → 등급/차주통합 → 사유 → DB저장 → 엑셀 2종.
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
    stage2_enabled: bool = None


@dataclass
class RunResult:
    borrowers: pd.DataFrame
    report_path: str
    feedback_path: str
    diagnostics: Dict = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    model_active: bool = False


def _log(progress: Optional[Callable], msg: str):
    if progress:
        progress(msg)


def _model_path(data_dir: str) -> str:
    return os.path.join(data_dir, config.MODEL_FILENAME)


def _snapshot_path(data_dir: str) -> str:
    return os.path.join(data_dir, "train_snapshot.joblib")


def _db_path(data_dir: str) -> str:
    return os.path.join(data_dir, config.SQLITE_FILENAME)


# ---------------------------------------------------------------------------
# 학습 라벨/스냅샷 구성 (최초 실행)
# ---------------------------------------------------------------------------
def build_training_arrays(train_df: pd.DataFrame, ref_date: _dt.date):
    """학습데이터 → (X, y, groups, time_order, recovery). §1-4 라벨."""
    pay = payments.derive_payments(train_df, ref_date)
    age_bands, _gender = features.compute_age_gender_bands(train_df, ref_date)
    fb = features.build_features(train_df, ref_date, age_bands=age_bands)
    X = fb.X

    ledger = io_loader.get_col(train_df, "원장상태").map(util.clean_str)
    has_pay = pay["입금_이력유무"].astype(bool)
    y = has_pay.astype(int).values
    # 학습 제외: 해지(비활동) & 무입금
    keep_mask = (has_pay | (ledger == config.LEDGER_ACTIVE)).values

    # groups: 차주키
    gonum = io_loader.get_col(train_df, "고객번호").map(util.clean_str)
    name = io_loader.get_col(train_df, "성명").map(util.clean_str)
    groups = (gonum + "|" + name).values

    # time_order: 매입일자(serial 근사)
    buy = io_loader.get_col(train_df, "매입일자")
    time_order = np.array([
        (util.parse_date(v).toordinal() if util.parse_date(v) else np.nan)
        for v in buy
    ], dtype=float)

    # recovery(stage2용): 총입금 / 매입당시OPB (고정분모, 누수방지)
    opb = io_loader.get_col(train_df, "매입당시OPB")
    recovery = []
    for i in range(len(train_df)):
        base = util.to_number(opb.iloc[i])
        tot = float(pay["입금_총액"].iloc[i])
        recovery.append(util.clamp(tot / base, 0, 3) if (base and base > 0) else np.nan)
    recovery = np.array(recovery, dtype=float)

    return (X[keep_mask], y[keep_mask], groups[keep_mask],
            time_order[keep_mask], recovery[keep_mask], fb.cat_features)


def _reconstruct_pool_X(pool_df: pd.DataFrame, feature_names, cat_features):
    """recommendations.피처벡터(JSON) → X DataFrame(y=성공여부)."""
    rows = []
    ys = []
    recs = []
    for _, r in pool_df.iterrows():
        try:
            fv = json.loads(r["피처벡터"]) if r["피처벡터"] else None
        except (ValueError, TypeError):
            fv = None
        if not fv:
            continue
        rows.append(fv)
        ys.append(int(r["성공여부"]))
        rv = r.get("회수비율")
        recs.append(float(rv) if pd.notna(rv) else np.nan)
    if not rows:
        return None, None, None
    X = pd.DataFrame(rows)
    for c in feature_names:
        if c not in X.columns:
            X[c] = np.nan
    X = X[feature_names]
    for c in cat_features:
        X[c] = X[c].fillna("unknown").astype(str)
    return X, np.array(ys, dtype=int), np.array(recs, dtype=float)


# ---------------------------------------------------------------------------
# 메인 실행
# ---------------------------------------------------------------------------
def run_pipeline(opt: RunOptions, progress: Optional[Callable] = None) -> RunResult:
    warnings: List[str] = []
    diagnostics: Dict = {}

    data_dir = opt.data_dir or store_mod.default_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    store = store_mod.Store(_db_path(data_dir))

    try:
        # 1) 로드
        _log(progress, "활동데이터 로딩...")
        act = io_loader.load_table(opt.activity_path)
        diagnostics["활동데이터 행수"] = len(act)

        # 2) 입금 파생 (평가 기준일 기준)
        _log(progress, "입금내역 파싱...")
        pay = payments.derive_payments(act, opt.ref_date)

        # 3) 제외 판정 + 차주단위 전파
        _log(progress, "강제 제외 규칙 적용...")
        act_ex = exclusions.apply_exclusions(
            act, opt.ref_date, opt.exclude_expired_prescription)
        excluded_df = act_ex[act_ex["_제외여부"]].copy()
        diagnostics["제외 계좌 수"] = int(len(excluded_df))
        # 채권상태(대)=특수채권 & 원장상태=활동 → 진단만 표시(제외 아님, §2)
        special_diag = exclusions.special_bond_diagnostic(act_ex)
        diagnostics["특수채권 상태값 확인 필요(계좌)"] = int((special_diag != "").sum())

        # 4) 나이대/성별 (주민번호 원본 폐기)
        _log(progress, "나이대/성별 산출(주민번호 원본 폐기)...")
        age_bands, gender = features.compute_age_gender_bands(act, opt.ref_date)

        # 5) 피처 (누수 게이팅 포함)
        _log(progress, "ML 피처 빌드 + 누수 게이팅...")
        fb_bundle = features.build_features(act, opt.ref_date, age_bands=age_bands)
        X_all = fb_bundle.X

        # 6) 모델 준비
        the_model, diag_m = _prepare_model(opt, store, data_dir, progress)
        diagnostics.update(diag_m)

        # 7) 비제외 대상만 채점 모집단 + 담당자/팀 배정
        keep = ~act_ex["_제외여부"].values
        idx_keep = act.index[keep]
        act_keep = act.loc[idx_keep].copy()
        act_keep["_차주키"] = act_ex.loc[idx_keep, "_차주키"].values
        pay_keep = pay.loc[idx_keep]
        X_keep = X_all.loc[idx_keep]
        age_keep = age_bands.loc[idx_keep]
        biz_keep = X_all.loc[idx_keep, "개인사업자"]
        managers = grading.assign_managers(act_keep)
        teams = grading.assign_teams(managers)
        diagnostics["팀 불일치 건수(roster 우선)"] = grading.team_mismatch_count(act_keep, managers)

        # 8) 모델 예측확률(비활성 시 None)
        paid_prob = None
        if the_model is not None and the_model.active:
            prob_arr = the_model.predict_proba(X_keep)
            if prob_arr is not None:
                paid_prob = pd.Series(prob_arr, index=idx_keep)
        ml_active = paid_prob is not None
        diagnostics["ML 활성"] = ml_active
        if the_model is not None and the_model.warning:
            warnings.append(the_model.warning)

        # 8-2) Stage2 예상회수(활성 시) → 소폭 additive 보너스
        stage2_adj = None
        recovery_pred = None
        if the_model is not None and getattr(the_model, "stage2_enabled", False):
            rec_arr = the_model.predict_recovery(X_keep)
            if rec_arr is not None:
                recovery_pred = pd.Series(rec_arr, index=idx_keep)
                stage2_adj = recovery_pred.clip(lower=0, upper=1) * config.STAGE2_BONUS_CAP
                diagnostics["Stage2 예상회수 평균"] = round(float(recovery_pred.mean()), 4)

        # 9) 피드백 EB → 계좌레벨 feedback_adj Series
        feedback_adj = None
        calib, mix, fb_warn, fb_diag = _obtain_calibration(opt, store, progress)
        warnings.extend(fb_warn)
        diagnostics.update(fb_diag)
        if calib is not None and mix > 0 and len(act_keep) > 0:
            seg_frame = feedback.build_segment_frame(
                act_keep, managers, teams, age_keep, biz_keep, X_keep["담보부NPL"])
            feedback_adj = feedback.compute_feedback_adj(seg_frame, calib, mix)

        # 10) 점수
        _log(progress, "점수 계산...")
        scores = scoring.compute_scores(
            act_keep, pay_keep, opt.ref_date,
            age_bands=age_keep, biz_series=biz_keep,
            paid_prob=paid_prob, feedback_adj=feedback_adj, stage2_adj=stage2_adj,
            weights=opt.weights, ml_active=ml_active)

        # 11) 등급/차주통합
        _log(progress, "차주 통합 · 등급 · 상한/최소보장...")
        borrowers = grading.full_grading_pipeline(
            act_keep, scores, pay_keep, managers, teams,
            cuts=opt.cuts, cap_s=opt.cap_s, cap_a=opt.cap_a, min_reco=opt.min_reco,
            age_bands=age_keep, biz_series=biz_keep)

        # 12) 추천사유 (+ Stage2 예상회수 대표값 부착)
        if recovery_pred is not None and len(borrowers) > 0:
            borrowers["예상회수"] = [
                (round(float(recovery_pred.get(r["대표index"])), 4)
                 if r.get("대표index") in recovery_pred.index else None)
                for _, r in borrowers.iterrows()
            ]
        borrowers = reasons.add_reasons(borrowers)

        # 13) 피처 JSON (대표계좌 기준)
        feature_json_by_key = _build_feature_json(borrowers, X_all)

        # 14) 공정성 진단(성별 — 점수 미반영)
        diagnostics.update(_fairness_diag(borrowers, act_keep, gender))

        model_version = the_model.version if the_model else config.MODEL_SCHEMA_VERSION

        # 15) DB 저장
        _log(progress, "SQLite 누적 저장...")
        store.save_recommendations(opt.month, borrowers, feature_json_by_key, model_version)

        # 16) 엑셀 2종
        _log(progress, "엑셀 보고서 생성...")
        report_path = opt.out_report_path or os.path.join(
            data_dir, f"추천보고서_{opt.month}.xlsx")
        feedback_path = opt.out_feedback_path or os.path.join(
            data_dir, f"피드백_{opt.month}.xlsx")

        detail_df = _detail_frame(borrowers) if opt.include_detail else None
        report_excel.write_report(
            report_path, borrowers, diagnostics=diagnostics, detail_df=detail_df,
            excluded_df=excluded_df, include_detail=opt.include_detail, month=opt.month)
        feedback_excel.write_feedback_file(
            feedback_path, borrowers, feature_json_by_key, model_version, opt.month)

        _log(progress, "완료.")
        return RunResult(
            borrowers=borrowers, report_path=report_path, feedback_path=feedback_path,
            diagnostics=diagnostics, warnings=warnings,
            model_active=ml_active)
    finally:
        store.close()


# ---------------------------------------------------------------------------
def _prepare_model(opt: RunOptions, store, data_dir, progress):
    """모델 학습(최초) 또는 로드+증분(월별)."""
    diag = {}
    mpath = _model_path(data_dir)
    spath = _snapshot_path(data_dir)

    if opt.mode == "first":
        if not opt.training_path:
            diag["모델"] = "학습데이터 없음 — 규칙/prior 폴백"
            return None, diag
        _log(progress, "학습데이터 로딩 및 모델 학습...")
        train_df = io_loader.load_table(opt.training_path)
        X, y, groups, time_order, recovery, cat_features = build_training_arrays(
            train_df, opt.ref_date)
        m = model_mod.train_and_validate(
            X, y, cat_features, groups=groups, time_order=time_order,
            recovery=recovery, stage2_enabled=opt.stage2_enabled,
            champion_lift=None)
        m.save(mpath)
        _save_snapshot(spath, X, y, groups, time_order, recovery, cat_features)
        store.save_model_meta(m.version, m.n_positive, m.n_negative, m.metrics,
                              mpath, adopted=m.metrics.get("adopted", True))
        diag["모델버전"] = m.version
        diag["학습 positive"] = m.n_positive
        diag["학습 negative"] = m.n_negative
        diag["Lift@K"] = m.metrics.get("lift@k")
        diag["Stage2(예상회수) 활성"] = m.stage2_enabled
        return m, diag

    # monthly
    if not os.path.exists(mpath):
        diag["모델"] = "저장된 모델 없음 — 규칙/prior 폴백(최초 실행을 먼저 하세요)"
        return None, diag
    _log(progress, "저장된 모델 로드...")
    champion = model_mod.HurdleModel.load(mpath)
    diag["모델버전"] = champion.version

    # 보조 ML 증분(피드백 풀 + 원본 스냅샷) — champion-challenger
    challenger = _try_incremental(store, champion, spath, progress)
    if challenger is not None:
        champ_meta = store.get_latest_model_meta()
        champ_lift = (json.loads(champ_meta["지표"]).get("lift@k")
                      if champ_meta and champ_meta.get("지표") else None)
        if challenger.metrics.get("lift@k", 0) >= (champ_lift or 0) and challenger.active:
            challenger.save(mpath)
            store.save_model_meta(challenger.version, challenger.n_positive,
                                  challenger.n_negative, challenger.metrics, mpath, True)
            diag["모델"] = f"challenger 채택(Lift@K {challenger.metrics.get('lift@k')})"
            return challenger, diag
        else:
            diag["모델"] = "challenger 미채택 — 기존 챔피언 유지"
    return champion, diag


def _try_incremental(store, champion, spath, progress):
    """원본 스냅샷 + 피드백 풀로 challenger 학습(과적합 방지)."""
    pool = store.get_training_pool()
    if pool is None or len(pool) < 10:
        return None
    if not os.path.exists(spath):
        return None
    try:
        import joblib
        snap = joblib.load(spath)
    except Exception:  # noqa: BLE001
        return None
    Xp, yp, recp = _reconstruct_pool_X(pool, snap["feature_names"], snap["cat_features"])
    if Xp is None:
        return None
    _log(progress, "피드백 증분 학습(challenger)...")
    X = pd.concat([snap["X"], Xp], ignore_index=True)
    y = np.concatenate([snap["y"], yp])
    groups = np.concatenate([snap["groups"], np.array([f"fb_{i}" for i in range(len(yp))])])
    time_order = np.concatenate([snap["time_order"], np.full(len(yp), np.nanmax(snap["time_order"]) + 1
                                                             if np.isfinite(snap["time_order"]).any() else 1)])
    challenger = model_mod.train_and_validate(
        X, y, snap["cat_features"], groups=groups, time_order=time_order)
    return challenger


def _save_snapshot(spath, X, y, groups, time_order, recovery, cat_features):
    import joblib
    joblib.dump({
        "X": X, "y": y, "groups": groups, "time_order": time_order,
        "recovery": recovery, "cat_features": cat_features,
        "feature_names": list(X.columns),
    }, spath)


def _obtain_calibration(opt, store, progress):
    """월별이면 전월 피드백을 DB 누적 후, DB 전체 피드백으로 EB 캘리브레이션.

    반환: (calibration|None, mix_weight, warnings, diag)
    """
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


def _build_feature_json(borrowers, X_all):
    out = {}
    for _, r in borrowers.iterrows():
        rep_idx = r.get("대표index")
        loan_no = str(r.get("대표대출번호", ""))
        if rep_idx is not None and rep_idx in X_all.index:
            fv = X_all.loc[rep_idx].to_dict()
            # numpy 타입 → 순수 파이썬
            fv = {k: (float(v) if isinstance(v, (np.floating, float)) and not pd.isna(v)
                      else (None if (isinstance(v, float) and pd.isna(v)) else
                            (str(v) if isinstance(v, str) else _py(v))))
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
    """성별 공정성 진단(점수 미반영, 표시용)."""
    diag = {}
    try:
        g = gender.reindex(act_keep.index)
        # 차주 대표 성별은 생략하고 계좌기준 분포만 표시
        vc = g.value_counts()
        for k in ("남", "여", "unknown"):
            diag[f"성별 분포({k})"] = int(vc.get(k, 0))
    except Exception:  # noqa: BLE001
        pass
    if len(borrowers) > 0:
        diag["추천 차주 수"] = int((borrowers.get("추천여부", True) == True).sum())
        diag["S등급 수"] = int((borrowers["등급"] == "S").sum())
        diag["A등급 수"] = int((borrowers["등급"] == "A").sum())
    return diag


def _detail_frame(borrowers):
    cols = ["고객번호", "성명", "부담당자", "팀", "등급", "등급_raw", "등급상한",
            "borrower_score", "base_score", "paid_similarity_score",
            "payment_history_score", "burden_score", "external_prior_adj",
            "sensitive_penalty", "feedback_adj", "stage2_adj", "예상회수",
            "collateral_key", "회생", "채권상태(중)", "나이대", "개인사업자",
            "원금잔액", "다중계좌원금잔액", "활동계좌수", "추천여부", "추천사유"]
    have = [c for c in cols if c in borrowers.columns]
    return borrowers[have].copy()
