# -*- coding: utf-8 -*-
"""§5-1 학습기억 무결성·백업·누적통계 테스트."""
import os

import pandas as pd

from corona_reco import store as store_mod


def _store(tmp_path):
    return store_mod.Store(str(tmp_path / "data" / "corona.sqlite"))


def test_fresh_store_is_red(tmp_path):
    s = _store(tmp_path)
    s.ensure_store_state()
    integ = s.check_integrity(model_exists=False)
    assert integ["level"] == "red"        # 학습표본 0 → 위험
    assert any("찾을 수 없습니다" in m for m in integ["messages"])
    s.close()


def test_store_id_stable_and_green(tmp_path):
    s = _store(tmp_path)
    st1 = s.ensure_store_state()
    sid = st1["store_id"]
    # 학습 통계 채우면 green
    s.update_train_stats(cum_samples=1000, cum_positive=50, feedback_months=0, is_first=True)
    s.mark_clean()
    integ = s.check_integrity(model_exists=True)
    assert integ["level"] == "green"
    assert integ["store_id"] == sid
    s.close()
    # 재오픈 시 store_id 유지
    s2 = _store(tmp_path)
    assert s2.ensure_store_state()["store_id"] == sid
    s2.close()


def test_dirty_flag_yellow(tmp_path):
    s = _store(tmp_path)
    s.ensure_store_state()
    s.update_train_stats(cum_samples=1000, cum_positive=50, feedback_months=1, is_first=True)
    s.mark_dirty()  # 비정상 종료 흔적 남김(mark_clean 안 함)
    integ = s.check_integrity(model_exists=True)
    assert integ["level"] == "yellow"
    assert any("비정상 종료" in m for m in integ["messages"])
    s.close()


def test_backup_creates_file(tmp_path):
    s = _store(tmp_path)
    s.ensure_store_state()
    s.update_train_stats(cum_samples=10, cum_positive=1, feedback_months=0, is_first=True)
    b = s.backup()
    assert b and os.path.exists(b)
    assert len(s.list_backups()) >= 1
    s.close()


def test_training_rows_persist(tmp_path):
    s = _store(tmp_path)
    s.ensure_store_state()
    df = pd.DataFrame([{"고객번호": "C1", "현재원금": 1000000},
                       {"고객번호": "C2", "현재원금": 2000000}])
    n = s.save_training_rows(df, "최초학습")
    assert n == 2 and s.training_row_count() == 2
    s.close()


def test_checksum_mismatch_yellow(tmp_path):
    s = _store(tmp_path)
    s.ensure_store_state()
    s.update_train_stats(cum_samples=10, cum_positive=1, feedback_months=0, is_first=True)
    s.mark_clean()  # 체크섬 저장
    # 이후 데이터 추가 → 체크섬 불일치
    s.save_training_rows(pd.DataFrame([{"고객번호": "X"}]), "최초학습")
    integ = s.check_integrity(model_exists=True)
    assert integ["level"] == "yellow"
    assert any("체크섬" in m for m in integ["messages"])
    s.close()
