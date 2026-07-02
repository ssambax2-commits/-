# -*- coding: utf-8 -*-
"""
main.py — tkinter GUI 진입점 (§10)
===================================
모드 선택, 파일 선택(학습/활동/전월 피드백), 평가 기준일, 보정 옵션·가중치·
등급컷, 실행, 진단·지표 표시, 엑셀 2종 저장 경로, 진행 로그.
실행(추론) 중 외부 네트워크 호출은 없다.
"""
from __future__ import annotations

import datetime as _dt
import os
import queue
import sys
import threading
import traceback

# 패키지/단독 실행 모두 지원
try:
    from corona_reco import config, pipeline, store as store_mod
except ImportError:  # 스크립트 직접 실행 시
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from corona_reco import config, pipeline, store as store_mod


def app_base_dir() -> str:
    """exe(frozen) 옆 또는 스크립트 폴더. SQLite/모델/데이터 저장 기준."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title(f"{config.APP_NAME}  v{config.APP_VERSION}")
    root.geometry("860x720")

    log_q: "queue.Queue[str]" = queue.Queue()
    state = {"running": False}

    default_data_dir = os.path.join(app_base_dir(), config.DATA_DIRNAME)

    # ------------------------- 변수 -------------------------
    var_mode = tk.StringVar(value="first")
    var_activity = tk.StringVar()
    var_training = tk.StringVar()
    var_feedback = tk.StringVar()
    var_refdate = tk.StringVar(value=_dt.date.today().isoformat())
    var_month = tk.StringVar(value=_dt.date.today().strftime("%Y-%m"))
    var_datadir = tk.StringVar(value=default_data_dir)

    var_w_base = tk.StringVar(value=str(config.DEFAULT_WEIGHTS["base"]))
    var_w_sim = tk.StringVar(value=str(config.DEFAULT_WEIGHTS["paid_similarity"]))
    var_w_pay = tk.StringVar(value=str(config.DEFAULT_WEIGHTS["payment_history"]))
    var_w_burden = tk.StringVar(value=str(config.DEFAULT_WEIGHTS["burden"]))

    var_cut_s = tk.StringVar(value=str(config.GRADE_CUTS["S"]))
    var_cut_a = tk.StringVar(value=str(config.GRADE_CUTS["A"]))
    var_cut_b = tk.StringVar(value=str(config.GRADE_CUTS["B"]))
    var_cut_c = tk.StringVar(value=str(config.GRADE_CUTS["C"]))

    var_cap_s = tk.StringVar(value=str(config.CAP_S_PER_ASSIGNEE))
    var_cap_a = tk.StringVar(value=str(config.CAP_A_PER_ASSIGNEE))
    var_min = tk.StringVar(value=str(config.MIN_RECO_PER_ASSIGNEE))
    var_exclude_expired = tk.BooleanVar(value=config.EXCLUDE_EXPIRED_PRESCRIPTION_DEFAULT)
    var_detail = tk.BooleanVar(value=False)
    var_stage2 = tk.BooleanVar(value=config.STAGE2_ENABLED_DEFAULT)

    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=8, pady=8)
    tab_run = ttk.Frame(nb)
    tab_opt = ttk.Frame(nb)
    tab_diag = ttk.Frame(nb)
    nb.add(tab_run, text="실행")
    nb.add(tab_opt, text="옵션(가중치·등급컷)")
    nb.add(tab_diag, text="진단·지표")

    # ------------------------- 실행 탭 -------------------------
    frm = ttk.LabelFrame(tab_run, text="실행 모드")
    frm.pack(fill="x", padx=6, pady=6)
    ttk.Radiobutton(frm, text="최초 실행(학습데이터 + 활동데이터)", variable=var_mode,
                    value="first").pack(anchor="w", padx=8, pady=2)
    ttk.Radiobutton(frm, text="월별 실행(활동데이터 + 전월 피드백)", variable=var_mode,
                    value="monthly").pack(anchor="w", padx=8, pady=2)

    def _file_row(parent, label, var, patterns):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=3)
        ttk.Label(row, text=label, width=16).pack(side="left")
        ttk.Entry(row, textvariable=var).pack(side="left", fill="x", expand=True)

        def pick():
            p = filedialog.askopenfilename(filetypes=patterns)
            if p:
                var.set(p)
        ttk.Button(row, text="찾기", command=pick).pack(side="left", padx=4)

    files = ttk.LabelFrame(tab_run, text="입력 파일")
    files.pack(fill="x", padx=6, pady=6)
    _file_row(files, "활동데이터", var_activity,
              [("표", "*.csv *.xlsx *.xls"), ("모든 파일", "*.*")])
    _file_row(files, "학습데이터(최초)", var_training,
              [("표", "*.csv *.xlsx *.xls"), ("모든 파일", "*.*")])
    _file_row(files, "전월 피드백(월별)", var_feedback,
              [("엑셀", "*.xlsx *.xls"), ("모든 파일", "*.*")])

    meta = ttk.LabelFrame(tab_run, text="기준 설정")
    meta.pack(fill="x", padx=6, pady=6)
    r1 = ttk.Frame(meta); r1.pack(fill="x", padx=8, pady=3)
    ttk.Label(r1, text="평가 기준일(YYYY-MM-DD)", width=24).pack(side="left")
    ttk.Entry(r1, textvariable=var_refdate, width=16).pack(side="left")
    ttk.Label(r1, text="   추천월(YYYY-MM)", width=16).pack(side="left")
    ttk.Entry(r1, textvariable=var_month, width=12).pack(side="left")
    r2 = ttk.Frame(meta); r2.pack(fill="x", padx=8, pady=3)
    ttk.Label(r2, text="데이터 폴더", width=24).pack(side="left")
    ttk.Entry(r2, textvariable=var_datadir).pack(side="left", fill="x", expand=True)

    btns = ttk.Frame(tab_run)
    btns.pack(fill="x", padx=6, pady=6)
    run_btn = ttk.Button(btns, text="실행")
    run_btn.pack(side="left")
    prog = ttk.Progressbar(btns, mode="indeterminate", length=280)
    prog.pack(side="left", padx=10)

    log_txt = tk.Text(tab_run, height=16, wrap="word")
    log_txt.pack(fill="both", expand=True, padx=6, pady=6)

    def log(msg):
        log_txt.insert("end", msg + "\n")
        log_txt.see("end")

    # ------------------------- 옵션 탭 -------------------------
    wfrm = ttk.LabelFrame(tab_opt, text="가중치 (합은 자동 정규화)")
    wfrm.pack(fill="x", padx=6, pady=6)
    for lbl, v in [("base", var_w_base), ("paid_similarity", var_w_sim),
                   ("payment_history", var_w_pay), ("burden", var_w_burden)]:
        row = ttk.Frame(wfrm); row.pack(fill="x", padx=8, pady=2)
        ttk.Label(row, text=lbl, width=20).pack(side="left")
        ttk.Entry(row, textvariable=v, width=10).pack(side="left")

    cfrm = ttk.LabelFrame(tab_opt, text="등급 절대컷")
    cfrm.pack(fill="x", padx=6, pady=6)
    for lbl, v in [("S≥", var_cut_s), ("A≥", var_cut_a), ("B≥", var_cut_b), ("C≥", var_cut_c)]:
        row = ttk.Frame(cfrm); row.pack(side="left", padx=8, pady=2)
        ttk.Label(row, text=lbl).pack(side="left")
        ttk.Entry(row, textvariable=v, width=6).pack(side="left")

    pfrm = ttk.LabelFrame(tab_opt, text="부담당자 상한 · 최소보장")
    pfrm.pack(fill="x", padx=6, pady=6)
    for lbl, v in [("S 최대", var_cap_s), ("A 최대", var_cap_a), ("최소 추천", var_min)]:
        row = ttk.Frame(pfrm); row.pack(side="left", padx=8, pady=2)
        ttk.Label(row, text=lbl).pack(side="left")
        ttk.Entry(row, textvariable=v, width=6).pack(side="left")

    ofrm = ttk.LabelFrame(tab_opt, text="기타 옵션")
    ofrm.pack(fill="x", padx=6, pady=6)
    ttk.Checkbutton(ofrm, text="소멸시효 완성채권 제외(권장 ON)",
                    variable=var_exclude_expired).pack(anchor="w", padx=8, pady=2)
    ttk.Checkbutton(ofrm, text="상세진단(개발용) 시트 포함",
                    variable=var_detail).pack(anchor="w", padx=8, pady=2)
    ttk.Checkbutton(ofrm, text="Stage2 예상회수 회귀 사용(표본 충분 시)",
                    variable=var_stage2).pack(anchor="w", padx=8, pady=2)

    # ------------------------- 진단 탭 -------------------------
    diag_txt = tk.Text(tab_diag, wrap="word")
    diag_txt.pack(fill="both", expand=True, padx=6, pady=6)

    # ------------------------- 실행 로직 -------------------------
    def build_options():
        ref = _dt.date.fromisoformat(var_refdate.get().strip())
        weights = {
            "base": float(var_w_base.get()),
            "paid_similarity": float(var_w_sim.get()),
            "payment_history": float(var_w_pay.get()),
            "burden": float(var_w_burden.get()),
        }
        cuts = {"S": float(var_cut_s.get()), "A": float(var_cut_a.get()),
                "B": float(var_cut_b.get()), "C": float(var_cut_c.get())}
        return pipeline.RunOptions(
            mode=var_mode.get(),
            activity_path=var_activity.get().strip(),
            training_path=(var_training.get().strip() or None),
            prev_feedback_path=(var_feedback.get().strip() or None),
            ref_date=ref, month=var_month.get().strip(),
            data_dir=var_datadir.get().strip(),
            weights=weights, cuts=cuts,
            cap_s=int(var_cap_s.get()), cap_a=int(var_cap_a.get()),
            min_reco=int(var_min.get()),
            exclude_expired_prescription=var_exclude_expired.get(),
            include_detail=var_detail.get(),
            stage2_enabled=var_stage2.get(),
        )

    def worker(opt):
        try:
            res = pipeline.run_pipeline(opt, progress=lambda m: log_q.put(("log", m)))
            log_q.put(("done", res))
        except Exception as e:  # noqa: BLE001
            log_q.put(("error", f"{e}\n{traceback.format_exc()}"))

    def on_run():
        if state["running"]:
            return
        if not var_activity.get().strip():
            messagebox.showwarning("입력 필요", "활동데이터 파일을 선택하세요.")
            return
        if var_mode.get() == "first" and not var_training.get().strip():
            if not messagebox.askyesno("확인", "학습데이터 없이 진행하면 규칙/prior 폴백으로 동작합니다. 계속할까요?"):
                return
        if var_mode.get() == "monthly" and not var_feedback.get().strip():
            if not messagebox.askyesno("확인", "전월 피드백 없이 월별 실행합니다. 계속할까요?"):
                return
        try:
            opt = build_options()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("옵션 오류", str(e))
            return
        state["running"] = True
        run_btn.config(state="disabled")
        prog.start(12)
        log("실행 시작...")
        threading.Thread(target=worker, args=(opt,), daemon=True).start()

    run_btn.config(command=on_run)

    def poll():
        try:
            while True:
                kind, payload = log_q.get_nowait()
                if kind == "log":
                    log(payload)
                elif kind == "done":
                    _finish(payload)
                elif kind == "error":
                    prog.stop(); run_btn.config(state="normal"); state["running"] = False
                    log("오류: " + payload)
                    messagebox.showerror("실행 오류", payload.splitlines()[0])
        except queue.Empty:
            pass
        root.after(150, poll)

    def _finish(res):
        prog.stop()
        run_btn.config(state="normal")
        state["running"] = False
        log("완료!")
        log(f"추천 보고서: {res.report_path}")
        log(f"피드백 파일: {res.feedback_path}")
        diag_txt.delete("1.0", "end")
        diag_txt.insert("end", "■ 진단·지표\n")
        for k, v in res.diagnostics.items():
            diag_txt.insert("end", f"  - {k}: {v}\n")
        if res.warnings:
            diag_txt.insert("end", "\n■ 경고\n")
            for w in res.warnings:
                diag_txt.insert("end", f"  ! {w}\n")
        nb.select(tab_diag)
        messagebox.showinfo("완료", f"추천 보고서와 피드백 파일을 생성했습니다.\n{res.report_path}")

    poll()
    root.mainloop()


def main():
    _run_gui()


if __name__ == "__main__":
    main()
