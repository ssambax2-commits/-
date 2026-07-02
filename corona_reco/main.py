# -*- coding: utf-8 -*-
"""
main.py — 데스크톱 GUI 진입점 (§10)
====================================
모드 선택, 파일 선택(학습/활동/전월 피드백), 평가 기준일, 보정 옵션·가중치·
등급컷, 실행, 진단·지표 표시, 엑셀 2종 저장 경로, 진행 로그.

UI는 ttkbootstrap(현대적 플랫 테마)을 사용하며, 없을 경우 표준 ttk로 자동
폴백한다. 무거운 파이프라인 모듈은 지연 임포트하여 창이 즉시 뜨게 한다.
실행(추론) 중 외부 네트워크 호출은 없다.
"""
from __future__ import annotations

import datetime as _dt
import os
import queue
import sys
import threading
import traceback

# 경량 상수만 즉시 로드(무거운 파이프라인은 실행 시 지연 임포트)
try:
    from corona_reco import config
except ImportError:  # 스크립트 직접 실행 대비
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from corona_reco import config

DEFAULT_THEME = "cosmo"   # ttkbootstrap 라이트 테마(전문적·플랫)
_FALLBACK_THEMES = ["cosmo", "flatly", "litera", "yeti", "minty", "darkly"]

# 팔레트(폴백 tk 위젯 색상용)
COLOR_PRIMARY = "#2b6cb0"
COLOR_HEADER_FG = "#ffffff"
COLOR_MUTED = "#5a6b7b"


def app_base_dir() -> str:
    """exe(frozen) 옆 또는 스크립트 폴더. SQLite/모델/데이터 저장 기준."""
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def _pick_font(tkfont):
    """한글 렌더링 가능한 기본 글꼴 선택(플랫폼 무관)."""
    prefer = ["Malgun Gothic", "맑은 고딕", "AppleGothic", "NanumGothic",
              "NanumSquareRound", "Noto Sans CJK KR", "Segoe UI", "Helvetica"]
    try:
        fams = set(tkfont.families())
    except Exception:  # noqa: BLE001
        fams = set()
    for f in prefer:
        if f in fams:
            return f
    return "TkDefaultFont"


def build_app(theme: str = DEFAULT_THEME, base_dir: str = None):
    """모든 위젯을 구성해 반환한다(mainloop 미실행 — 렌더 테스트/실행 공용).

    반환 ctx dict: root, nb, tabs, vars, log(), poll(), on_run 등.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    import tkinter.font as tkfont

    # --- 테마 로더(ttkbootstrap 우선, 실패 시 표준 ttk) ---
    tb = None
    root = None
    using_tb = False
    try:
        import ttkbootstrap as tb  # type: ignore
        try:
            root = tb.Window(themename=theme)
        except Exception:
            root = tb.Window(themename="flatly")
        using_tb = True
    except Exception:  # noqa: BLE001
        root = tk.Tk()

    def bs(widget_kwargs, **extra):
        """ttkbootstrap일 때만 bootstyle 등 전달, 아니면 제거."""
        kw = dict(widget_kwargs)
        kw.update(extra)
        if not using_tb:
            kw.pop("bootstyle", None)
        return kw

    base_dir = base_dir or app_base_dir()
    default_data_dir = os.path.join(base_dir, config.DATA_DIRNAME)

    root.title(f"{config.APP_NAME}  ·  자발 상환 우선순위 추천")
    root.geometry("960x780")
    root.minsize(880, 700)

    fam = _pick_font(tkfont)
    base_font = (fam, 10)
    try:
        for fn in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont"):
            tkfont.nametofont(fn).configure(family=fam, size=10)
    except Exception:  # noqa: BLE001
        pass
    mono = (fam, 9)

    log_q: "queue.Queue" = queue.Queue()
    state = {"running": False, "last": None}

    # ======================= 헤더 배너 =======================
    if using_tb:
        header = tb.Frame(root, bootstyle="primary")
    else:
        header = tk.Frame(root, bg=COLOR_PRIMARY)
    header.pack(fill="x")
    inner = (tb.Frame(header, bootstyle="primary") if using_tb
             else tk.Frame(header, bg=COLOR_PRIMARY))
    inner.pack(fill="x", padx=18, pady=12)

    def hlabel(parent, text, size, bold, muted=False):
        if using_tb:
            style = "inverse-primary"
            lb = tb.Label(parent, text=text, bootstyle=style,
                          font=(fam, size, "bold" if bold else "normal"))
        else:
            lb = tk.Label(parent, text=text, bg=COLOR_PRIMARY, fg=COLOR_HEADER_FG,
                          font=(fam, size, "bold" if bold else "normal"))
        return lb

    hlabel(inner, config.APP_NAME, 17, True).pack(anchor="w")
    hlabel(inner, "채무조정 폐지 코로나채권 중 1개월 내 자발 상환 가능성이 높은 채권을 "
                  "부담당자별로 추천합니다.", 10, False).pack(anchor="w", pady=(2, 0))
    hlabel(inner, f"버전 {config.APP_VERSION}   ·   실행(추론) 중 외부 네트워크 호출 없음",
           9, False).pack(anchor="w", pady=(2, 0))

    # ======================= 노트북(탭) =======================
    nb = ttk.Notebook(root)
    nb.pack(fill="both", expand=True, padx=12, pady=12)
    tab_run = ttk.Frame(nb, padding=10)
    tab_opt = ttk.Frame(nb, padding=10)
    tab_diag = ttk.Frame(nb, padding=10)
    nb.add(tab_run, text="  ▶  실행  ")
    nb.add(tab_opt, text="  ⚙  옵션(가중치·등급컷)  ")
    nb.add(tab_diag, text="  📊  진단·지표  ")

    # --- 변수 ---
    var_mode = tk.StringVar(value="first")
    var_activity = tk.StringVar()
    var_training = tk.StringVar()
    var_feedback = tk.StringVar()
    var_refdate = tk.StringVar(value=_dt.date.today().isoformat())
    var_month = tk.StringVar(value=_dt.date.today().strftime("%Y-%m"))
    var_datadir = tk.StringVar(value=default_data_dir)
    var_theme = tk.StringVar(value=theme)

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

    def LabelFrame(parent, text):
        return ttk.Labelframe(parent, text=text, padding=12)

    # ---------------- 실행 탭 ----------------
    mode_lf = LabelFrame(tab_run, "① 실행 모드")
    mode_lf.pack(fill="x", pady=(0, 8))
    ttk.Radiobutton(mode_lf, text="최초 실행  ·  학습데이터 + 활동데이터(모델 학습)",
                    variable=var_mode, value="first").pack(anchor="w", pady=2)
    ttk.Radiobutton(mode_lf, text="월별 실행  ·  활동데이터 + 전월 피드백(누적 학습, 학습데이터 재업로드 없음)",
                    variable=var_mode, value="monthly").pack(anchor="w", pady=2)

    files_lf = LabelFrame(tab_run, "② 입력 파일")
    files_lf.pack(fill="x", pady=8)
    files_lf.columnconfigure(1, weight=1)

    def file_row(r, label, var, patterns, hint):
        ttk.Label(files_lf, text=label, width=16).grid(row=r, column=0, sticky="w", pady=4)
        ent = ttk.Entry(files_lf, textvariable=var)
        ent.grid(row=r, column=1, sticky="ew", padx=6)

        def pick():
            p = filedialog.askopenfilename(filetypes=patterns)
            if p:
                var.set(p)
        ttk.Button(files_lf, **bs({}, text="찾기…", command=pick, width=8,
                                  bootstyle="secondary-outline")).grid(row=r, column=2)
        ttk.Label(files_lf, text=hint, foreground=COLOR_MUTED,
                  font=(fam, 8)).grid(row=r + 1, column=1, sticky="w", padx=6)

    file_row(0, "활동데이터 *", var_activity,
             [("표", "*.csv *.xlsx *.xls"), ("모든 파일", "*.*")], "이번 달 관리 채권(추천 대상). 필수.")
    file_row(2, "학습데이터", var_training,
             [("표", "*.csv *.xlsx *.xls"), ("모든 파일", "*.*")], "최초 실행에만. 입금내역이 오른쪽에 붙은 파일.")
    file_row(4, "전월 피드백", var_feedback,
             [("엑셀", "*.xlsx *.xls"), ("모든 파일", "*.*")], "월별 실행에만. 지난달 피드백 파일(값 기입).")

    meta_lf = LabelFrame(tab_run, "③ 기준 설정")
    meta_lf.pack(fill="x", pady=8)
    row = ttk.Frame(meta_lf); row.pack(fill="x", pady=3)
    ttk.Label(row, text="평가 기준일", width=16).pack(side="left")
    ttk.Entry(row, textvariable=var_refdate, width=15).pack(side="left")
    ttk.Label(row, text="  (YYYY-MM-DD · 최근입금 판정 기준, 시스템 날짜 아님)",
              foreground=COLOR_MUTED, font=(fam, 8)).pack(side="left")
    row2 = ttk.Frame(meta_lf); row2.pack(fill="x", pady=3)
    ttk.Label(row2, text="추천월", width=16).pack(side="left")
    ttk.Entry(row2, textvariable=var_month, width=15).pack(side="left")
    ttk.Label(row2, text="  (YYYY-MM · DB 누적/보고서 키)",
              foreground=COLOR_MUTED, font=(fam, 8)).pack(side="left")
    row3 = ttk.Frame(meta_lf); row3.pack(fill="x", pady=3)
    ttk.Label(row3, text="데이터 폴더", width=16).pack(side="left")
    ttk.Entry(row3, textvariable=var_datadir).pack(side="left", fill="x", expand=True)

    # 실행 버튼 + 진행바
    action = ttk.Frame(tab_run); action.pack(fill="x", pady=(10, 4))
    run_btn = ttk.Button(action, **bs({}, text="  ▶  실행하기  ", bootstyle="success"))
    run_btn.pack(side="left")
    prog = ttk.Progressbar(action, mode="indeterminate", length=260,
                           **bs({}, bootstyle="success-striped"))
    prog.pack(side="left", padx=12)
    status_var = tk.StringVar(value="대기 중")
    ttk.Label(action, textvariable=status_var, foreground=COLOR_MUTED).pack(side="left")

    def theme_text_colors():
        if using_tb:
            try:
                c = root.style.colors
                return c.inputbg, c.inputfg
            except Exception:  # noqa: BLE001
                pass
        return "#f7f9fb", "#1a1a1a"

    log_lf = LabelFrame(tab_run, "실행 로그")
    log_lf.pack(fill="both", expand=True, pady=(8, 0))
    _lbg, _lfg = theme_text_colors()
    log_txt = tk.Text(log_lf, height=10, wrap="word", font=mono, relief="flat",
                      background=_lbg, foreground=_lfg, insertbackground=_lfg,
                      borderwidth=0)
    log_txt.pack(fill="both", expand=True, side="left")
    log_sb = ttk.Scrollbar(log_lf, command=log_txt.yview)
    log_sb.pack(side="right", fill="y")
    log_txt.config(yscrollcommand=log_sb.set)

    def log(msg):
        log_txt.insert("end", f"· {msg}\n")
        log_txt.see("end")

    # ---------------- 옵션 탭 ----------------
    wfrm = LabelFrame(tab_opt, "가중치 (합은 자동 정규화)")
    wfrm.pack(fill="x", pady=(0, 8))
    grid = ttk.Frame(wfrm); grid.pack(fill="x")
    for i, (lbl, v) in enumerate([("base 규칙점수", var_w_base),
                                  ("paid_similarity ML", var_w_sim),
                                  ("payment_history 입금", var_w_pay),
                                  ("burden 잔액부담", var_w_burden)]):
        ttk.Label(grid, text=lbl, width=22).grid(row=i // 2, column=(i % 2) * 2,
                                                  sticky="w", padx=6, pady=4)
        ttk.Entry(grid, textvariable=v, width=8).grid(row=i // 2, column=(i % 2) * 2 + 1,
                                                      sticky="w", pady=4)

    cfrm = LabelFrame(tab_opt, "등급 절대컷")
    cfrm.pack(fill="x", pady=8)
    line = ttk.Frame(cfrm); line.pack(fill="x")
    for lbl, v in [("S ≥", var_cut_s), ("A ≥", var_cut_a), ("B ≥", var_cut_b), ("C ≥", var_cut_c)]:
        cell = ttk.Frame(line); cell.pack(side="left", padx=(0, 16))
        ttk.Label(cell, text=lbl).pack(side="left", padx=(0, 4))
        ttk.Entry(cell, textvariable=v, width=6).pack(side="left")
    ttk.Label(cfrm, text="C컷 미만은 D. 담보부NPL 등급상한이 별도로 적용됩니다.",
              foreground=COLOR_MUTED, font=(fam, 8)).pack(anchor="w", pady=(6, 0))

    pfrm = LabelFrame(tab_opt, "부담당자 상한 · 최소 추천 보장")
    pfrm.pack(fill="x", pady=8)
    line2 = ttk.Frame(pfrm); line2.pack(fill="x")
    for lbl, v in [("S 최대(명)", var_cap_s), ("A 최대(명)", var_cap_a), ("최소 추천(명)", var_min)]:
        cell = ttk.Frame(line2); cell.pack(side="left", padx=(0, 16))
        ttk.Label(cell, text=lbl).pack(side="left", padx=(0, 4))
        ttk.Entry(cell, textvariable=v, width=7).pack(side="left")

    ofrm = LabelFrame(tab_opt, "기타 옵션")
    ofrm.pack(fill="x", pady=8)
    ttk.Checkbutton(ofrm, text="소멸시효 완성채권 제외 (권장 ON)",
                    variable=var_exclude_expired).pack(anchor="w", pady=2)
    ttk.Checkbutton(ofrm, text="상세진단(개발용) 시트 포함",
                    variable=var_detail).pack(anchor="w", pady=2)
    ttk.Checkbutton(ofrm, text="Stage2 예상회수 회귀 사용 (표본 충분 시)",
                    variable=var_stage2).pack(anchor="w", pady=2)

    tfrm = LabelFrame(tab_opt, "화면 테마")
    tfrm.pack(fill="x", pady=8)
    trow = ttk.Frame(tfrm); trow.pack(fill="x")
    ttk.Label(trow, text="테마", width=8).pack(side="left")
    theme_cb = ttk.Combobox(trow, textvariable=var_theme, width=16,
                            values=_FALLBACK_THEMES, state="readonly")
    theme_cb.pack(side="left")

    def on_theme(*_):
        if using_tb:
            try:
                root.style.theme_use(var_theme.get())
                lbg, lfg = theme_text_colors()
                log_txt.config(background=lbg, foreground=lfg, insertbackground=lfg)
            except Exception:  # noqa: BLE001
                pass
    theme_cb.bind("<<ComboboxSelected>>", on_theme)
    if not using_tb:
        ttk.Label(tfrm, text="(ttkbootstrap 미탑재 — 표준 테마)",
                  foreground=COLOR_MUTED, font=(fam, 8)).pack(anchor="w", pady=(4, 0))

    # ---------------- 진단 탭 ----------------
    diag_lf = LabelFrame(tab_diag, "실행 결과 진단·지표")
    diag_lf.pack(fill="both", expand=True)
    diag_tv = ttk.Treeview(diag_lf, columns=("항목", "값"), show="headings", height=16)
    diag_tv.heading("항목", text="항목")
    diag_tv.heading("값", text="값")
    diag_tv.column("항목", width=320, anchor="w")
    diag_tv.column("값", width=360, anchor="w")
    diag_tv.pack(fill="both", expand=True, side="left")
    diag_sb = ttk.Scrollbar(diag_lf, command=diag_tv.yview)
    diag_sb.pack(side="right", fill="y")
    diag_tv.config(yscrollcommand=diag_sb.set)
    warn_lbl = ttk.Label(tab_diag, text="", foreground="#b23b3b", font=(fam, 9), wraplength=880)
    warn_lbl.pack(anchor="w", pady=(8, 0))

    # ======================= 실행 로직 =======================
    def build_options():
        from corona_reco import pipeline  # 지연 임포트
        ref = _dt.date.fromisoformat(var_refdate.get().strip())
        weights = {
            "base": float(var_w_base.get()), "paid_similarity": float(var_w_sim.get()),
            "payment_history": float(var_w_pay.get()), "burden": float(var_w_burden.get()),
        }
        cuts = {"S": float(var_cut_s.get()), "A": float(var_cut_a.get()),
                "B": float(var_cut_b.get()), "C": float(var_cut_c.get())}
        return pipeline.RunOptions(
            mode=var_mode.get(), activity_path=var_activity.get().strip(),
            training_path=(var_training.get().strip() or None),
            prev_feedback_path=(var_feedback.get().strip() or None),
            ref_date=ref, month=var_month.get().strip(),
            data_dir=var_datadir.get().strip(), weights=weights, cuts=cuts,
            cap_s=int(var_cap_s.get()), cap_a=int(var_cap_a.get()), min_reco=int(var_min.get()),
            exclude_expired_prescription=var_exclude_expired.get(),
            include_detail=var_detail.get(), stage2_enabled=var_stage2.get())

    def worker(opt):
        try:
            from corona_reco import pipeline
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
        status_var.set("실행 중…")
        log("실행 시작")
        threading.Thread(target=worker, args=(opt,), daemon=True).start()

    run_btn.config(command=on_run)

    def _finish(res):
        prog.stop(); run_btn.config(state="normal"); state["running"] = False
        state["last"] = res
        status_var.set("완료")
        log("완료!")
        log(f"추천 보고서: {res.report_path}")
        log(f"피드백 파일: {res.feedback_path}")
        for i in diag_tv.get_children():
            diag_tv.delete(i)
        for k, v in res.diagnostics.items():
            diag_tv.insert("", "end", values=(str(k), str(v)))
        warn_lbl.config(text=("⚠ " + "  |  ".join(res.warnings)) if res.warnings else "")
        nb.select(tab_diag)
        messagebox.showinfo("완료", f"추천 보고서와 피드백 파일을 생성했습니다.\n\n{res.report_path}")

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
                    status_var.set("오류")
                    log("오류: " + str(payload).splitlines()[0])
                    messagebox.showerror("실행 오류", str(payload).splitlines()[0])
        except queue.Empty:
            pass
        root.after(150, poll)

    ctx = {
        "root": root, "nb": nb, "tabs": (tab_run, tab_opt, tab_diag),
        "log": log, "poll": poll, "on_run": on_run, "using_tb": using_tb,
        "run_btn": run_btn,
    }
    return ctx


def _run_gui(theme: str = DEFAULT_THEME):
    ctx = build_app(theme=theme)
    ctx["poll"]()
    ctx["root"].mainloop()


def main():
    _run_gui()


if __name__ == "__main__":
    main()
