# -*- coding: utf-8 -*-
"""
main.py — CustomTkinter 데스크톱 GUI (§4 애플풍 · §5-1 무결성 배너 · §3 랜덤박스)
================================================================================
- CustomTkinter(둥근 모서리·여백·플랫·라이트/다크 토글, 시스템 폰트). 오프라인 번들.
- 창 확대(+80px), 카드형 레이아웃. 웰컴 CI(레드 배너 + 오렌지 포인트) 계승.
- 상단 학습기억 무결성 배너(🟢/🟡/🔴) + 실행 시 초기화 방지 모달.
- 랜덤박스 버튼(합산 1,000만↑ 상위20 중 무작위5, 매 클릭 상이).
- 무거운 파이프라인은 지연 임포트하여 창이 즉시 뜬다. 실행 중 외부호출 없음.
"""
from __future__ import annotations

import datetime as _dt
import os
import queue
import sys
import threading
import traceback

try:
    from corona_reco import config
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from corona_reco import config

# 웰컴금융그룹 CI
WELCOME_RED = "#EA002C"
WELCOME_RED_DK = "#B80023"
WELCOME_ORANGE = "#FF6B00"
WELCOME_ORANGE_DK = "#E85D00"
WARM_GRAY = "#EAE0D3"


def app_base_dir() -> str:
    if getattr(sys, "frozen", False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))


def default_data_dir() -> str:
    return os.path.join(app_base_dir(), config.DATA_DIRNAME)


# ---------------------------------------------------------------------------
def build_app(appearance: str = "light", data_dir: str = None):
    """CustomTkinter 앱 구성 후 ctx 반환(mainloop 미실행 — 렌더/실행 공용)."""
    import customtkinter as ctk
    from tkinter import filedialog, messagebox

    ctk.set_appearance_mode(appearance)
    ctk.set_default_color_theme("dark-blue")

    root = ctk.CTk()
    root.title(f"{config.APP_NAME} · 우선관리 추천")
    root.geometry("1040x860")          # 기존 960x780 대비 +80px(§4)
    root.minsize(980, 800)

    data_dir = data_dir or default_data_dir()
    log_q: "queue.Queue" = queue.Queue()
    state = {"running": False, "last_borrowers": None, "rb_win": None,
             "data_dir": data_dir}

    FONT = ("Malgun Gothic", 13)
    FONT_B = ("Malgun Gothic", 14, "bold")
    FONT_TITLE = ("Malgun Gothic", 20, "bold")
    FONT_SM = ("Malgun Gothic", 11)

    # ===================== 헤더(웰컴 레드 배너) =====================
    header = ctk.CTkFrame(root, fg_color=WELCOME_RED, corner_radius=0, height=88)
    header.pack(fill="x")
    header.pack_propagate(False)
    hin = ctk.CTkFrame(header, fg_color="transparent")
    hin.pack(fill="both", expand=True, padx=20, pady=12)
    badge = ctk.CTkLabel(hin, text=" W ", font=("Malgun Gothic", 26, "bold"),
                         text_color=WELCOME_RED, fg_color="#FFFFFF", corner_radius=24,
                         width=48, height=48)
    badge.pack(side="left", padx=(0, 16))
    htext = ctk.CTkFrame(hin, fg_color="transparent")
    htext.pack(side="left", fill="y")
    ctk.CTkLabel(htext, text="웰컴금융그룹 · WELCOME FINANCIAL GROUP",
                 font=FONT_SM, text_color=WARM_GRAY).pack(anchor="w")
    ctk.CTkLabel(htext, text=config.APP_NAME, font=FONT_TITLE,
                 text_color="#FFFFFF").pack(anchor="w")
    ctk.CTkLabel(htext, text="접촉 시 자발 변제가능성이 높은 차주를 부담당자별로 추천",
                 font=FONT_SM, text_color="#FFE9DC").pack(anchor="w")
    # 오렌지 포인트 스트라이프
    ctk.CTkFrame(root, fg_color=WELCOME_ORANGE, corner_radius=0, height=4).pack(fill="x")

    # ===================== 무결성 배너(§5-1) =====================
    banner = ctk.CTkFrame(root, corner_radius=8, height=44)
    banner.pack(fill="x", padx=12, pady=(8, 0))
    banner_lbl = ctk.CTkLabel(banner, text="학습기억 상태 점검 중…", font=FONT,
                              anchor="w", justify="left")
    banner_lbl.pack(side="left", fill="x", expand=True, padx=12, pady=6)
    state["integrity"] = None

    def refresh_banner():
        try:
            from corona_reco import store as store_mod, pipeline as pl
            s = store_mod.Store(os.path.join(state["data_dir"], config.SQLITE_FILENAME))
            s.ensure_store_state()
            integ = s.check_integrity(os.path.exists(pl._model_path(state["data_dir"])))
            s.close()
        except Exception as e:  # noqa: BLE001
            integ = {"level": "yellow", "messages": [f"상태 확인 실패: {e}"], "state": None}
        state["integrity"] = integ
        color = {"green": "#1E7F4F", "yellow": "#B8860B", "red": WELCOME_RED}[integ["level"]]
        icon = {"green": "🟢", "yellow": "🟡", "red": "🔴"}[integ["level"]]
        banner.configure(fg_color=color)
        banner_lbl.configure(text=icon + " " + "  ".join(integ["messages"])[:200],
                             text_color="#FFFFFF")

    # ===================== 본문 탭 =====================
    tabs = ctk.CTkTabview(root, corner_radius=10)
    tabs.pack(fill="both", expand=True, padx=12, pady=10)
    tab_run = tabs.add("  실행  ")
    tab_opt = tabs.add("  옵션  ")
    tab_diag = tabs.add("  진단·지표  ")

    var = {
        "mode": ctk.StringVar(value="first"),
        "activity": ctk.StringVar(),
        "training": ctk.StringVar(),
        "feedback": ctk.StringVar(),
        "refdate": ctk.StringVar(value=_dt.date.today().isoformat()),
        "month": ctk.StringVar(value=_dt.date.today().strftime("%Y-%m")),
        "datadir": ctk.StringVar(value=data_dir),
        "w_base": ctk.StringVar(value=str(config.DEFAULT_WEIGHTS["base"])),
        "w_sim": ctk.StringVar(value=str(config.DEFAULT_WEIGHTS["paid_similarity"])),
        "w_pay": ctk.StringVar(value=str(config.DEFAULT_WEIGHTS["payment_history"])),
        "w_burden": ctk.StringVar(value=str(config.DEFAULT_WEIGHTS["burden"])),
        "base_cap": ctk.StringVar(value=str(config.ASSIGNEE_BASE_CAP)),
        "hard_cap": ctk.StringVar(value=str(config.ASSIGNEE_HARD_CAP)),
        "detail": ctk.BooleanVar(value=False),
        "exclude_expired": ctk.BooleanVar(value=config.EXCLUDE_EXPIRED_PRESCRIPTION_DEFAULT),
        "appearance": ctk.StringVar(value=appearance),
    }

    def card(parent, title):
        c = ctk.CTkFrame(parent, corner_radius=10)
        c.pack(fill="x", padx=10, pady=8)
        ctk.CTkLabel(c, text=title, font=FONT_B).pack(anchor="w", padx=14, pady=(10, 2))
        return c

    # ---- 실행 탭 ----
    run_scroll = ctk.CTkScrollableFrame(tab_run, fg_color="transparent")
    run_scroll.pack(fill="both", expand=True)

    c1 = card(run_scroll, "① 실행 모드")
    ctk.CTkRadioButton(c1, text="최초 실행  ·  학습데이터 + 활동데이터(모델 학습)",
                       variable=var["mode"], value="first", font=FONT).pack(anchor="w", padx=16, pady=3)
    ctk.CTkRadioButton(c1, text="월별 실행  ·  활동데이터 + 전월 피드백(누적 재학습)",
                       variable=var["mode"], value="monthly", font=FONT).pack(anchor="w", padx=16, pady=(0, 10))

    c2 = card(run_scroll, "② 입력 파일")

    def file_row(parent, label, key, patterns, hint):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=3)
        ctk.CTkLabel(row, text=label, width=110, anchor="w", font=FONT).pack(side="left")
        ent = ctk.CTkEntry(row, textvariable=var[key], font=FONT)
        ent.pack(side="left", fill="x", expand=True, padx=6)

        def pick():
            p = filedialog.askopenfilename(filetypes=patterns)
            if p:
                var[key].set(p)
        ctk.CTkButton(row, text="찾기", width=64, command=pick,
                      fg_color=WELCOME_ORANGE, hover_color=WELCOME_ORANGE_DK,
                      font=FONT).pack(side="left")
        ctk.CTkLabel(parent, text="    " + hint, font=FONT_SM,
                     text_color="gray").pack(anchor="w", padx=14)

    file_row(c2, "활동데이터 *", "activity",
             [("표", "*.csv *.xlsx *.xlsm *.xls"), ("모든 파일", "*.*")], "이번 달 관리 채권(필수)")
    file_row(c2, "학습데이터", "training",
             [("표", "*.csv *.xlsx *.xlsm *.xls"), ("모든 파일", "*.*")], "최초 실행에만(입금내역 포함)")
    file_row(c2, "전월 피드백", "feedback",
             [("엑셀", "*.xlsx *.xls"), ("모든 파일", "*.*")], "월별 실행에만(값 기입 후)")
    ctk.CTkLabel(c2, text="", height=4).pack()

    c3 = card(run_scroll, "③ 기준 설정")
    for label, key, hint in [("평가 기준일", "refdate", "YYYY-MM-DD · 최근입금 판정 기준"),
                             ("추천월", "month", "YYYY-MM · DB 누적/보고서 키"),
                             ("데이터 폴더", "datadir", "SQLite·모델·백업 저장 위치")]:
        row = ctk.CTkFrame(c3, fg_color="transparent")
        row.pack(fill="x", padx=14, pady=3)
        ctk.CTkLabel(row, text=label, width=110, anchor="w", font=FONT).pack(side="left")
        ctk.CTkEntry(row, textvariable=var[key], font=FONT).pack(side="left", fill="x", expand=True, padx=6)
        ctk.CTkLabel(row, text=hint, font=FONT_SM, text_color="gray").pack(side="left")
    ctk.CTkLabel(c3, text="", height=4).pack()

    # 실행/랜덤박스 버튼 + 진행바
    actions = ctk.CTkFrame(run_scroll, fg_color="transparent")
    actions.pack(fill="x", padx=10, pady=(4, 2))
    run_btn = ctk.CTkButton(actions, text="▶  실행하기", font=FONT_B, height=44, width=160,
                            fg_color=WELCOME_RED, hover_color=WELCOME_RED_DK)
    run_btn.pack(side="left")
    rb_btn = ctk.CTkButton(actions, text="🎲  랜덤박스", font=FONT_B, height=44, width=140,
                           fg_color=WELCOME_ORANGE, hover_color=WELCOME_ORANGE_DK)
    rb_btn.pack(side="left", padx=10)
    prog = ctk.CTkProgressBar(actions, width=220, mode="indeterminate")
    prog.pack(side="left", padx=10)
    prog.set(0)
    status_lbl = ctk.CTkLabel(actions, text="대기 중", font=FONT, text_color="gray")
    status_lbl.pack(side="left", padx=6)

    log_box = ctk.CTkTextbox(run_scroll, height=180, font=("Consolas", 11))
    log_box.pack(fill="both", expand=True, padx=10, pady=(8, 4))

    def log(msg):
        log_box.insert("end", f"· {msg}\n")
        log_box.see("end")

    # ---- 옵션 탭 ----
    o1 = card(tab_opt, "가중치 (합 자동 정규화)")
    grid = ctk.CTkFrame(o1, fg_color="transparent")
    grid.pack(fill="x", padx=14, pady=(0, 10))
    for i, (lbl, key) in enumerate([("base 규칙", "w_base"), ("paid_similarity ML", "w_sim"),
                                    ("payment 입금", "w_pay"), ("burden 잔액", "w_burden")]):
        ctk.CTkLabel(grid, text=lbl, width=150, anchor="w", font=FONT).grid(
            row=i // 2, column=(i % 2) * 2, padx=6, pady=4, sticky="w")
        ctk.CTkEntry(grid, textvariable=var[key], width=80, font=FONT).grid(
            row=i // 2, column=(i % 2) * 2 + 1, padx=6, pady=4)

    o2 = card(tab_opt, "부담당자 상한 (즉시+당월, 차주 기준)")
    row = ctk.CTkFrame(o2, fg_color="transparent")
    row.pack(fill="x", padx=14, pady=(0, 10))
    ctk.CTkLabel(row, text="기본 정원", font=FONT).pack(side="left", padx=(0, 4))
    ctk.CTkEntry(row, textvariable=var["base_cap"], width=70, font=FONT).pack(side="left", padx=(0, 16))
    ctk.CTkLabel(row, text="하드캡(고스코어 초과편입 한계)", font=FONT).pack(side="left", padx=(0, 4))
    ctk.CTkEntry(row, textvariable=var["hard_cap"], width=70, font=FONT).pack(side="left")

    o3 = card(tab_opt, "기타 옵션")
    ctk.CTkCheckBox(o3, text="소멸시효 완성채권 제외(권장)", variable=var["exclude_expired"],
                    font=FONT).pack(anchor="w", padx=16, pady=3)
    ctk.CTkCheckBox(o3, text="차주 상세(개발용) 시트 포함", variable=var["detail"],
                    font=FONT).pack(anchor="w", padx=16, pady=(0, 10))

    o4 = card(tab_opt, "화면")
    def on_appearance():
        ctk.set_appearance_mode(var["appearance"].get())
    ctk.CTkSegmentedButton(o4, values=["light", "dark"], variable=var["appearance"],
                           command=lambda _v: on_appearance(), font=FONT).pack(
        anchor="w", padx=16, pady=(0, 10))

    # ---- 진단 탭 ----
    diag_box = ctk.CTkTextbox(tab_diag, font=("Consolas", 12))
    diag_box.pack(fill="both", expand=True, padx=10, pady=10)

    # ===================== 실행 로직 =====================
    def build_options():
        from corona_reco import pipeline
        ref = _dt.date.fromisoformat(var["refdate"].get().strip())
        weights = {"base": float(var["w_base"].get()), "paid_similarity": float(var["w_sim"].get()),
                   "payment_history": float(var["w_pay"].get()), "burden": float(var["w_burden"].get())}
        return pipeline.RunOptions(
            mode=var["mode"].get(), activity_path=var["activity"].get().strip(),
            training_path=(var["training"].get().strip() or None),
            prev_feedback_path=(var["feedback"].get().strip() or None),
            ref_date=ref, month=var["month"].get().strip(),
            data_dir=var["datadir"].get().strip(), weights=weights,
            base_cap=int(var["base_cap"].get()), hard_cap=int(var["hard_cap"].get()),
            exclude_expired_prescription=var["exclude_expired"].get(),
            include_detail=var["detail"].get())

    def worker(opt):
        try:
            from corona_reco import pipeline
            res = pipeline.run_pipeline(opt, progress=lambda m: log_q.put(("log", m)))
            log_q.put(("done", res))
        except Exception as e:  # noqa: BLE001
            from corona_reco.util import UserFacingError
            if isinstance(e, UserFacingError):
                log_q.put(("uerror", e.message + (f"\n상세: {e.detail}" if e.detail else "")))
            else:
                log_q.put(("error", f"{e}\n{traceback.format_exc()}"))

    def start_run():
        state["running"] = True
        run_btn.configure(state="disabled")
        prog.start()
        status_lbl.configure(text="실행 중…")
        log("실행 시작")
        threading.Thread(target=worker, args=(build_options(),), daemon=True).start()

    def on_run():
        if state["running"]:
            return
        if not var["activity"].get().strip():
            messagebox.showwarning("입력 필요", "활동데이터 파일을 선택하세요.")
            return
        # 데이터 폴더 바뀌었으면 배너 갱신
        if var["datadir"].get().strip() != state["data_dir"]:
            state["data_dir"] = var["datadir"].get().strip()
            refresh_banner()
        # §5-1 초기화 방지 가드: 🔴/🟡에서 최초 실행 시 모달
        integ = state.get("integrity") or {}
        if integ.get("level") in ("red", "yellow") and var["mode"].get() == "first":
            if not _init_guard_modal(ctk, root, integ, state):
                log("실행 취소(무결성 확인 모달)")
                return
        try:
            opt = build_options()
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("옵션 오류", str(e))
            return
        _ = opt
        start_run()

    run_btn.configure(command=on_run)

    def on_randombox():
        b = state.get("last_borrowers")
        if b is None or len(b) == 0:
            messagebox.showinfo("랜덤박스", "먼저 '실행하기'로 추천을 생성하세요.")
            return
        _open_randombox(ctk, root, state)

    rb_btn.configure(command=on_randombox)

    def finish(res):
        prog.stop(); prog.set(0)
        run_btn.configure(state="normal"); state["running"] = False
        status_lbl.configure(text="완료")
        state["last_borrowers"] = res.borrowers
        log("완료!  보고서: " + os.path.basename(res.report_path))
        log("피드백: " + os.path.basename(res.feedback_path))
        diag_box.delete("1.0", "end")
        diag_box.insert("end", "■ 진단·지표\n")
        for k, v in res.diagnostics.items():
            diag_box.insert("end", f"  {k}: {v}\n")
        if res.warnings:
            diag_box.insert("end", "\n■ 경고\n")
            for w in res.warnings:
                diag_box.insert("end", f"  ! {w}\n")
        refresh_banner()
        tabs.set("  진단·지표  ")
        messagebox.showinfo("완료", f"추천 보고서와 피드백 파일을 생성했습니다.\n\n{res.report_path}")

    def poll():
        try:
            while True:
                kind, payload = log_q.get_nowait()
                if kind == "log":
                    log(payload)
                elif kind == "done":
                    finish(payload)
                elif kind == "uerror":
                    prog.stop(); prog.set(0); run_btn.configure(state="normal")
                    state["running"] = False; status_lbl.configure(text="중단")
                    for ln in str(payload).splitlines():
                        log("안내: " + ln)
                    messagebox.showwarning("실행 중단 안내", str(payload))
                elif kind == "error":
                    prog.stop(); prog.set(0); run_btn.configure(state="normal")
                    state["running"] = False; status_lbl.configure(text="오류")
                    log("오류: " + str(payload).splitlines()[0])
                    messagebox.showerror("실행 오류", str(payload).splitlines()[0])
        except queue.Empty:
            pass
        root.after(150, poll)

    refresh_banner()
    return {"root": root, "tabs": tabs, "poll": poll, "var": var, "state": state,
            "refresh_banner": refresh_banner, "on_run": on_run, "on_randombox": on_randombox}


# ---------------------------------------------------------------------------
def _init_guard_modal(ctk, root, integ, state) -> bool:
    """🔴/🟡 상태에서 최초 실행 시 확인 모달. 반환 True=진행, False=취소."""
    from tkinter import filedialog
    win = ctk.CTkToplevel(root)
    win.title("학습기억 확인")
    win.geometry("560x420")
    win.transient(root)
    win.grab_set()
    result = {"go": False}

    ctk.CTkLabel(win, text="⚠ 학습기억 상태 확인", font=("Malgun Gothic", 17, "bold"),
                 text_color=WELCOME_RED).pack(anchor="w", padx=20, pady=(18, 6))
    for m in integ.get("messages", []):
        ctk.CTkLabel(win, text="· " + m, font=("Malgun Gothic", 12), wraplength=520,
                     justify="left", anchor="w").pack(anchor="w", padx=24, pady=2)

    st = integ.get("state") or {}
    info = (f"마지막 알려진 상태 — 최초학습 {st.get('first_train_date') or '-'}, "
            f"반영 피드백 {st.get('feedback_months', 0)}개월, "
            f"누적표본 {st.get('cum_samples', 0)}")
    ctk.CTkLabel(win, text=info, font=("Malgun Gothic", 11), text_color="gray",
                 wraplength=520, justify="left").pack(anchor="w", padx=24, pady=(8, 10))

    agree = ctk.BooleanVar(value=False)
    ctk.CTkCheckBox(win, text="기억이 초기화됨을 이해합니다(새로 시작).",
                    variable=agree, font=("Malgun Gothic", 12)).pack(anchor="w", padx=24, pady=6)

    btns = ctk.CTkFrame(win, fg_color="transparent")
    btns.pack(side="bottom", fill="x", padx=20, pady=16)

    def recover():
        from corona_reco import store as store_mod
        p = filedialog.askopenfilename(title="복구할 DB/백업 파일 선택",
                                       filetypes=[("SQLite", "*.db *.sqlite"), ("모든 파일", "*.*")])
        if not p:
            return
        try:
            import shutil
            dst = os.path.join(state["data_dir"], config.SQLITE_FILENAME)
            os.makedirs(state["data_dir"], exist_ok=True)
            shutil.copy2(p, dst)
            s = store_mod.Store(dst); sid = (s.get_store_state() or {}).get("store_id"); s.close()
            from tkinter import messagebox
            messagebox.showinfo("복구", f"DB를 복구했습니다. store_id {str(sid)[:8]}")
            win.destroy()
        except Exception as e:  # noqa: BLE001
            from tkinter import messagebox
            messagebox.showerror("복구 실패", str(e))

    def new_start():
        from tkinter import messagebox
        if not agree.get():
            messagebox.showwarning("확인 필요", "'기억이 초기화됨을 이해합니다'에 체크해야 새로 시작할 수 있습니다.")
            return
        result["go"] = True
        win.destroy()

    def cancel():
        win.destroy()

    ctk.CTkButton(btns, text="기억 복구 시도", command=recover,
                  fg_color=WELCOME_ORANGE, hover_color=WELCOME_ORANGE_DK).pack(side="left")
    ctk.CTkButton(btns, text="새로 시작", command=new_start,
                  fg_color=WELCOME_RED, hover_color=WELCOME_RED_DK).pack(side="left", padx=8)
    ctk.CTkButton(btns, text="취소", command=cancel, fg_color="gray").pack(side="right")

    root.wait_window(win)
    return result["go"]


def _open_randombox(ctk, root, state):
    """§3 랜덤박스 창: 합산 1,000만↑ 상위20 중 무작위 5, 재추첨 버튼."""
    from corona_reco import grading
    import random
    win = state.get("rb_win")
    if win is not None and win.winfo_exists():
        win.lift()
    else:
        win = ctk.CTkToplevel(root)
        win.title("🎲 랜덤박스")
        win.geometry("460x420")
        win.transient(root)
        state["rb_win"] = win
    for w in win.winfo_children():
        w.destroy()
    ctk.CTkLabel(win, text="🎲 랜덤박스", font=("Malgun Gothic", 18, "bold"),
                 text_color=WELCOME_ORANGE).pack(anchor="w", padx=18, pady=(14, 2))
    ctk.CTkLabel(win, text=f"차주합산 원금잔액 {config.RANDOMBOX_MIN_MULTI_PRINCIPAL // 10000:,}만원 이상 "
                           f"상위 {config.RANDOMBOX_TOP_POOL}명 중 무작위 {config.RANDOMBOX_PICK}명",
                 font=("Malgun Gothic", 11), text_color="gray").pack(anchor="w", padx=18)

    box = ctk.CTkFrame(win)
    box.pack(fill="both", expand=True, padx=16, pady=10)

    def roll():
        for w in box.winfo_children():
            w.destroy()
        pick = grading.randombox_pick(state["last_borrowers"], seed=random.randint(1, 10**9))
        hdr = ctk.CTkFrame(box, fg_color="transparent"); hdr.pack(fill="x", padx=8, pady=(8, 2))
        for t, wd in [("고객번호", 120), ("부담당자", 100), ("합산 원금잔액", 140)]:
            ctk.CTkLabel(hdr, text=t, width=wd, anchor="w",
                         font=("Malgun Gothic", 12, "bold")).pack(side="left")
        if len(pick) == 0:
            ctk.CTkLabel(box, text="대상 풀이 없습니다(1,000만↑ 차주 없음).",
                         font=("Malgun Gothic", 12)).pack(padx=8, pady=10)
            return
        for _, r in pick.iterrows():
            row = ctk.CTkFrame(box, fg_color="transparent"); row.pack(fill="x", padx=8, pady=2)
            ctk.CTkLabel(row, text=str(r["고객번호"]), width=120, anchor="w",
                         font=("Malgun Gothic", 12)).pack(side="left")
            ctk.CTkLabel(row, text=str(r["부담당자"]), width=100, anchor="w",
                         font=("Malgun Gothic", 12)).pack(side="left")
            ctk.CTkLabel(row, text=f"{int(r['동일차주계좌합산원금잔액']):,} 원", width=140, anchor="w",
                         font=("Malgun Gothic", 12)).pack(side="left")

    ctk.CTkButton(win, text="다시 뽑기", command=roll, fg_color=WELCOME_ORANGE,
                  hover_color=WELCOME_ORANGE_DK, font=("Malgun Gothic", 13, "bold")).pack(pady=(0, 14))
    roll()
    win.grab_set()


# ---------------------------------------------------------------------------
def _run_gui(appearance: str = "light"):
    ctx = build_app(appearance=appearance)
    ctx["poll"]()
    ctx["root"].mainloop()


def main():
    try:
        _run_gui()
    except Exception:  # noqa: BLE001
        # CustomTkinter/tkinter 미탑재 등 → 최소 안내
        try:
            import tkinter as tk
            from tkinter import messagebox
            r = tk.Tk(); r.withdraw()
            messagebox.showerror("실행 오류",
                                 "GUI를 초기화할 수 없습니다.\n" + traceback.format_exc().splitlines()[-1])
        except Exception:
            print("GUI 초기화 실패:\n" + traceback.format_exc())


if __name__ == "__main__":
    main()
