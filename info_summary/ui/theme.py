from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont, ttk


COLORS = {
    "page": "#F6F5F1",
    "card": "#FFFFFF",
    "text": "#243C35",
    "muted": "#778079",
    "subtle": "#A3ABA5",
    "line": "#E3E6DF",
    "blue": "#28634F",
    "blue_hover": "#1C4E3D",
    "blue_soft": "#EDF3EC",
    "navy": "#F6F5F1",
    "navy_soft": "#E8EDE5",
    "success": "#28634F",
    "success_soft": "#EDF3EC",
    "warning": "#B54708",
    "log": "#F6F7F3",
    "off": "#A3ABA5",
}


def configure_theme(root: tk.Tk) -> None:
    root.configure(background=COLORS["page"])
    root.option_add("*tearOff", False)
    base = tkfont.nametofont("TkDefaultFont")
    base.configure(family="Helvetica Neue", size=13)
    heading = base.copy()
    heading.configure(size=20, weight="bold")
    section = base.copy()
    section.configure(size=14, weight="bold")
    small = base.copy()
    small.configure(size=11)

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except tk.TclError:
        pass
    # Keep named fonts alive for the lifetime of the window.
    root._theme_fonts = (base, heading, section, small)
    style.configure("Page.TFrame", background=COLORS["page"])
    style.configure("Hero.TFrame", background=COLORS["navy"])
    style.configure("Card.TFrame", background=COLORS["card"], relief="flat", borderwidth=0)
    style.configure(
        "Panel.TFrame",
        background=COLORS["card"],
        relief="flat",
        borderwidth=1,
        lightcolor=COLORS["line"],
        darkcolor=COLORS["line"],
    )
    style.configure("Header.TLabel", background=COLORS["navy"], foreground=COLORS["text"], font=heading)
    style.configure("Subtitle.TLabel", background=COLORS["navy"], foreground="#778079")
    style.configure("CardTitle.TLabel", background=COLORS["card"], foreground=COLORS["text"], font=section)
    style.configure("CardText.TLabel", background=COLORS["card"], foreground=COLORS["muted"])
    style.configure("Field.TLabel", background=COLORS["card"], foreground=COLORS["text"], font=small)
    style.configure("Hint.TLabel", background=COLORS["card"], foreground=COLORS["muted"], font=small)
    style.configure("Preview.TLabel", background=COLORS["card"], foreground=COLORS["blue"], font=small)
    style.configure("Status.TLabel", background=COLORS["navy_soft"], foreground=COLORS["blue"], padding=(12, 7), font=small)
    style.configure("TButton", padding=(12, 8), borderwidth=0)
    style.configure("Secondary.TButton", background="#F0F2EC", foreground=COLORS["text"], padding=(13, 9), font=small)
    style.map("Secondary.TButton", background=[("active", "#E3E8DE")])
    style.configure("Compact.TButton", background="#F0F2EC", foreground=COLORS["text"], padding=(10, 7), font=small)
    style.map("Compact.TButton", background=[("active", "#E3E8DE")], foreground=[("disabled", COLORS["subtle"])])
    style.configure("Selected.TButton", background=COLORS["blue"], foreground="#FFFFFF", padding=(13, 9), font=small)
    style.map("Selected.TButton", background=[("disabled", COLORS["subtle"]), ("active", COLORS["blue_hover"])], foreground=[("disabled", "#FFFFFF")])
    style.configure("Primary.TButton", background=COLORS["blue"], foreground="#FFFFFF", padding=(24, 12), font=section)
    style.map("Primary.TButton", background=[("disabled", "#A3ABA5"), ("active", COLORS["blue_hover"])], foreground=[("disabled", "#FFFFFF")])
    style.configure("Soft.TButton", background=COLORS["blue_soft"], foreground=COLORS["blue"], padding=(14, 8), font=small)
    style.map("Soft.TButton", background=[("active", "#DFEADC")])
    style.configure("TEntry", padding=10, fieldbackground="#FFFFFF", bordercolor=COLORS["line"], lightcolor=COLORS["line"], darkcolor=COLORS["line"])
    style.map("TEntry", bordercolor=[("focus", COLORS["blue"])], lightcolor=[("focus", COLORS["blue"])], darkcolor=[("focus", COLORS["blue"])])
    style.configure("TCombobox", padding=9, fieldbackground="#FFFFFF")
    style.configure("Blue.Horizontal.TProgressbar", background=COLORS["blue"], troughcolor="#E4EAE1", thickness=4)
    style.configure("Tasks.Treeview", background=COLORS["card"], fieldbackground=COLORS["card"],
                    foreground=COLORS["text"], borderwidth=0, rowheight=29, font=small)
    style.configure("Tasks.Treeview.Heading", background=COLORS["log"], foreground=COLORS["muted"],
                    relief="flat", padding=(8, 5), font=small)
    style.map("Tasks.Treeview", background=[("selected", COLORS["blue_soft"])],
              foreground=[("selected", COLORS["text"])])
