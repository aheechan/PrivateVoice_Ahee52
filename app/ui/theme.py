"""深色渐变 + 紫色主题。只做视觉，不改任何行为。"""
from __future__ import annotations

PURPLE = "#b388ff"
PURPLE_DIM = "#8c6bd8"
TEXT = "#d9cdf5"
MUTED = "#8f83ad"
BG_TOP = "#0b0d16"
BG_BOTTOM = "#161226"

THEME_QSS = f"""
QWidget {{
    background-color: transparent;
    color: {TEXT};
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI";
    font-size: 13px;
}}
QMainWindow, QDialog, QWizard, QWidget#Root {{
    background: qlineargradient(x1:0, y1:0, x2:0.6, y2:1,
                stop:0 {BG_TOP}, stop:0.55 #121022, stop:1 {BG_BOTTOM});
}}
QMenu {{
    background: #14111f;
    color: {TEXT};
    border: 1px solid #2c2545;
    border-radius: 8px;
    padding: 6px;
}}
QMenu::item {{ padding: 7px 22px; border-radius: 6px; }}
QMenu::item:selected {{ background: #2a2140; color: #ffffff; }}
QMenu::separator {{ height: 1px; background: #2c2545; margin: 5px 8px; }}

QLabel {{ background: transparent; }}
QLabel#Title {{ font-size: 20px; font-weight: 600; color: {PURPLE}; }}
QLabel#Muted {{ color: {MUTED}; }}
QLabel#Status {{ color: {PURPLE}; }}

QPushButton {{
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #241b3d, stop:1 #1a1430);
    color: {TEXT};
    border: 1px solid #3a2e5c;
    border-radius: 9px;
    padding: 7px 16px;
}}
QPushButton:hover {{
    border: 1px solid {PURPLE_DIM};
    color: #ffffff;
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #2e2350, stop:1 #221a3c);
}}
QPushButton:pressed {{ background: #1a1430; border-color: {PURPLE}; }}
QPushButton:disabled {{ color: #5b5470; border-color: #2a2440; background: #14111f; }}
QPushButton#Primary {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #5b3fa8, stop:1 #7c53d6);
    border: 1px solid #9a76ff;
    color: #ffffff;
    font-size: 20px;
    font-weight: 600;
    border-radius: 16px;
}}
QPushButton#Primary:hover {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #6a4bc0, stop:1 #8d63e8);
}}
QPushButton#Primary:pressed {{ background: #e5534b; border-color: #ff8a80; }}
QPushButton#Primary:disabled {{ background: #241b3d; color: #6b6383; border-color: #33294f; }}

QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: #120f1c;
    color: {TEXT};
    border: 1px solid #322a4d;
    border-radius: 8px;
    padding: 6px 10px;
    selection-background-color: #4a3878;
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus {{
    border: 1px solid {PURPLE_DIM};
}}
QComboBox QAbstractItemView {{
    background: #14111f;
    color: {TEXT};
    border: 1px solid #2c2545;
    selection-background-color: #2a2140;
}}
QTableWidget, QTableView {{
    background: #100d1a;
    alternate-background-color: #141020;
    gridline-color: #241d38;
    border: 1px solid #2a2340;
    border-radius: 10px;
    selection-background-color: #2f2550;
    selection-color: #ffffff;
}}
QHeaderView::section {{
    background: #1a1530;
    color: {PURPLE};
    border: none;
    border-bottom: 1px solid #322a4d;
    padding: 7px;
}}
QProgressBar {{
    background: #14111f;
    border: 1px solid #2c2545;
    border-radius: 8px;
    text-align: center;
    color: {TEXT};
}}
QProgressBar::chunk {{
    border-radius: 7px;
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #5b3fa8, stop:1 #9a76ff);
}}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #3a2e5c; border-radius: 5px; min-height: 26px; }}
QScrollBar::handle:vertical:hover {{ background: {PURPLE_DIM}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; }}
QGroupBox {{
    border: 1px solid #2c2545;
    border-radius: 10px;
    margin-top: 14px;
    padding-top: 8px;
    color: {PURPLE};
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 12px; padding: 0 6px; }}
QCheckBox, QRadioButton {{ color: {TEXT}; spacing: 8px; }}
QToolTip {{
    background: #1a1530; color: {TEXT};
    border: 1px solid {PURPLE_DIM}; border-radius: 6px; padding: 5px;
}}
QWizard QPushButton {{ min-width: 88px; }}
QSplitter::handle {{ background: #241d38; }}
"""


def apply(app) -> None:
    app.setStyleSheet(THEME_QSS)


__all__ = ["THEME_QSS", "apply", "PURPLE", "TEXT", "MUTED"]
