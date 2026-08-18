"""Neutral pink-accented desktop theme for the native control center."""

from __future__ import annotations


WINDOW_STYLESHEET = """
* {
    font-family: "Segoe UI Variable", "Microsoft YaHei UI";
    font-size: 13px;
    color: #292629;
}
QWidget#desktopRoot { background: transparent; }
QFrame#windowShell {
    background: #F6F5F6;
    border: 1px solid #DDDADD;
    border-radius: 14px;
}
QFrame#sidebar {
    background: #FCFBFC;
    border: none;
    border-right: 1px solid #E8E5E7;
    border-top-left-radius: 13px;
    border-bottom-left-radius: 13px;
}
QLabel#brandLogo { background: #F4ECEF; border-radius: 9px; }
QLabel#brandTitle { font-size: 16px; font-weight: 700; color: #262225; }
QLabel#brandSubtitle { font-size: 9px; font-weight: 600; color: #A27C8B; }
QPushButton#navButton {
    min-height: 42px;
    text-align: left;
    padding: 0 14px;
    border: none;
    border-radius: 9px;
    background: transparent;
    color: #686166;
    font-weight: 600;
}
QPushButton#navButton:hover { background: #F4F1F2; color: #312C30; }
QPushButton#navButton[active="true"] { background: #F8E5EC; color: #B43E6A; }
QFrame#titleBar {
    background: #FCFBFC;
    border: none;
    border-bottom: 1px solid #E8E5E7;
    border-top-right-radius: 13px;
}
QLabel#pageTitle { font-size: 17px; font-weight: 700; color: #292529; }
QPushButton#windowControl {
    min-width: 36px;
    max-width: 36px;
    min-height: 30px;
    max-height: 30px;
    padding: 0;
    border: none;
    border-radius: 7px;
    background: transparent;
    color: #6F686D;
    font-weight: 500;
}
QPushButton#windowControl:hover { background: #EEECEE; color: #292529; }
QPushButton#windowControl[danger="true"]:hover { background: #D94F62; color: white; }
QWidget#page, QWidget#settingsContent, QWidget#settingRow { background: transparent; }
QFrame#card, QFrame#settingsCard {
    background: #FFFFFF;
    border: 1px solid #E5E2E4;
    border-radius: 12px;
}
QLabel#eyebrow { color: #B45477; font-size: 10px; font-weight: 700; }
QLabel#heroTitle { font-size: 23px; font-weight: 700; color: #252125; }
QLabel#sectionTitle { font-size: 15px; font-weight: 700; color: #302B2F; }
QLabel#muted { color: #787176; }
QLabel#statusValue { font-size: 16px; font-weight: 700; color: #302B2F; }
QLabel#statusPill, QLabel#autoSaveStatus {
    background: #F8E6ED;
    color: #AD3D67;
    border-radius: 8px;
    padding: 4px 9px;
    font-size: 10px;
    font-weight: 700;
}
QLabel#settingsHint { color: #746D72; }
QLabel#settingsSectionTitle { font-size: 15px; font-weight: 700; color: #2D292C; }
QLabel#settingsSectionDescription { color: #817A7E; font-size: 11px; }
QLabel#settingTitle { color: #373236; font-weight: 600; }
QLabel#settingDescription { color: #8A8387; font-size: 11px; }
QFrame#settingDivider { background: #EEECEE; border: none; min-height: 1px; max-height: 1px; }
QPushButton {
    min-height: 34px;
    padding: 0 14px;
    border: 1px solid #DDD9DC;
    border-radius: 8px;
    background: #FFFFFF;
    color: #554E53;
    font-weight: 600;
}
QPushButton:hover { background: #F7F4F5; border-color: #CFC9CD; color: #292529; }
QPushButton:pressed { background: #EEE9EB; }
QPushButton:disabled { color: #B9B4B7; background: #F5F4F5; border-color: #E8E5E7; }
QPushButton#primaryButton { background: #D1517F; color: white; border: none; }
QPushButton#primaryButton:hover { background: #BF456F; }
QPushButton#primaryButton:disabled { background: #D8D3D6; color: #F7F5F6; }
QPushButton#dangerButton { color: #C9475A; border-color: #E9C5CB; }
QPushButton#compactButton { min-height: 28px; max-height: 28px; padding: 0 9px; border-radius: 7px; }
QTabWidget::pane { border: none; background: transparent; top: -1px; }
QTabBar::tab {
    background: transparent;
    color: #756E73;
    border: none;
    border-radius: 8px;
    padding: 8px 15px;
    margin-right: 4px;
    font-weight: 600;
}
QTabBar::tab:hover { background: #EEECEE; }
QTabBar::tab:selected { background: #F8E5EC; color: #AD3D67; }
QTableWidget {
    background: #FFFFFF;
    alternate-background-color: #FAF9FA;
    border: 1px solid #E4E1E3;
    border-radius: 10px;
    gridline-color: transparent;
    selection-background-color: #F9E8EE;
    selection-color: #302B2F;
}
QTableWidget::item { padding: 7px; border-bottom: 1px solid #F0EEF0; }
QHeaderView::section {
    background: #F7F6F7;
    color: #716A6F;
    border: none;
    border-bottom: 1px solid #E4E1E3;
    padding: 9px;
    font-size: 10px;
    font-weight: 700;
}
QProgressBar {
    min-height: 7px;
    max-height: 7px;
    border: none;
    border-radius: 3px;
    background: #ECE9EB;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk { background: #DC6A92; border-radius: 3px; }
QPlainTextEdit {
    background: #252326;
    color: #ECEAEC;
    border: 1px solid #353236;
    border-radius: 10px;
    padding: 12px;
    font-size: 11px;
    selection-background-color: #8E4561;
}
QLineEdit, QSpinBox, QComboBox {
    min-height: 34px;
    padding: 0 10px;
    background: #FFFFFF;
    border: 1px solid #DCD8DB;
    border-radius: 8px;
    selection-background-color: #EBA9BF;
}
QLineEdit:hover, QSpinBox:hover, QComboBox:hover { border-color: #C8C2C6; }
QLineEdit:focus, QSpinBox:focus, QComboBox:focus { border: 1px solid #D45A84; }
QLineEdit:read-only { background: #F5F4F5; color: #70696E; }
QLineEdit:disabled, QSpinBox:disabled, QComboBox:disabled {
    background: #F2F1F2;
    color: #AAA4A8;
    border-color: #E6E3E5;
}
QComboBox::drop-down { border: none; width: 28px; }
QComboBox QAbstractItemView { background: white; border: 1px solid #DDD9DC; selection-background-color: #F8E5EC; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 7px; margin: 3px 0; }
QScrollBar::handle:vertical { background: #CEC8CC; border-radius: 3px; min-height: 30px; }
QScrollBar::handle:vertical:hover { background: #B8B1B5; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QMenu#petMenu, QMenu {
    background: #FFFFFF;
    border: 1px solid #DDD9DC;
    border-radius: 9px;
    padding: 5px;
}
QMenu::item { padding: 7px 20px; border-radius: 6px; }
QMenu::item:selected { background: #F8E5EC; color: #AD3D67; }
QToolTip { background: #302C2F; color: white; border: none; padding: 5px 8px; }
"""
