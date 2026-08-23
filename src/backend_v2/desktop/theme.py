"""Shared visual system for the native desktop control center."""

from __future__ import annotations


WINDOW_STYLESHEET = """
* {
    font-family: "Segoe UI Variable", "Microsoft YaHei UI";
    font-size: 14px;
    color: #2D292C;
}
QWidget#desktopRoot { background: transparent; }
QFrame#windowShell {
    background: #F7F6F7;
    border: 1px solid #DEDADD;
    border-radius: 14px;
}
QFrame#sidebar {
    background: #FCFBFC;
    border: none;
    border-right: 1px solid #E7E3E6;
    border-top-left-radius: 13px;
    border-bottom-left-radius: 13px;
}
QLabel#brandLogo { background: #F4ECEF; border-radius: 9px; }
QLabel#brandTitle { font-size: 16px; font-weight: 700; color: #272326; }
QLabel#brandSubtitle { font-size: 10px; font-weight: 600; color: #9B7483; }
QPushButton#navButton {
    min-height: 44px;
    text-align: left;
    padding: 0 16px;
    border: none;
    border-radius: 9px;
    background: transparent;
    color: #625C60;
    font-size: 14px;
    font-weight: 600;
}
QPushButton#navButton:hover { background: #F2EFF1; color: #302B2F; }
QPushButton#navButton[active="true"] { background: #F7E4EB; color: #B13E68; }
QFrame#titleBar {
    background: #FCFBFC;
    border: none;
    border-bottom: 1px solid #E7E3E6;
    border-top-right-radius: 13px;
}
QLabel#pageTitle { font-size: 19px; font-weight: 700; color: #292529; }
QPushButton#windowControl {
    min-width: 38px;
    max-width: 38px;
    min-height: 32px;
    max-height: 32px;
    padding: 0;
    border: none;
    border-radius: 7px;
    background: transparent;
    color: #6B6569;
    font-size: 15px;
    font-weight: 500;
}
QPushButton#windowControl:hover { background: #EEECEE; color: #292529; }
QPushButton#windowControl[danger="true"]:hover { background: #D94F62; color: white; }
QWidget#page, QWidget#settingsContent, QWidget#settingRow,
QWidget#actionCell { background: transparent; }
QFrame#card, QFrame#settingsCard, QFrame#toolbarCard {
    background: #FFFFFF;
    border: 1px solid #E4E0E3;
    border-radius: 12px;
}
QLabel#eyebrow { color: #B04B70; font-size: 11px; font-weight: 700; }
QLabel#heroTitle { font-size: 22px; font-weight: 700; color: #272326; }
QLabel#sectionTitle { font-size: 16px; font-weight: 700; color: #302B2F; }
QLabel#muted { color: #756F73; font-size: 13px; }
QLabel#statusValue { font-size: 17px; font-weight: 700; color: #302B2F; }
QLabel#statusPill, QLabel#autoSaveStatus {
    background: #F8E5EC;
    color: #AD3D67;
    border-radius: 8px;
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 700;
}
QLabel#settingsHint { color: #746D72; font-size: 13px; }
QLabel#settingsSectionTitle { font-size: 16px; font-weight: 700; color: #2D292C; }
QLabel#settingsSectionDescription { color: #7D767A; font-size: 12px; }
QLabel#settingTitle { color: #373236; font-size: 14px; font-weight: 600; }
QLabel#settingDescription { color: #858084; font-size: 12px; }
QFrame#settingDivider { background: #ECE9EB; border: none; min-height: 1px; max-height: 1px; }
QPushButton {
    min-height: 36px;
    padding: 0 15px;
    border: 1px solid #DCD7DA;
    border-radius: 8px;
    background: #FFFFFF;
    color: #514B4F;
    font-size: 13px;
    font-weight: 600;
}
QPushButton:hover { background: #F7F4F5; border-color: #C9C3C7; color: #292529; }
QPushButton:pressed { background: #EEE9EB; }
QPushButton:disabled { color: #B7B1B5; background: #F3F2F3; border-color: #E6E3E5; }
QPushButton#primaryButton { background: #D1517F; color: white; border: none; }
QPushButton#primaryButton:hover { background: #BF456F; }
QPushButton#primaryButton:disabled { background: #D8D3D6; color: #F7F5F6; }
QPushButton#dangerButton { color: #C9475A; border-color: #E8C2C9; }
QPushButton#compactButton {
    min-width: 48px;
    min-height: 32px;
    max-height: 32px;
    padding: 0 10px;
    border-radius: 7px;
    font-size: 13px;
}
QTabWidget::pane { border: none; background: transparent; }
QTabWidget#taskTabs::pane { top: 8px; }
QTabBar::tab {
    background: transparent;
    color: #716A6F;
    border: none;
    border-radius: 8px;
    padding: 9px 15px;
    margin-right: 4px;
    font-size: 14px;
    font-weight: 600;
}
QTabBar::tab:hover { background: #EEECEE; }
QTabBar::tab:selected { background: #F8E5EC; color: #AD3D67; }
QTableWidget {
    background: #FFFFFF;
    alternate-background-color: #FBFAFB;
    border: 1px solid #E3DFE2;
    border-radius: 10px;
    gridline-color: transparent;
    selection-background-color: #F9E8EE;
    selection-color: #302B2F;
    font-size: 13px;
}
QTableWidget::item { padding: 8px; border-bottom: 1px solid #F0EDF0; }
QHeaderView::section {
    background: #F7F5F6;
    color: #6F686D;
    border: none;
    border-bottom: 1px solid #E3DFE2;
    padding: 10px 8px;
    font-size: 12px;
    font-weight: 700;
}
QProgressBar {
    min-height: 8px;
    max-height: 8px;
    border: none;
    border-radius: 4px;
    background: #ECE9EB;
    text-align: center;
    color: transparent;
}
QProgressBar::chunk { background: #D65F89; border-radius: 4px; }
QPlainTextEdit {
    background: #252326;
    color: #EEECEE;
    border: 1px solid #353236;
    border-radius: 10px;
    padding: 14px;
    font-size: 12px;
    selection-background-color: #8E4561;
}
QLineEdit, QSpinBox, QComboBox {
    min-height: 36px;
    padding: 0 11px;
    background: #FFFFFF;
    border: 1px solid #DCD8DB;
    border-radius: 8px;
    font-size: 13px;
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
QComboBox::drop-down { border: none; width: 30px; }
QComboBox QAbstractItemView {
    background: white;
    border: 1px solid #DDD9DC;
    selection-background-color: #F8E5EC;
    font-size: 13px;
    outline: none;
}
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: transparent; width: 8px; margin: 3px 0; }
QScrollBar::handle:vertical { background: #C9C3C7; border-radius: 4px; min-height: 32px; }
QScrollBar::handle:vertical:hover { background: #B3ACB0; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar:horizontal { background: transparent; height: 8px; margin: 0 3px; }
QScrollBar::handle:horizontal { background: #777176; border-radius: 4px; min-width: 32px; }
QScrollBar::handle:horizontal:hover { background: #8A8388; }
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: transparent; }
QMenu#petMenu, QMenu {
    background: #FFFFFF;
    border: 1px solid #DDD9DC;
    border-radius: 9px;
    padding: 5px;
    font-size: 13px;
}
QMenu::item { padding: 8px 22px; border-radius: 6px; }
QMenu::item:selected { background: #F8E5EC; color: #AD3D67; }
QToolTip {
    background: #302C2F;
    color: white;
    border: none;
    padding: 6px 9px;
    font-size: 12px;
}
QMessageBox QLabel { font-size: 13px; }
QMessageBox QLabel#qt_msgbox_label { min-width: 280px; }
QMessageBox QPushButton { min-width: 84px; }
"""
