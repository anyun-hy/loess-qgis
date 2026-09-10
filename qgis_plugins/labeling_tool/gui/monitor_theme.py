"""Monitor-local visual tokens; no QGIS application-wide palette changes."""

from pathlib import Path

TABLE_CELL_PADDING = 10
OVERVIEW_ICON_SIZE = 22
BODY_FONT_PT = 11
SECTION_FONT_PT = 13
HEADLINE_FONT_PT = 20
METRIC_FONT_PT = 30
VALUE_FONT_PT = 16
DETAIL_LINE_HEIGHT = 128
TABLE_ROW_MIN_HEIGHT = 36
TABLE_HEADER_MIN_HEIGHT = 32
# A row's text starts after its padding, icon slot, and the native item gap.
OVERVIEW_HEADER_TEXT_INSET = 42

PALETTES = {
    "dark": dict(background="#142D43", panel="#1C3851", inset="#22445F",
                 field="#19344B", border="#365B76", text="#EDF6FF", muted="#A9C3D8",
                 accent="#40C8EF", accent_end="#71DCFA", track="#345B77",
                 selected="#244D68", hover="#2D536D", panel_end="#18334A",
                 success="#63DFB2", warning="#F8C266", failed="#FF7D8C"),
    "light": dict(background="#EDF2F6", panel="#FFFFFF", inset="#F3F7FA",
                  field="#FFFFFF", border="#CBD7E0", text="#253B4D", muted="#566D80",
                  accent="#148BAF", accent_end="#53B9D5", track="#D9E6EE",
                  selected="#DCECF3", hover="#E9F2F7", panel_end="#F7FAFC",
                  success="#24735B", warning="#876018", failed="#AE3948"),
}


def _stylesheet(p):
    arrow = (Path(__file__).parent / "icons" / (
        "chevron-dark.svg" if p is PALETTES["dark"] else "chevron-light.svg"
    )).as_posix()
    return f"""
QDialog#InferenceMonitor {{ background: {p['background']}; color: {p['text']}; }}
QWidget {{ font-size: {BODY_FONT_PT}pt; }}
QWidget#MonitorPage, QScrollArea#MonitorScroll, QScrollArea#MonitorScroll > QWidget > QWidget {{
  background: {p['background']}; color: {p['text']}; border: 0;
}}
QFrame[monitorPanel="true"] {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 {p['panel']},stop:1 {p['panel_end']}); border: 1px solid {p['border']}; border-radius: 10px; }}
QFrame[monitorSubPanel="true"] {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 {p['inset']},stop:1 {p['panel']}); border: 1px solid {p['border']}; border-radius: 8px; }}
QLabel {{ color: {p['text']}; background: transparent; border: 0; }}
QLabel[muted="true"] {{ color: {p['muted']}; }}
QLabel[hero="true"] {{ font-size: 18pt; font-weight: 600; }}
QLabel[metric="true"] {{ font-size: {METRIC_FONT_PT}pt; font-weight: 600; }}
QLabel[sectionTitle="true"] {{ font-size: {SECTION_FONT_PT}pt; font-weight: 600; }}
QLabel[headline="true"] {{ font-size: {HEADLINE_FONT_PT}pt; font-weight: 600; }}
QLabel[value="true"] {{ font-size: {VALUE_FONT_PT}pt; font-weight: 600; }}
QLabel[accent="true"] {{ color: {p['accent']}; }}
QFrame#MonitorDivider {{ background: {p['border']}; border: 0; }}
QLabel[status] {{ background: {p['inset']}; border: 1px solid {p['border']}; border-radius: 6px; padding: 6px 10px; font-weight: 600; }}
QLabel[status="active"] {{ color: {p['success']}; }}
QLabel[status="warning"] {{ color: {p['warning']}; }}
QLabel[status="failed"] {{ color: {p['failed']}; }}
QLabel[status="neutral"] {{ color: {p['muted']}; }}
QPushButton {{ color: {p['text']}; background: {p['inset']}; border: 1px solid {p['border']}; border-radius: 7px; padding: 7px 16px; min-height: 20px; }}
QPushButton[quiet="true"] {{ background: transparent; border-color: transparent; color: {p['muted']}; }}
QPushButton#ThemeToggle {{ padding: 8px; min-width: 20px; min-height: 20px; }}
QPushButton[link="true"] {{ background: transparent; border-color: transparent; color: {p['accent']}; padding: 5px 0; }}
QPushButton:focus, QLineEdit:focus, QComboBox:focus {{ border: 1px solid {p['accent']}; }}
QPushButton:hover {{ background: {p['hover']}; border-color: {p['accent']}; }}
QPushButton:pressed {{ background: {p['selected']}; }}
QPushButton:disabled {{ color: {p['muted']}; background: {p['panel']}; }}
QPushButton#StopButton {{ color: {p['failed']}; border-color: {p['failed']}; font-size: 13pt; font-weight: 600; padding: 12px 22px; }}
QTabWidget::pane {{ border: 0; }}
QTabBar::base {{ border: 0; background: transparent; }}
QTabBar::tab {{ color: {p['muted']}; background: transparent; border: 0; border-bottom: 2px solid transparent; padding: 10px 24px; min-width: 85px; }}
QTabBar::tab:selected {{ color: {p['text']}; background: {p['panel']}; border-bottom: 2px solid {p['accent']}; }}
QTabBar::tab:hover {{ color: {p['text']}; background: {p['hover']}; }}
QTableWidget, QListWidget, QTextBrowser, QLineEdit, QComboBox, QPlainTextEdit {{
  color: {p['text']}; background: {p['field']}; alternate-background-color: {p['inset']};
  border: 1px solid {p['border']}; border-radius: 5px; selection-background-color: {p['selected']}; selection-color: {p['text']};
}}
QLineEdit, QComboBox {{ padding: 7px 8px; min-height: 20px; }}
QComboBox {{ padding-right: 30px; border-radius: 7px; }}
QComboBox:hover, QComboBox:on {{ border-color: {p['accent']}; }}
QComboBox:disabled {{ color: {p['muted']}; background: {p['panel']}; }}
QComboBox::drop-down {{ subcontrol-origin: padding; subcontrol-position: top right;
  width: 28px; border: 0; background: transparent;
  border-top-right-radius: 7px; border-bottom-right-radius: 7px;
}}
QComboBox::down-arrow {{ image: url("{arrow}"); width: 12px; height: 8px; }}
QListWidget::item {{ padding: 8px; }}
QTableWidget::item {{ padding: 7px {TABLE_CELL_PADDING}px; border-bottom: 1px solid {p['border']}; }}
QHeaderView {{ background: {p['inset']}; border: 0; }}
QHeaderView::section, QTableCornerButton::section {{ color: {p['muted']}; background: {p['inset']}; border: 0; padding: 4px {TABLE_CELL_PADDING}px; font-weight: 600; }}
/* Overview headers begin at the same x-coordinate as text after each 22px
   cell icon, rather than at the left edge of the icon decoration slot. */
QTableWidget#OverviewResults QHeaderView::section {{ padding-left: {OVERVIEW_HEADER_TEXT_INSET}px; text-align: left; }}
QProgressBar {{ color: {p['text']}; background: {p['border']}; border: 0; border-radius: 3px; min-height: 6px; text-align: center; }}
QProgressBar::chunk {{ background: {p['accent']}; border-radius: 3px; }}
QSplitter::handle {{ background: {p['background']}; width: 10px; height: 10px; }}
QScrollBar:vertical {{ background: {p['background']}; width: 9px; margin: 0; }}
QScrollBar:horizontal {{ background: {p['background']}; height: 9px; margin: 0; }}
QScrollBar::handle {{ background: {p['border']}; border-radius: 4px; min-height: 24px; min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
QToolTip {{ color: {p['text']}; background: {p['panel']}; border: 1px solid {p['border']}; padding: 8px; }}
QFrame#MonitorHealth {{ background: {p['field']}; border: 1px solid {p['border']}; border-radius: 8px; }}
QFrame#MonitorHealth[attention="true"] {{ border-color: {p['warning']}; }}
QCheckBox {{ color: {p['muted']}; spacing: 7px; }}
"""


MONITOR_STYLE = {theme: _stylesheet(palette) for theme, palette in PALETTES.items()}


def combo_popup_style(theme):
    """Style the separate native popup explicitly, not the application's menus."""
    p = PALETTES.get(theme, PALETTES["dark"])
    return f"""
QListView {{ background: {p['field']}; color: {p['text']};
  border: 1px solid {p['border']}; border-radius: 7px; padding: 4px;
  outline: 0; font-size: {BODY_FONT_PT}pt;
  selection-background-color: {p['selected']}; selection-color: {p['text']};
}}
QListView::item {{ min-height: 20px; padding: 7px 10px; border: 0; border-radius: 4px; }}
QListView::item:hover {{ background: {p['hover']}; }}
QListView::item:selected {{ background: {p['selected']}; color: {p['text']}; }}
QListView::item:disabled {{ color: {p['muted']}; }}
QScrollBar:vertical {{ background: {p['field']}; width: 9px; }}
QScrollBar::handle:vertical {{ background: {p['border']}; border-radius: 4px; min-height: 24px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: {p['field']}; }}
"""


def status_color(theme, status):
    semantic = {"运行中": "accent", "成功": "success", "失败": "failed", "已停止": "warning"}
    return PALETTES.get(theme, PALETTES["dark"])[semantic.get(status, "muted")]
