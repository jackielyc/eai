"""ui.theme — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def _combo_down_arrow_path() -> str:
    """Light chevron for dark-theme QComboBox (Fusion default arrow is too dim)."""
    path = _eai_platform_bootstrap.os.path.join(_eai_configuration_models.EAI_DIR, ".cache", "ui", "combo_down_arrow.svg")
    try:
        _eai_platform_bootstrap.os.makedirs(_eai_platform_bootstrap.os.path.dirname(path), exist_ok=True)
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="12" height="8" '
            'viewBox="0 0 12 8">'
            f'<path fill="{_eai_configuration_policies.UI_TEXT_SECONDARY}" d="M1 1.2h10L6 7.2z"/>'
            "</svg>\n"
        )
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(svg)
    except OSError:
        pass
    return path.replace("\\", "/")


def apply_viewer_theme(app: _eai_platform_bootstrap.QApplication) -> None:
    """统一深色主题：表面层级、焦点环、标签页强调色、滚动条。"""
    palette = _eai_platform_bootstrap.QPalette()
    palette.setColor(_eai_platform_bootstrap.QPalette.Window, _eai_platform_bootstrap.QColor(_eai_configuration_policies.UI_BG_WINDOW))
    palette.setColor(_eai_platform_bootstrap.QPalette.WindowText, _eai_platform_bootstrap.QColor(_eai_configuration_policies.UI_TEXT_PRIMARY))
    palette.setColor(_eai_platform_bootstrap.QPalette.Base, _eai_platform_bootstrap.QColor(_eai_configuration_policies.UI_BG_INPUT))
    palette.setColor(_eai_platform_bootstrap.QPalette.AlternateBase, _eai_platform_bootstrap.QColor(_eai_configuration_policies.UI_BG_ELEVATED))
    palette.setColor(_eai_platform_bootstrap.QPalette.Text, _eai_platform_bootstrap.QColor(_eai_configuration_policies.UI_TEXT_PRIMARY))
    palette.setColor(_eai_platform_bootstrap.QPalette.Button, _eai_platform_bootstrap.QColor(_eai_configuration_policies.UI_BG_ELEVATED))
    palette.setColor(_eai_platform_bootstrap.QPalette.ButtonText, _eai_platform_bootstrap.QColor(_eai_configuration_policies.UI_TEXT_PRIMARY))
    palette.setColor(_eai_platform_bootstrap.QPalette.Highlight, _eai_platform_bootstrap.QColor(_eai_configuration_policies.UI_HIGHLIGHT))
    palette.setColor(_eai_platform_bootstrap.QPalette.HighlightedText, _eai_platform_bootstrap.QColor("#ffffff"))
    palette.setColor(_eai_platform_bootstrap.QPalette.ToolTipBase, _eai_platform_bootstrap.QColor(_eai_configuration_policies.UI_BG_PANEL))
    palette.setColor(_eai_platform_bootstrap.QPalette.ToolTipText, _eai_platform_bootstrap.QColor(_eai_configuration_policies.UI_TEXT_PRIMARY))
    palette.setColor(_eai_platform_bootstrap.QPalette.PlaceholderText, _eai_platform_bootstrap.QColor(_eai_configuration_policies.UI_TEXT_PLACEHOLDER))
    palette.setColor(_eai_platform_bootstrap.QPalette.Light, _eai_platform_bootstrap.QColor(_eai_configuration_policies.UI_BORDER_STRONG))
    palette.setColor(_eai_platform_bootstrap.QPalette.Mid, _eai_platform_bootstrap.QColor(_eai_configuration_policies.UI_BORDER))
    palette.setColor(_eai_platform_bootstrap.QPalette.Dark, _eai_platform_bootstrap.QColor(_eai_configuration_policies.UI_BG_INPUT))
    app.setPalette(palette)
    combo_arrow = _combo_down_arrow_path()
    app.setStyleSheet(
        f"""
        QWidget {{
            font-size: {_eai_configuration_policies.UI_MONO_SIZE_NORMAL}pt;
            color: {_eai_configuration_policies.UI_TEXT_PRIMARY};
        }}
        QMainWindow, QWidget#viewerCentral {{
            background-color: {_eai_configuration_policies.UI_BG_WINDOW};
        }}
        QLabel {{
            color: {_eai_configuration_policies.UI_TEXT_PRIMARY};
            background: transparent;
        }}
        QToolTip {{
            color: {_eai_configuration_policies.UI_TEXT_PRIMARY};
            background-color: {_eai_configuration_policies.UI_BG_PANEL};
            border: 1px solid {_eai_configuration_policies.UI_BORDER};
            padding: 4px 8px;
        }}
        QTabWidget#controlTabs::pane, QTabWidget::pane {{
            border: 1px solid {_eai_configuration_policies.UI_BORDER};
            border-radius: {_eai_configuration_policies.UI_RADIUS};
            background: {_eai_configuration_policies.UI_BG_PANEL};
            top: -1px;
        }}
        QTabBar::tab {{
            color: {_eai_configuration_policies.UI_TEXT_SECONDARY};
            background: {_eai_configuration_policies.UI_BG_WINDOW};
            padding: 7px 12px;
            margin-right: 1px;
            border: 1px solid {_eai_configuration_policies.UI_BORDER};
            border-bottom: none;
            border-top-left-radius: {_eai_configuration_policies.UI_RADIUS};
            border-top-right-radius: {_eai_configuration_policies.UI_RADIUS};
            min-width: 4em;
        }}
        QTabBar::tab:hover:!selected {{
            color: {_eai_configuration_policies.UI_TEXT_PRIMARY};
            background: {_eai_configuration_policies.UI_BG_HOVER};
        }}
        QTabBar::tab:selected {{
            color: #ffffff;
            background: {_eai_configuration_policies.UI_BG_PANEL};
            font-weight: bold;
            border-color: {_eai_configuration_policies.UI_BORDER};
            border-bottom: 2px solid {_eai_configuration_policies.UI_ACCENT_BLUE_BRIGHT};
            padding-bottom: 5px;
        }}
        QTabBar::scroller {{
            width: 28px;
        }}
        QTabBar QToolButton {{
            background: {_eai_configuration_policies.UI_BG_ELEVATED};
            border: 1px solid {_eai_configuration_policies.UI_BORDER};
            border-radius: 2px;
            padding: 2px;
        }}
        QTabBar QToolButton:hover {{
            background: {_eai_configuration_policies.UI_BG_HOVER};
            border-color: {_eai_configuration_policies.UI_ACCENT_BLUE};
        }}
        QGroupBox {{
            color: {_eai_configuration_policies.UI_TEXT_PRIMARY};
            border: 1px solid {_eai_configuration_policies.UI_BORDER};
            border-radius: {_eai_configuration_policies.UI_RADIUS};
            margin-top: 10px;
            padding: 10px 8px 8px 8px;
            background-color: {_eai_configuration_policies.UI_BG_ELEVATED};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 6px;
            color: {_eai_configuration_policies.UI_TEXT_PRIMARY};
            font-weight: bold;
        }}
        QCheckBox, QRadioButton {{
            color: {_eai_configuration_policies.UI_TEXT_PRIMARY};
            spacing: 6px;
        }}
        QCheckBox::indicator, QRadioButton::indicator {{
            width: 14px;
            height: 14px;
        }}
        QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QPlainTextEdit, QTextEdit {{
            color: {_eai_configuration_policies.UI_TEXT_PRIMARY};
            background-color: {_eai_configuration_policies.UI_BG_INPUT};
            border: 1px solid {_eai_configuration_policies.UI_BORDER};
            border-radius: {_eai_configuration_policies.UI_RADIUS};
            padding: 3px 6px;
            selection-background-color: {_eai_configuration_policies.UI_HIGHLIGHT};
        }}
        QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
        QPlainTextEdit:focus, QTextEdit:focus {{
            border: 1px solid {_eai_configuration_policies.UI_ACCENT_BLUE_BRIGHT};
        }}
        QComboBox {{
            padding-right: 28px;
        }}
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 26px;
            border: none;
            border-left: 1px solid {_eai_configuration_policies.UI_BORDER};
            border-top-right-radius: {_eai_configuration_policies.UI_RADIUS};
            border-bottom-right-radius: {_eai_configuration_policies.UI_RADIUS};
            background-color: {_eai_configuration_policies.UI_BG_ELEVATED};
        }}
        QComboBox::drop-down:hover {{
            background-color: {_eai_configuration_policies.UI_BG_HOVER};
            border-left-color: {_eai_configuration_policies.UI_BORDER_STRONG};
        }}
        QComboBox::drop-down:disabled {{
            background-color: {_eai_configuration_policies.UI_BG_WINDOW};
            border-left-color: {_eai_configuration_policies.UI_BORDER};
        }}
        QComboBox::down-arrow {{
            image: url("{combo_arrow}");
            width: 12px;
            height: 8px;
        }}
        QComboBox::down-arrow:disabled {{
            image: url("{combo_arrow}");
        }}
        QComboBox QAbstractItemView {{
            color: {_eai_configuration_policies.UI_TEXT_PRIMARY};
            background-color: {_eai_configuration_policies.UI_BG_PANEL};
            border: 1px solid {_eai_configuration_policies.UI_BORDER};
            selection-background-color: {_eai_configuration_policies.UI_HIGHLIGHT};
            outline: 0;
        }}
        QPushButton {{
            color: {_eai_configuration_policies.UI_TEXT_PRIMARY};
            background-color: {_eai_configuration_policies.UI_BG_ELEVATED};
            border: 1px solid {_eai_configuration_policies.UI_BORDER};
            border-radius: {_eai_configuration_policies.UI_RADIUS};
            padding: 5px 12px;
            min-height: 1.2em;
        }}
        QPushButton:hover {{
            background-color: {_eai_configuration_policies.UI_BG_HOVER};
            border-color: {_eai_configuration_policies.UI_BORDER_STRONG};
        }}
        QPushButton:pressed {{
            background-color: {_eai_configuration_policies.UI_BG_INPUT};
        }}
        QPushButton:disabled {{
            color: {_eai_configuration_policies.UI_TEXT_MUTED};
            background-color: {_eai_configuration_policies.UI_BG_WINDOW};
            border-color: {_eai_configuration_policies.UI_BORDER};
        }}
        QPushButton#primaryAction {{
            color: #ffffff;
            background-color: {_eai_configuration_policies.UI_HIGHLIGHT};
            border: 1px solid {_eai_configuration_policies.UI_ACCENT_BLUE_BRIGHT};
            font-weight: bold;
        }}
        QPushButton#primaryAction:hover {{
            background-color: {_eai_configuration_policies.UI_ACCENT_BLUE_BRIGHT};
        }}
        QPushButton#primaryAction:disabled {{
            color: {_eai_configuration_policies.UI_TEXT_MUTED};
            background-color: {_eai_configuration_policies.UI_BG_ELEVATED};
            border-color: {_eai_configuration_policies.UI_BORDER};
            font-weight: normal;
        }}
        QPushButton#dangerAction {{
            color: {_eai_configuration_policies.UI_ACCENT_RED};
            border-color: #6a4040;
        }}
        QPushButton#dangerAction:hover {{
            background-color: #3a2828;
            border-color: {_eai_configuration_policies.UI_ACCENT_RED};
        }}
        QSplitter::handle {{
            background: {_eai_configuration_policies.UI_BORDER};
        }}
        QSplitter::handle:horizontal {{
            width: 3px;
            margin: 2px 0;
        }}
        QSplitter::handle:vertical {{
            height: 3px;
            margin: 0 2px;
        }}
        QSplitter::handle:hover {{
            background: {_eai_configuration_policies.UI_ACCENT_BLUE};
        }}
        QStatusBar {{
            color: {_eai_configuration_policies.UI_TEXT_SECONDARY};
            background: {_eai_configuration_policies.UI_BG_PANEL};
            border-top: 1px solid {_eai_configuration_policies.UI_BORDER};
            padding: 2px 6px;
        }}
        QStatusBar::item {{
            border: none;
        }}
        QScrollArea {{
            border: none;
            background: transparent;
        }}
        QScrollBar:vertical {{
            background: {_eai_configuration_policies.UI_BG_WINDOW};
            width: 10px;
            margin: 0;
            border: none;
        }}
        QScrollBar::handle:vertical {{
            background: {_eai_configuration_policies.UI_BORDER_STRONG};
            border-radius: 4px;
            min-height: 28px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {_eai_configuration_policies.UI_ACCENT_BLUE};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            height: 0;
            background: none;
        }}
        QScrollBar:horizontal {{
            background: {_eai_configuration_policies.UI_BG_WINDOW};
            height: 10px;
            margin: 0;
            border: none;
        }}
        QScrollBar::handle:horizontal {{
            background: {_eai_configuration_policies.UI_BORDER_STRONG};
            border-radius: 4px;
            min-width: 28px;
        }}
        QScrollBar::handle:horizontal:hover {{
            background: {_eai_configuration_policies.UI_ACCENT_BLUE};
        }}
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
            width: 0;
            background: none;
        }}
        QListWidget, QTreeWidget, QTableWidget {{
            background-color: {_eai_configuration_policies.UI_BG_INPUT};
            border: 1px solid {_eai_configuration_policies.UI_BORDER};
            border-radius: {_eai_configuration_policies.UI_RADIUS};
            outline: 0;
        }}
        QListWidget::item:selected, QTreeWidget::item:selected,
        QTableWidget::item:selected {{
            background-color: {_eai_configuration_policies.UI_HIGHLIGHT};
            color: #ffffff;
        }}
        QHeaderView::section {{
            background-color: {_eai_configuration_policies.UI_BG_ELEVATED};
            color: {_eai_configuration_policies.UI_TEXT_SECONDARY};
            border: none;
            border-right: 1px solid {_eai_configuration_policies.UI_BORDER};
            border-bottom: 1px solid {_eai_configuration_policies.UI_BORDER};
            padding: 4px 8px;
        }}
        QProgressBar {{
            border: 1px solid {_eai_configuration_policies.UI_BORDER};
            border-radius: {_eai_configuration_policies.UI_RADIUS};
            background: {_eai_configuration_policies.UI_BG_INPUT};
            text-align: center;
            color: {_eai_configuration_policies.UI_TEXT_PRIMARY};
        }}
        QProgressBar::chunk {{
            background-color: {_eai_configuration_policies.UI_ACCENT_BLUE_BRIGHT};
            border-radius: 2px;
        }}
        """
    )

