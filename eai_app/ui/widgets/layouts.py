"""ui.widgets.layouts — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class _LogHeightDragBar(_eai_platform_bootstrap.QFrame):
    """宽拖动手柄：拖动可调整关联文本框高度（比 QSplitter 细条更好抓）。"""

    def __init__(
        self,
        log_widget: _eai_platform_bootstrap.QWidget,
        *,
        grow_down: bool,
        min_h: int = 80,
        max_h: int = 900,
        label: str = "拖动调整高度",
        parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._log = log_widget
        self._grow_down = grow_down
        self._min_h = min_h
        self._max_h = max_h
        self._dragging = False
        self._press_y = 0
        self._press_h = 0
        self._was_at_bottom = True
        self.setFixedHeight(22)
        self.setCursor(_eai_platform_bootstrap.Qt.SizeVerCursor)
        self.setToolTip(f"按住上下拖动，{label}")
        self.setStyleSheet(
            "QFrame {"
            f"  background-color: {_eai_configuration_policies.UI_BG_ELEVATED};"
            f"  border: 1px solid {_eai_configuration_policies.UI_BORDER};"
            f"  border-radius: {_eai_configuration_policies.UI_RADIUS};"
            "}"
            "QFrame:hover {"
            f"  background-color: {_eai_configuration_policies.UI_BG_HOVER};"
            f"  border: 1px solid {_eai_configuration_policies.UI_ACCENT_BLUE};"
            "}"
        )
        lay = _eai_platform_bootstrap.QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        tip = _eai_platform_bootstrap.QLabel(f"⋮⋮ {label} ⋮⋮")
        tip.setAlignment(_eai_platform_bootstrap.Qt.AlignCenter)
        tip.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED}; background: transparent; border: none;")
        tip.setAttribute(_eai_platform_bootstrap.Qt.WA_TransparentForMouseEvents, True)
        lay.addWidget(tip)

    def _log_at_bottom(self) -> bool:
        bar = getattr(self._log, "verticalScrollBar", lambda: None)()
        if bar is None:
            return True
        return bar.value() >= max(0, bar.maximum() - 4)

    def _fit_max_height(self) -> int:
        """Clamp so the log (and siblings) stay fully inside the parent — avoid
        growing past the window and clipping the bottom of the log view."""
        parent = self._log.parentWidget()
        if parent is None:
            return self._max_h
        layout = parent.layout()
        parent_h = parent.height()
        if parent_h <= 0:
            return self._max_h
        used = 0
        if layout is not None:
            margins = layout.contentsMargins()
            used += margins.top() + margins.bottom()
            for i in range(layout.count()):
                item = layout.itemAt(i)
                if item is None:
                    continue
                w = item.widget()
                if w is self._log:
                    continue
                if w is not None:
                    if not w.isVisible():
                        continue
                    stretch = 0
                    try:
                        stretch = int(layout.stretch(i))
                    except Exception:  # noqa: BLE001
                        stretch = 0
                    if stretch > 0:
                        # Leave stretch panes a usable minimum so they (and the
                        # log below/above) are not pushed off-screen.
                        used += max(48, w.minimumSizeHint().height())
                    else:
                        used += w.height() if w.height() > 0 else max(
                            w.minimumSizeHint().height(), w.sizeHint().height()
                        )
                    continue
                lay = item.layout()
                if lay is not None:
                    used += max(lay.minimumSize().height(), lay.sizeHint().height())
                    continue
                sp = item.spacerItem()
                if sp is not None and sp.sizeHint().height() > 0:
                    used += sp.sizeHint().height()
            if layout.count() > 1:
                used += (layout.count() - 1) * max(0, layout.spacing())
        room = parent_h - used
        return max(self._min_h, min(self._max_h, room))

    def _apply_height(self, new_h: int) -> None:
        fit = self._fit_max_height()
        h = max(self._min_h, min(fit, new_h))
        self._log.setFixedHeight(h)
        self._log.updateGeometry()
        parent = self._log.parentWidget()
        if parent is not None and parent.layout() is not None:
            parent.layout().activate()
        # If we were pinned to the latest lines, keep the bottom visible after
        # the viewport height change (otherwise the clipped bottom stays hidden).
        if self._was_at_bottom:
            bar = getattr(self._log, "verticalScrollBar", lambda: None)()
            if bar is not None:
                bar.setValue(bar.maximum())

    def eventFilter(self, obj, event):  # noqa: N802
        # Parent shrunk → pull log back into view so bottom lines aren't clipped.
        if obj is self._log.parentWidget() and event.type() == _eai_platform_bootstrap.QEvent.Resize:
            if self._log.height() > self._fit_max_height():
                self._was_at_bottom = self._log_at_bottom()
                self._apply_height(self._log.height())
        return super().eventFilter(obj, event)

    def showEvent(self, event) -> None:  # noqa: N802
        parent = self._log.parentWidget()
        if parent is not None:
            parent.installEventFilter(self)
        super().showEvent(event)

    def mousePressEvent(self, event: _eai_platform_bootstrap.QMouseEvent) -> None:
        if event.button() == _eai_platform_bootstrap.Qt.LeftButton:
            self._dragging = True
            self._press_y = event.globalY()
            self._press_h = max(self._log.height(), self._min_h)
            self._was_at_bottom = self._log_at_bottom()
            self.grabMouse()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: _eai_platform_bootstrap.QMouseEvent) -> None:
        if not self._dragging:
            super().mouseMoveEvent(event)
            return
        dy = event.globalY() - self._press_y
        delta = dy if self._grow_down else -dy
        self._apply_height(self._press_h + delta)
        event.accept()

    def mouseReleaseEvent(self, event: _eai_platform_bootstrap.QMouseEvent) -> None:
        if event.button() == _eai_platform_bootstrap.Qt.LeftButton and self._dragging:
            self._dragging = False
            self.releaseMouse()
            # Final clamp after sibling layouts settle.
            self._apply_height(self._log.height())
            event.accept()
            return
        super().mouseReleaseEvent(event)


def _configure_resizable_text_edit(
    edit: _eai_platform_bootstrap.QTextEdit,
    *,
    default_h: int,
    tooltip: str = "",
) -> None:
    """给多行文本框设合理默认高度（不再附加拖动手柄以节省纵向空间）。"""
    edit.setMinimumHeight(0)
    edit.setMaximumHeight(16777215)
    edit.setFixedHeight(int(default_h))
    edit.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Fixed)
    if tooltip:
        edit.setToolTip(tooltip)


def _add_clear_log_button_row(layout: _eai_platform_bootstrap.QVBoxLayout, edit: _eai_platform_bootstrap.QTextEdit) -> _eai_platform_bootstrap.QPushButton:
    """在日志文本框上方放一排右对齐的「清空日志」按钮。"""
    row = _eai_platform_bootstrap.QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(6)
    row.addStretch(1)
    btn = _eai_platform_bootstrap.QPushButton("清空日志")
    btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
    btn.setToolTip("清空下方日志内容")
    btn.clicked.connect(edit.clear)
    row.addWidget(btn)
    layout.addLayout(row)
    return btn


def _add_text_edit_with_drag_bar(
    layout: _eai_platform_bootstrap.QVBoxLayout,
    edit: _eai_platform_bootstrap.QTextEdit,
    *,
    default_h: int,
    min_h: int = 80,
    max_h: int = 900,
    grow_down: bool = True,
    bar_before: bool = False,
    label: str = "拖动调整高度",
    clearable: bool = False,
) -> _eai_platform_bootstrap.Optional[_LogHeightDragBar]:
    """放入文本框；拖动手柄已移除（保留参数兼容旧调用）。"""
    del min_h, max_h, grow_down, bar_before, label  # unused — drag bars removed
    if clearable:
        _add_clear_log_button_row(layout, edit)
    _configure_resizable_text_edit(edit, default_h=default_h, tooltip="")
    layout.addWidget(edit)
    return None


class _FlowLayout(_eai_platform_bootstrap.QLayout):
    """Horizontal wrapping layout so controls stay readable on narrow widths."""

    def __init__(
        self,
        parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None,
        *,
        margin: int = 0,
        spacing: int = 6,
    ) -> None:
        super().__init__(parent)
        self._items: _eai_platform_bootstrap.List[_eai_platform_bootstrap.Any] = []
        self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(int(spacing))

    def addWidget(self, w, *args, **kwargs) -> None:  # noqa: N802
        # Accept QHBoxLayout-style stretch/alignment; flow layout ignores them.
        super().addWidget(w)

    def addItem(self, item) -> None:  # noqa: N802
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):  # noqa: N802
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self):  # noqa: N802
        return _eai_platform_bootstrap.Qt.Orientations(_eai_platform_bootstrap.Qt.Orientation(0))

    def hasHeightForWidth(self) -> bool:  # noqa: N802
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802
        return self._do_layout(_eai_platform_bootstrap.QRect(0, 0, width, 0), True)

    def setGeometry(self, rect: _eai_platform_bootstrap.QRect) -> None:  # noqa: N802
        super().setGeometry(rect)
        self._do_layout(rect, False)

    def sizeHint(self) -> _eai_platform_bootstrap.QSize:  # noqa: N802
        return self.minimumSize()

    def minimumSize(self) -> _eai_platform_bootstrap.QSize:  # noqa: N802
        size = _eai_platform_bootstrap.QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        m = self.contentsMargins()
        size += _eai_platform_bootstrap.QSize(m.left() + m.right(), m.top() + m.bottom())
        size.setHeight(max(size.height(), 22))
        return size

    def _do_layout(self, rect: _eai_platform_bootstrap.QRect, test_only: bool) -> int:
        m = self.contentsMargins()
        effective = rect.adjusted(m.left(), m.top(), -m.right(), -m.bottom())
        x = effective.x()
        y = effective.y()
        line_h = 0
        space = max(0, self.spacing())
        for item in self._items:
            hint = item.sizeHint()
            next_x = x + hint.width() + space
            if line_h > 0 and next_x - space > effective.right() + 1:
                x = effective.x()
                y = y + line_h + space
                next_x = x + hint.width() + space
                line_h = 0
            if not test_only:
                item.setGeometry(_eai_platform_bootstrap.QRect(_eai_platform_bootstrap.QPoint(x, y), hint))
            x = next_x
            line_h = max(line_h, hint.height())
        return y + line_h - rect.y() + m.bottom()


def _wrap_in_vscroll(inner: _eai_platform_bootstrap.QWidget) -> _eai_platform_bootstrap.QScrollArea:
    """Put *inner* in a scroll area so nothing is clipped when the tab is short."""
    scroll = _eai_platform_bootstrap.QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(_eai_platform_bootstrap.QFrame.NoFrame)
    scroll.setHorizontalScrollBarPolicy(_eai_platform_bootstrap.Qt.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(_eai_platform_bootstrap.Qt.ScrollBarAsNeeded)
    scroll.setWidget(inner)
    scroll.setSizePolicy(_eai_platform_bootstrap.QSizePolicy.Expanding, _eai_platform_bootstrap.QSizePolicy.Expanding)
    return scroll


def _tab_page_matches(widget: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget], original: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget]) -> bool:
    if widget is None or original is None:
        return False
    if widget is original or getattr(widget, '_scroll_original', None) is original:
        return True
    return isinstance(widget, _eai_platform_bootstrap.QScrollArea) and widget.widget() is original


def _install_scroll_for_tab_widget(tabs: _eai_platform_bootstrap.QTabWidget) -> int:
    """Wrap every non-scroll page while preserving its original QWidget."""
    wrapped = 0
    current = tabs.currentIndex()
    tabs.blockSignals(True)
    try:
        for index in range(tabs.count()):
            page = tabs.widget(index)
            if page is None or isinstance(page, _eai_platform_bootstrap.QScrollArea):
                continue
            if getattr(page, '_auto_tab_scroll_content', False):
                continue
            title = tabs.tabText(index)
            icon = tabs.tabIcon(index)
            tooltip = tabs.tabToolTip(index)
            whats_this = tabs.tabWhatsThis(index)
            enabled = tabs.isTabEnabled(index)
            data = tabs.tabBar().tabData(index)
            tabs.removeTab(index)
            scroll = _wrap_in_vscroll(page)
            scroll._scroll_original = page
            page._auto_tab_scroll_content = True
            tabs.insertTab(index, scroll, icon, title)
            tabs.setTabToolTip(index, tooltip)
            tabs.setTabWhatsThis(index, whats_this)
            tabs.setTabEnabled(index, enabled)
            tabs.tabBar().setTabData(index, data)
            wrapped += 1
    finally:
        if tabs.count():
            tabs.setCurrentIndex(max(0, min(current, tabs.count()-1)))
        tabs.blockSignals(False)
    return wrapped


def _always_visible_splitter(orientation: _eai_platform_bootstrap.Qt.Orientation = _eai_platform_bootstrap.Qt.Vertical) -> _eai_platform_bootstrap.QSplitter:
    split = _eai_platform_bootstrap.QSplitter(orientation)
    split.setChildrenCollapsible(False)
    split.setHandleWidth(8)
    split.setOpaqueResize(True)
    return split

