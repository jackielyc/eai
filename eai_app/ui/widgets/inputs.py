"""ui.widgets.inputs — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_models = module_ref('eai_app.configuration.models')
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

def _is_text_ime_widget(widget) -> bool:
    if widget is None:
        return False
    if not isinstance(widget, (_eai_platform_bootstrap.QLineEdit, _eai_platform_bootstrap.QTextEdit)):
        return False
    try:
        if hasattr(widget, "isReadOnly") and widget.isReadOnly():
            return False
    except Exception:
        pass
    return True


def _remember_text_focus() -> None:
    pass
    try:
        fw = _eai_platform_bootstrap.QApplication.focusWidget()
        if _is_text_ime_widget(fw):
            _eai_platform_bootstrap._IME_LAST_TEXT_WIDGET = fw
    except Exception:
        pass


def _keep_text_focus() -> None:
    w = _eai_platform_bootstrap._IME_LAST_TEXT_WIDGET
    if not _is_text_ime_widget(w):
        return
    try:
        w.setFocus(_eai_platform_bootstrap.Qt.OtherFocusReason)
    except Exception:
        pass


def _revive_ime_after_combo() -> None:
    """下拉关闭后只交还焦点，不重建控件（重建反而常弄死 fcitx）。"""
    _keep_text_focus()


def restore_fcitx_input_method(widget=None) -> None:
    _revive_ime_after_combo()


def _schedule_fcitx_restore(widget=None) -> None:
    try:
        _eai_platform_bootstrap.QTimer.singleShot(0, _revive_ime_after_combo)
        _eai_platform_bootstrap.QTimer.singleShot(80, _revive_ime_after_combo)
    except Exception:
        pass


def _notify_ime_widget(widget, *, full_query: bool = False) -> None:
    """文本框获焦后通知 Qt/fcitx 更新输入法上下文。"""
    try:
        app = _eai_platform_bootstrap.QApplication.instance()
        if app is None or widget is None:
            return
        im = app.inputMethod()
        if im is None:
            return
        if full_query:
            im.update(_eai_platform_bootstrap.Qt.ImQueryAll)
        else:
            im.update(
                _eai_platform_bootstrap.Qt.ImEnabled
                | _eai_platform_bootstrap.Qt.ImCursorRectangle
                | _eai_platform_bootstrap.Qt.ImHints
                | _eai_platform_bootstrap.Qt.ImInputItemClipRectangle
            )
    except Exception:
        pass


class ImeSafeLineEdit(_eai_platform_bootstrap.QLineEdit):
    """支持 fcitx 中文输入的单行框。"""

    def __init__(self, text: str = "", parent=None) -> None:
        # 兼容 ImeSafeLineEdit(parent) / ImeSafeLineEdit("", parent)
        if not isinstance(text, str):
            parent = text
            text = ""
        super().__init__(text, parent)
        self.setAttribute(_eai_platform_bootstrap.Qt.WA_InputMethodEnabled, True)
        self.setFocusPolicy(_eai_platform_bootstrap.Qt.StrongFocus)

    def focusInEvent(self, event) -> None:  # type: ignore[override]
        super().focusInEvent(event)
        pass
        _eai_platform_bootstrap._IME_LAST_TEXT_WIDGET = self
        _notify_ime_widget(self)


class ImeSafeTextEdit(_eai_platform_bootstrap.QTextEdit):
    """支持 fcitx 中文输入的多行框（无 Enter 发送逻辑）。"""

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(_eai_platform_bootstrap.Qt.WA_InputMethodEnabled, True)
        self.setFocusPolicy(_eai_platform_bootstrap.Qt.StrongFocus)

    def focusInEvent(self, event) -> None:  # type: ignore[override]
        super().focusInEvent(event)
        pass
        _eai_platform_bootstrap._IME_LAST_TEXT_WIDGET = self
        _notify_ime_widget(self)


class HoverWheelSpinBox(_eai_platform_bootstrap.QSpinBox):
    """悬停即可滚轮改值（无需先点击获得焦点）。"""

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None) -> None:
        super().__init__(parent)
        self.setFocusPolicy(_eai_platform_bootstrap.Qt.StrongFocus)

    def wheelEvent(self, event) -> None:  # type: ignore[override]
        if not self.isEnabled():
            event.ignore()
            return
        if not self.hasFocus():
            self.setFocus(_eai_platform_bootstrap.Qt.MouseFocusReason)
        super().wheelEvent(event)


class ImeSafeComboBox(_eai_platform_bootstrap.QComboBox):
    """鼠标可选下拉：列表做主窗口子控件浮层，不创建新的 X11 窗口。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAttribute(_eai_platform_bootstrap.Qt.WA_InputMethodEnabled, False)
        self.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        self._ime_list: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QListWidget] = None

    def _ensure_list(self) -> _eai_platform_bootstrap.QListWidget:
        win = self.window()
        if self._ime_list is not None:
            # 父窗口变了则重建
            if self._ime_list.parent() is win:
                return self._ime_list
            try:
                self._ime_list.hide()
                self._ime_list.setParent(None)
                self._ime_list.deleteLater()
            except Exception:
                pass
            self._ime_list = None
        lw = _eai_platform_bootstrap.QListWidget(win)
        # 普通子控件，禁止变顶层 Tool/Popup
        lw.setWindowFlags(_eai_platform_bootstrap.Qt.Widget)
        lw.setAttribute(_eai_platform_bootstrap.Qt.WA_InputMethodEnabled, False)
        lw.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        lw.setHorizontalScrollBarPolicy(_eai_platform_bootstrap.Qt.ScrollBarAlwaysOff)
        lw.setStyleSheet(
            "QListWidget { background: #2d2d2d; color: #eee; border: 1px solid #666; }"
            "QListWidget::item:selected { background: #3d6ea8; }"
            "QListWidget::item:hover { background: #454545; }"
        )
        lw.itemClicked.connect(self._on_ime_item_clicked)
        self._ime_list = lw
        return lw

    def showPopup(self) -> None:  # type: ignore[override]
        _remember_text_focus()
        for c in list(_eai_platform_bootstrap._IME_OPEN_POPUPS):
            try:
                if c is not self:
                    c.hidePopup()
            except Exception:
                pass
        lw = self._ensure_list()
        lw.clear()
        cur = self.currentIndex()
        has_thumb = getattr(self, "_eval_task_thumbs", None) is not None
        if has_thumb:
            lw.setIconSize(_eai_platform_bootstrap.QSize(120, 68))
            lw.setSpacing(3)
            lw.setTextElideMode(_eai_platform_bootstrap.Qt.ElideRight)
        else:
            lw.setIconSize(_eai_platform_bootstrap.QSize())
            lw.setSpacing(0)
        for i in range(self.count()):
            text = self.itemText(i)
            pix = _eai_configuration_models.combo_task_thumb(self, i) if has_thumb else None
            if pix is not None:
                item = _eai_platform_bootstrap.QListWidgetItem(_eai_platform_bootstrap.QIcon(pix), text)
                item.setSizeHint(_eai_platform_bootstrap.QSize(0, 76))
            else:
                icon = self.itemIcon(i)
                item = (
                    _eai_platform_bootstrap.QListWidgetItem(icon, text)
                    if not icon.isNull()
                    else _eai_platform_bootstrap.QListWidgetItem(text)
                )
            tip = self.itemData(i, _eai_platform_bootstrap.Qt.ToolTipRole)
            if tip:
                item.setToolTip(str(tip))
            try:
                enabled = bool(
                    self.model().flags(self.model().index(i, self.modelColumn()))
                    & _eai_platform_bootstrap.Qt.ItemIsEnabled
                )
            except Exception:
                enabled = True
            if not enabled:
                item.setFlags(item.flags() & ~_eai_platform_bootstrap.Qt.ItemIsEnabled & ~_eai_platform_bootstrap.Qt.ItemIsSelectable)
            lw.addItem(item)
        if 0 <= cur < lw.count():
            lw.setCurrentRow(cur)
        rows = max(lw.count(), 1)
        row_h = lw.sizeHintForRow(0) if lw.count() else 24
        if row_h <= 0:
            row_h = 24
        visible = min(rows, max(self.maxVisibleItems(), 8))
        if has_thumb:
            row_h = max(row_h, 76)
        popup_w = max(self.width(), 440 if has_thumb else 120)
        parent_for_w = lw.parentWidget()
        if parent_for_w is not None:
            popup_w = min(popup_w, max(self.width(), parent_for_w.width() - 12))
        lw.setFixedWidth(popup_w)
        lw.setFixedHeight(min(500 if has_thumb else 280, row_h * visible + 4))
        # 相对主窗口定位（同一 X 窗口内）
        parent = lw.parentWidget()
        gp = self.mapToGlobal(_eai_platform_bootstrap.QPoint(0, self.height()))
        if parent is not None:
            lp = parent.mapFromGlobal(gp)
            # 若底部不够，向上展开
            if lp.y() + lw.height() > parent.height():
                gp2 = self.mapToGlobal(_eai_platform_bootstrap.QPoint(0, 0))
                lp2 = parent.mapFromGlobal(gp2)
                lp = _eai_platform_bootstrap.QPoint(lp2.x(), max(0, lp2.y() - lw.height()))
            lw.move(lp)
        lw.show()
        lw.raise_()
        if self not in _eai_platform_bootstrap._IME_OPEN_POPUPS:
            _eai_platform_bootstrap._IME_OPEN_POPUPS.append(self)
        _keep_text_focus()
        _eai_platform_bootstrap.QTimer.singleShot(0, _keep_text_focus)

    def hidePopup(self) -> None:  # type: ignore[override]
        lw = self._ime_list
        if lw is not None:
            lw.hide()
        try:
            _eai_platform_bootstrap._IME_OPEN_POPUPS.remove(self)
        except ValueError:
            pass

    def _on_ime_item_clicked(self, item: _eai_platform_bootstrap.QListWidgetItem) -> None:
        if item is None or not (item.flags() & _eai_platform_bootstrap.Qt.ItemIsEnabled):
            return
        lw = self._ime_list
        row = lw.row(item) if lw is not None else -1
        if row < 0:
            return
        self.hidePopup()
        if row != self.currentIndex():
            self.setCurrentIndex(row)
        self.activated.emit(row)
        # 选完后再恢复 IC（此时回调已跑完）
        _eai_platform_bootstrap.QTimer.singleShot(0, _revive_ime_after_combo)
        _eai_platform_bootstrap.QTimer.singleShot(100, _revive_ime_after_combo)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == _eai_platform_bootstrap.Qt.LeftButton:
            if self._ime_list is not None and self._ime_list.isVisible():
                self.hidePopup()
                _eai_platform_bootstrap.QTimer.singleShot(0, _revive_ime_after_combo)
            else:
                self.showPopup()
            event.accept()
            return
        super().mousePressEvent(event)


def _combo_fuzzy_match(query: str, text: str) -> bool:
    """大小写不敏感：整句包含、空格分词全命中、或字符子序列。"""
    q = (query or "").strip().lower()
    t = (text or "").lower()
    if not q:
        return True
    if q in t:
        return True
    tokens = [tok for tok in q.split() if tok]
    if len(tokens) > 1 and all(tok in t for tok in tokens):
        return True
    i = 0
    for ch in t:
        if i < len(q) and ch == q[i]:
            i += 1
            if i >= len(q):
                return True
    return i >= len(q)


class FilterableImeSafeComboBox(ImeSafeComboBox):
    """带搜索框的 IME 安全下拉：在浮层顶部输入关键词模糊过滤。"""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._ime_popup: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None
        self._ime_filter: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QLineEdit] = None
        self._filter_source_indices: _eai_platform_bootstrap.List[int] = []

    def _ensure_popup(self) -> _eai_platform_bootstrap.Tuple[_eai_platform_bootstrap.QWidget, _eai_platform_bootstrap.QLineEdit, _eai_platform_bootstrap.QListWidget]:
        win = self.window()
        if self._ime_popup is not None and self._ime_popup.parent() is win:
            assert self._ime_filter is not None and self._ime_list is not None
            return self._ime_popup, self._ime_filter, self._ime_list
        if self._ime_popup is not None:
            try:
                self._ime_popup.hide()
                self._ime_popup.setParent(None)
                self._ime_popup.deleteLater()
            except Exception:
                pass
            self._ime_popup = None
            self._ime_list = None
            self._ime_filter = None

        panel = _eai_platform_bootstrap.QWidget(win)
        panel.setWindowFlags(_eai_platform_bootstrap.Qt.Widget)
        panel.setAttribute(_eai_platform_bootstrap.Qt.WA_InputMethodEnabled, False)
        panel.setStyleSheet(
            "QWidget { background: #2d2d2d; border: 1px solid #666; }"
        )
        layout = _eai_platform_bootstrap.QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        filt = ImeSafeLineEdit("", panel)
        filt.setPlaceholderText("输入关键词过滤…")
        filt.setClearButtonEnabled(True)
        filt.setStyleSheet(
            f"QLineEdit {{ background: #1e1e1e; color: #eee; border: 1px solid {_eai_configuration_policies.UI_BORDER}; "
            "padding: 3px 6px; }"
        )
        filt.textChanged.connect(self._on_filter_text_changed)
        filt.returnPressed.connect(self._on_filter_return)
        filt.installEventFilter(self)
        layout.addWidget(filt)

        lw = _eai_platform_bootstrap.QListWidget(panel)
        lw.setWindowFlags(_eai_platform_bootstrap.Qt.Widget)
        lw.setAttribute(_eai_platform_bootstrap.Qt.WA_InputMethodEnabled, False)
        lw.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        lw.setHorizontalScrollBarPolicy(_eai_platform_bootstrap.Qt.ScrollBarAlwaysOff)
        lw.setStyleSheet(
            "QListWidget { background: #2d2d2d; color: #eee; border: none; }"
            "QListWidget::item:selected { background: #3d6ea8; }"
            "QListWidget::item:hover { background: #454545; }"
        )
        lw.itemClicked.connect(self._on_ime_item_clicked)
        layout.addWidget(lw, 1)

        self._ime_popup = panel
        self._ime_filter = filt
        self._ime_list = lw
        return panel, filt, lw

    def _ensure_list(self) -> _eai_platform_bootstrap.QListWidget:
        return self._ensure_popup()[2]

    def _rebuild_filtered_list(self, query: str = "") -> None:
        lw = self._ime_list
        if lw is None:
            return
        lw.clear()
        self._filter_source_indices = []
        cur = self.currentIndex()
        select_row = -1
        has_thumb = getattr(self, "_eval_task_thumbs", None) is not None
        if has_thumb:
            lw.setIconSize(_eai_platform_bootstrap.QSize(120, 68))
            lw.setSpacing(3)
            lw.setTextElideMode(_eai_platform_bootstrap.Qt.ElideRight)
        else:
            lw.setIconSize(_eai_platform_bootstrap.QSize())
            lw.setSpacing(0)
        for i in range(self.count()):
            text = self.itemText(i)
            tip = self.itemData(i, _eai_platform_bootstrap.Qt.ToolTipRole)
            data = self.itemData(i)
            path_bits: _eai_platform_bootstrap.List[str] = []
            if isinstance(data, dict):
                path_bits.extend(
                    str(data.get(k) or "")
                    for k in ("path", "name", "model_id", "label", "root")
                )
            hay = " ".join([text, str(tip or ""), *path_bits])
            if not _combo_fuzzy_match(query, hay):
                continue
            pix = _eai_configuration_models.combo_task_thumb(self, i) if has_thumb else None
            if pix is not None:
                item = _eai_platform_bootstrap.QListWidgetItem(_eai_platform_bootstrap.QIcon(pix), text)
                item.setSizeHint(_eai_platform_bootstrap.QSize(0, 76))
            else:
                item = _eai_platform_bootstrap.QListWidgetItem(text)
            if tip:
                item.setToolTip(str(tip))
            item.setData(_eai_platform_bootstrap.Qt.UserRole, i)
            try:
                enabled = bool(
                    self.model().flags(self.model().index(i, self.modelColumn()))
                    & _eai_platform_bootstrap.Qt.ItemIsEnabled
                )
            except Exception:
                enabled = True
            if not enabled:
                item.setFlags(item.flags() & ~_eai_platform_bootstrap.Qt.ItemIsEnabled & ~_eai_platform_bootstrap.Qt.ItemIsSelectable)
            lw.addItem(item)
            self._filter_source_indices.append(i)
            if i == cur:
                select_row = lw.count() - 1
        if select_row >= 0:
            lw.setCurrentRow(select_row)
        elif lw.count() > 0:
            lw.setCurrentRow(0)

    def showPopup(self) -> None:  # type: ignore[override]
        _remember_text_focus()
        for c in list(_eai_platform_bootstrap._IME_OPEN_POPUPS):
            try:
                if c is not self:
                    c.hidePopup()
            except Exception:
                pass
        panel, filt, lw = self._ensure_popup()
        filt.blockSignals(True)
        filt.clear()
        filt.blockSignals(False)
        self._rebuild_filtered_list("")
        rows = max(lw.count(), 1)
        row_h = lw.sizeHintForRow(0) if lw.count() else 24
        if row_h <= 0:
            row_h = 24
        visible = min(rows, max(self.maxVisibleItems(), 10))
        has_thumb = lw.iconSize().width() >= 48
        if has_thumb:
            row_h = max(row_h, 76)
        list_h = min(500 if has_thumb else 320, row_h * visible + 4)
        panel.setFixedWidth(max(self.width(), 440 if has_thumb else 280))
        panel.setFixedHeight(list_h + filt.sizeHint().height() + 16)
        parent = panel.parentWidget()
        gp = self.mapToGlobal(_eai_platform_bootstrap.QPoint(0, self.height()))
        if parent is not None:
            lp = parent.mapFromGlobal(gp)
            if lp.y() + panel.height() > parent.height():
                gp2 = self.mapToGlobal(_eai_platform_bootstrap.QPoint(0, 0))
                lp2 = parent.mapFromGlobal(gp2)
                lp = _eai_platform_bootstrap.QPoint(lp2.x(), max(0, lp2.y() - panel.height()))
            panel.move(lp)
        panel.show()
        panel.raise_()
        if self not in _eai_platform_bootstrap._IME_OPEN_POPUPS:
            _eai_platform_bootstrap._IME_OPEN_POPUPS.append(self)
        _eai_platform_bootstrap.QTimer.singleShot(0, filt.setFocus)
        _eai_platform_bootstrap.QTimer.singleShot(0, filt.selectAll)

    def hidePopup(self) -> None:  # type: ignore[override]
        panel = self._ime_popup
        if panel is not None:
            panel.hide()
        elif self._ime_list is not None:
            self._ime_list.hide()
        try:
            _eai_platform_bootstrap._IME_OPEN_POPUPS.remove(self)
        except ValueError:
            pass

    def _on_filter_text_changed(self, text: str) -> None:
        self._rebuild_filtered_list(text)

    def _on_filter_return(self) -> None:
        lw = self._ime_list
        if lw is None or lw.count() <= 0:
            return
        item = lw.currentItem() or lw.item(0)
        if item is not None:
            self._on_ime_item_clicked(item)

    def _on_ime_item_clicked(self, item: _eai_platform_bootstrap.QListWidgetItem) -> None:
        if item is None or not (item.flags() & _eai_platform_bootstrap.Qt.ItemIsEnabled):
            return
        src = item.data(_eai_platform_bootstrap.Qt.UserRole)
        try:
            row = int(src)
        except (TypeError, ValueError):
            row = -1
        if row < 0:
            return
        self.hidePopup()
        if row != self.currentIndex():
            self.setCurrentIndex(row)
        self.activated.emit(row)
        _eai_platform_bootstrap.QTimer.singleShot(0, _revive_ime_after_combo)
        _eai_platform_bootstrap.QTimer.singleShot(100, _revive_ime_after_combo)

    def eventFilter(self, obj, event):  # type: ignore[override]
        if obj is self._ime_filter and event.type() == _eai_platform_bootstrap.QEvent.KeyPress:
            key = event.key()
            lw = self._ime_list
            if key == _eai_platform_bootstrap.Qt.Key_Escape:
                self.hidePopup()
                _eai_platform_bootstrap.QTimer.singleShot(0, _revive_ime_after_combo)
                return True
            if lw is not None and lw.count() > 0:
                row = lw.currentRow()
                if key == _eai_platform_bootstrap.Qt.Key_Down:
                    lw.setCurrentRow(min(row + 1, lw.count() - 1) if row >= 0 else 0)
                    return True
                if key == _eai_platform_bootstrap.Qt.Key_Up:
                    lw.setCurrentRow(max(row - 1, 0) if row >= 0 else 0)
                    return True
        return super().eventFilter(obj, event)

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if event.button() == _eai_platform_bootstrap.Qt.LeftButton:
            panel = self._ime_popup
            visible = panel is not None and panel.isVisible()
            if visible:
                self.hidePopup()
                _eai_platform_bootstrap.QTimer.singleShot(0, _revive_ime_after_combo)
            else:
                self.showPopup()
            event.accept()
            return
        super(ImeSafeComboBox, self).mousePressEvent(event)


class ImePopupDismissFilter(_eai_platform_bootstrap.QObject):
    """点击浮层外关闭；关闭后恢复中文 IC。"""

    def eventFilter(self, obj, event):  # type: ignore[override]
        try:
            et = event.type()
            if et == _eai_platform_bootstrap.QEvent.FocusIn and _is_text_ime_widget(obj):
                pass
                _eai_platform_bootstrap._IME_LAST_TEXT_WIDGET = obj
            if not _eai_platform_bootstrap._IME_OPEN_POPUPS:
                return False
            if et == _eai_platform_bootstrap.QEvent.MouseButtonPress:
                gp = event.globalPos() if hasattr(event, "globalPos") else None
                if gp is None:
                    return False
                closed = False
                for combo in list(_eai_platform_bootstrap._IME_OPEN_POPUPS):
                    popup = getattr(combo, "_ime_popup", None)
                    lw = getattr(combo, "_ime_list", None)
                    surface = popup if (popup is not None and popup.isVisible()) else lw
                    if surface is None or not surface.isVisible():
                        continue
                    if combo.rect().contains(combo.mapFromGlobal(gp)):
                        continue
                    if surface.rect().contains(surface.mapFromGlobal(gp)):
                        continue
                    combo.hidePopup()
                    closed = True
                if closed:
                    _eai_platform_bootstrap.QTimer.singleShot(0, _revive_ime_after_combo)
            elif et == _eai_platform_bootstrap.QEvent.KeyPress and event.key() == _eai_platform_bootstrap.Qt.Key_Escape:
                # 搜索框自己处理 Escape；此处兜底关闭无焦点浮层
                for combo in list(_eai_platform_bootstrap._IME_OPEN_POPUPS):
                    filt = getattr(combo, "_ime_filter", None)
                    if filt is not None and filt.hasFocus():
                        continue
                    combo.hidePopup()
                _eai_platform_bootstrap.QTimer.singleShot(0, _revive_ime_after_combo)
        except Exception:
            pass
        return False


def install_chinese_ime_guards(app: _eai_platform_bootstrap.QApplication) -> None:
    filt = ImePopupDismissFilter(app)
    app.installEventFilter(filt)
    setattr(app, "_ime_popup_dismiss_filter", filt)
    for w in app.allWidgets():
        try:
            if isinstance(w, _eai_platform_bootstrap.QComboBox):
                w.setAttribute(_eai_platform_bootstrap.Qt.WA_InputMethodEnabled, False)
                w.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        except Exception:
            pass

