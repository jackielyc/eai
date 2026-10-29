"""ui.chat.history dialogs — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_policies = module_ref('eai_app.configuration.policies')
_eai_infrastructure_ui_storage = module_ref('eai_app.infrastructure.ui_storage')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')
_eai_ui_widgets_inputs = module_ref('eai_app.ui.widgets.inputs')

class BagelHistoryDialog(_eai_platform_bootstrap.QWidget):
    """Bagel 历史对话浮层：加载 / 改标题 / 删除。"""

    finished = _eai_platform_bootstrap.pyqtSignal(int)

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None) -> None:
        win = parent.window() if parent is not None else parent
        super().__init__(win)
        self.selected_id: str = ""
        self.setAttribute(_eai_platform_bootstrap.Qt.WA_InputMethodEnabled, True)
        self.setFocusPolicy(_eai_platform_bootstrap.Qt.StrongFocus)
        self.setObjectName("bagelHistoryMask")
        self.setStyleSheet("#bagelHistoryMask { background-color: rgba(0,0,0,160); }")
        root = _eai_platform_bootstrap.QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        panel = _eai_platform_bootstrap.QFrame()
        panel.setObjectName("bagelHistoryPanel")
        panel.setStyleSheet(
            "#bagelHistoryPanel {"
            "  background-color: #2b2b2b;"
            "  border: 1px solid #666;"
            "  border-radius: 6px;"
            "}"
        )
        panel.setMinimumSize(480, 420)
        panel.setMaximumSize(720, 560)
        root.addStretch(1)
        row = _eai_platform_bootstrap.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(panel)
        row.addStretch(1)
        root.addLayout(row)
        root.addStretch(1)

        layout = _eai_platform_bootstrap.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        title = _eai_platform_bootstrap.QLabel("Bagel 历史对话")
        title.setStyleSheet(
            f"color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; font-weight: bold; font-size: 14pt;"
        )
        layout.addWidget(title)
        tip = _eai_platform_bootstrap.QLabel(
            "选择一条历史：双击或「加载」恢复该会话的调用轮次；"
            "可改标题或删除。点空白处或「关闭」退出。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
        layout.addWidget(tip)
        self.list_widget = _eai_platform_bootstrap.QListWidget()
        self.list_widget.setAttribute(_eai_platform_bootstrap.Qt.WA_InputMethodEnabled, False)
        self.list_widget.setFocusPolicy(_eai_platform_bootstrap.Qt.ClickFocus)
        self.list_widget.setStyleSheet(
            f"QListWidget {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; color: #ddd; border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
        )
        self.list_widget.itemDoubleClicked.connect(self._on_load_clicked)
        self.list_widget.currentItemChanged.connect(self._on_current_changed)
        layout.addWidget(self.list_widget, 1)
        title_row = _eai_platform_bootstrap.QHBoxLayout()
        title_row.addWidget(_eai_platform_bootstrap.QLabel("标题"))
        self.title_edit = _eai_ui_widgets_inputs.ImeSafeLineEdit()
        self.title_edit.setPlaceholderText("选中条目后可改标题")
        title_row.addWidget(self.title_edit, stretch=1)
        self.save_title_btn = _eai_platform_bootstrap.QPushButton("保存标题")
        self.save_title_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        self.save_title_btn.clicked.connect(self._on_save_title_clicked)
        title_row.addWidget(self.save_title_btn)
        layout.addLayout(title_row)
        btn_row = _eai_platform_bootstrap.QHBoxLayout()
        self.load_btn = _eai_platform_bootstrap.QPushButton("加载")
        self.load_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        self.load_btn.clicked.connect(self._on_load_clicked)
        btn_row.addWidget(self.load_btn)
        self.delete_btn = _eai_platform_bootstrap.QPushButton("删除")
        self.delete_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        self.delete_btn.setObjectName("dangerAction")
        self.delete_btn.clicked.connect(self._on_delete_clicked)
        btn_row.addWidget(self.delete_btn)
        self._delete_armed = False
        btn_row.addStretch(1)
        close_btn = _eai_platform_bootstrap.QPushButton("关闭")
        close_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)
        self._status = _eai_platform_bootstrap.QLabel("")
        self._status.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
        layout.addWidget(self._status)
        self._panel = panel
        self._reload()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self.parentWidget() is not None:
            self.setGeometry(self.parentWidget().rect())

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if self.parentWidget() is not None:
            self.setGeometry(self.parentWidget().rect())
        self.raise_()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        if not self._panel.geometry().contains(event.pos()):
            self.reject()
            event.accept()
            return
        super().mousePressEvent(event)

    def accept(self) -> None:
        self.hide()
        self.finished.emit(int(_eai_platform_bootstrap.QDialog.Accepted))

    def reject(self) -> None:
        self.hide()
        self.finished.emit(int(_eai_platform_bootstrap.QDialog.Rejected))

    def _reload(self) -> None:
        self._delete_armed = False
        self.delete_btn.setText("删除")
        self.list_widget.clear()
        self.title_edit.clear()
        for item in _eai_infrastructure_ui_storage.list_bagel_histories():
            title = str(item.get("title") or "")
            updated = str(item.get("updated_at") or "")
            count = int(item.get("turn_count") or 0)
            label = title
            if updated:
                label += f"  ·  {updated}"
            label += f"  ·  {count} 轮"
            row = _eai_platform_bootstrap.QListWidgetItem(label)
            row.setData(_eai_platform_bootstrap.Qt.UserRole, str(item.get("id") or ""))
            row.setData(_eai_platform_bootstrap.Qt.UserRole + 1, title)
            row.setToolTip(str(item.get("path") or ""))
            self.list_widget.addItem(row)
        if self.list_widget.count() == 0:
            empty = _eai_platform_bootstrap.QListWidgetItem("（暂无 Bagel 历史）")
            empty.setFlags(_eai_platform_bootstrap.Qt.NoItemFlags)
            self.list_widget.addItem(empty)
        self._status.setText("")

    def _current_id(self) -> str:
        item = self.list_widget.currentItem()
        if item is None:
            return ""
        return str(item.data(_eai_platform_bootstrap.Qt.UserRole) or "")

    def _on_current_changed(self, current, _previous) -> None:
        self._delete_armed = False
        self.delete_btn.setText("删除")
        if current is None or not (current.flags() & _eai_platform_bootstrap.Qt.ItemIsEnabled):
            self.title_edit.clear()
            return
        self.title_edit.setText(str(current.data(_eai_platform_bootstrap.Qt.UserRole + 1) or ""))
        self.title_edit.setFocus(_eai_platform_bootstrap.Qt.OtherFocusReason)

    def _on_load_clicked(self, *_args) -> None:
        hid = self._current_id()
        if not hid:
            self._status.setText("请先选择一条对话")
            return
        self.selected_id = hid
        self.accept()

    def _on_save_title_clicked(self) -> None:
        hid = self._current_id()
        if not hid:
            self._status.setText("请先选择一条对话")
            return
        new_title = self.title_edit.text().strip()
        if not new_title:
            self._status.setText("标题不能为空")
            return
        try:
            _eai_infrastructure_ui_storage.rename_bagel_history(hid, new_title)
        except Exception as exc:
            self._status.setText(f"保存标题失败: {exc}")
            return
        self._reload()
        self._status.setText("标题已保存")

    def _on_delete_clicked(self) -> None:
        hid = self._current_id()
        if not hid:
            self._status.setText("请先选择一条对话")
            return
        if not self._delete_armed:
            self._delete_armed = True
            self.delete_btn.setText("再点确认删除")
            self._status.setText("再点一次「再点确认删除」才会删除")
            return
        try:
            _eai_infrastructure_ui_storage.delete_bagel_history(hid)
        except Exception as exc:
            self._status.setText(f"删除失败: {exc}")
            self._delete_armed = False
            self.delete_btn.setText("删除")
            return
        self._reload()
        self._status.setText("已删除")


class ChatHistoryDialog(_eai_platform_bootstrap.QWidget):
    """主窗口内居中浮层（看起来像弹窗，但不新建 X11 窗口，避免弄坏 fcitx）。"""

    finished = _eai_platform_bootstrap.pyqtSignal(int)  # QDialog.Accepted / Rejected

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None) -> None:
        # parent 必须是顶层窗口，浮层盖在其上
        win = parent.window() if parent is not None else parent
        super().__init__(win)
        self.selected_id: str = ""
        self.setAttribute(_eai_platform_bootstrap.Qt.WA_InputMethodEnabled, True)
        self.setFocusPolicy(_eai_platform_bootstrap.Qt.StrongFocus)
        self.setObjectName("historyMask")
        self.setStyleSheet("#historyMask { background-color: rgba(0,0,0,160); }")
        root = _eai_platform_bootstrap.QVBoxLayout(self)
        root.setContentsMargins(24, 24, 24, 24)
        panel = _eai_platform_bootstrap.QFrame()
        panel.setObjectName("historyPanel")
        panel.setStyleSheet(
            "#historyPanel {"
            "  background-color: #2b2b2b;"
            "  border: 1px solid #666;"
            "  border-radius: 6px;"
            "}"
        )
        panel.setMinimumSize(480, 420)
        panel.setMaximumSize(720, 560)
        root.addStretch(1)
        row = _eai_platform_bootstrap.QHBoxLayout()
        row.addStretch(1)
        row.addWidget(panel)
        row.addStretch(1)
        root.addLayout(row)
        root.addStretch(1)

        layout = _eai_platform_bootstrap.QVBoxLayout(panel)
        layout.setContentsMargins(12, 12, 12, 12)
        title = _eai_platform_bootstrap.QLabel("历史对话")
        title.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_PRIMARY}; font-weight: bold; font-size: 14pt;")
        layout.addWidget(title)
        tip = _eai_platform_bootstrap.QLabel(
            "选择一条历史对话：双击或点「加载」将把最后一轮问题放入输入框"
            "（不自动请求）；可修改标题或删除。点空白处或「关闭」退出。"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_SECONDARY};")
        layout.addWidget(tip)
        self.list_widget = _eai_platform_bootstrap.QListWidget()
        self.list_widget.setAttribute(_eai_platform_bootstrap.Qt.WA_InputMethodEnabled, False)
        self.list_widget.setFocusPolicy(_eai_platform_bootstrap.Qt.ClickFocus)
        self.list_widget.setStyleSheet(
            f"QListWidget {{ background-color: {_eai_configuration_policies.UI_BG_INPUT}; color: #ddd; border: 1px solid {_eai_configuration_policies.UI_BORDER}; }}"
        )
        self.list_widget.itemDoubleClicked.connect(self._on_load_clicked)
        self.list_widget.currentItemChanged.connect(self._on_current_changed)
        layout.addWidget(self.list_widget, 1)
        title_row = _eai_platform_bootstrap.QHBoxLayout()
        title_row.addWidget(_eai_platform_bootstrap.QLabel("标题"))
        self.title_edit = _eai_ui_widgets_inputs.ImeSafeLineEdit()
        self.title_edit.setPlaceholderText("选中条目后可改标题（可输入中文）")
        title_row.addWidget(self.title_edit, stretch=1)
        self.save_title_btn = _eai_platform_bootstrap.QPushButton("保存标题")
        self.save_title_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        self.save_title_btn.clicked.connect(self._on_save_title_clicked)
        title_row.addWidget(self.save_title_btn)
        layout.addLayout(title_row)
        btn_row = _eai_platform_bootstrap.QHBoxLayout()
        self.load_btn = _eai_platform_bootstrap.QPushButton("加载")
        self.load_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        self.load_btn.clicked.connect(self._on_load_clicked)
        btn_row.addWidget(self.load_btn)
        self.delete_btn = _eai_platform_bootstrap.QPushButton("删除")
        self.delete_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        self.delete_btn.setObjectName("dangerAction")
        self.delete_btn.clicked.connect(self._on_delete_clicked)
        btn_row.addWidget(self.delete_btn)
        self._delete_armed = False
        btn_row.addStretch(1)
        close_btn = _eai_platform_bootstrap.QPushButton("关闭")
        close_btn.setFocusPolicy(_eai_platform_bootstrap.Qt.NoFocus)
        close_btn.clicked.connect(self.reject)
        btn_row.addWidget(close_btn)
        layout.addLayout(btn_row)
        self._status = _eai_platform_bootstrap.QLabel("")
        self._status.setStyleSheet(f"color: {_eai_configuration_policies.UI_TEXT_MUTED};")
        layout.addWidget(self._status)
        self._panel = panel
        self._reload()

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        if self.parentWidget() is not None:
            self.setGeometry(self.parentWidget().rect())

    def showEvent(self, event) -> None:  # type: ignore[override]
        super().showEvent(event)
        if self.parentWidget() is not None:
            self.setGeometry(self.parentWidget().rect())
        self.raise_()

    def mousePressEvent(self, event) -> None:  # type: ignore[override]
        # 点遮罩空白关闭
        if not self._panel.geometry().contains(event.pos()):
            self.reject()
            event.accept()
            return
        super().mousePressEvent(event)

    def accept(self) -> None:
        self.hide()
        self.finished.emit(int(_eai_platform_bootstrap.QDialog.Accepted))

    def reject(self) -> None:
        self.hide()
        self.finished.emit(int(_eai_platform_bootstrap.QDialog.Rejected))

    def _reload(self) -> None:
        self._delete_armed = False
        self.delete_btn.setText("删除")
        self.list_widget.clear()
        self.title_edit.clear()
        for item in _eai_infrastructure_ui_storage.list_chat_histories():
            title = str(item.get("title") or "")
            updated = str(item.get("updated_at") or "")
            model = str(item.get("model") or "")
            count = int(item.get("message_count") or 0)
            label = f"{title}"
            if updated:
                label += f"  ·  {updated}"
            if model:
                label += f"  ·  {model}"
            label += f"  ·  {count} 条"
            row = _eai_platform_bootstrap.QListWidgetItem(label)
            row.setData(_eai_platform_bootstrap.Qt.UserRole, str(item.get("id") or ""))
            row.setData(_eai_platform_bootstrap.Qt.UserRole + 1, title)
            row.setToolTip(str(item.get("path") or ""))
            self.list_widget.addItem(row)
        if self.list_widget.count() == 0:
            empty = _eai_platform_bootstrap.QListWidgetItem("（暂无保存的对话）")
            empty.setFlags(_eai_platform_bootstrap.Qt.NoItemFlags)
            self.list_widget.addItem(empty)
        self._status.setText("")

    def _current_id(self) -> str:
        item = self.list_widget.currentItem()
        if item is None:
            return ""
        return str(item.data(_eai_platform_bootstrap.Qt.UserRole) or "")

    def _on_current_changed(self, current, _previous) -> None:
        self._delete_armed = False
        self.delete_btn.setText("删除")
        if current is None or not (current.flags() & _eai_platform_bootstrap.Qt.ItemIsEnabled):
            self.title_edit.clear()
            return
        self.title_edit.setText(str(current.data(_eai_platform_bootstrap.Qt.UserRole + 1) or ""))
        self.title_edit.setFocus(_eai_platform_bootstrap.Qt.OtherFocusReason)

    def _on_load_clicked(self, *_args) -> None:
        hid = self._current_id()
        if not hid:
            self._status.setText("请先选择一条对话")
            return
        self.selected_id = hid
        self.accept()

    def _on_save_title_clicked(self) -> None:
        hid = self._current_id()
        if not hid:
            self._status.setText("请先选择一条对话")
            return
        new_title = self.title_edit.text().strip()
        if not new_title:
            self._status.setText("标题不能为空")
            return
        try:
            _eai_infrastructure_ui_storage.rename_chat_history(hid, new_title)
        except Exception as exc:
            self._status.setText(f"保存标题失败: {exc}")
            return
        self._reload()
        self._status.setText("标题已保存")

    def _on_delete_clicked(self) -> None:
        hid = self._current_id()
        if not hid:
            self._status.setText("请先选择一条对话")
            return
        if not self._delete_armed:
            self._delete_armed = True
            self.delete_btn.setText("再点确认删除")
            self._status.setText("再点一次「再点确认删除」才会删除")
            return
        try:
            _eai_infrastructure_ui_storage.delete_chat_history(hid)
        except Exception as exc:
            self._status.setText(f"删除失败: {exc}")
            self._delete_armed = False
            self.delete_btn.setText("删除")
            return
        self._reload()
        self._status.setText("已删除")


class ChatInputEdit(_eai_platform_bootstrap.QTextEdit):
    """对话输入框：获得焦点时主动唤醒 fcitx；Enter 发送但不打断中文组字。"""

    def __init__(self, parent: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QWidget] = None) -> None:
        super().__init__(parent)
        self.setAttribute(_eai_platform_bootstrap.Qt.WA_InputMethodEnabled, True)
        self.setFocusPolicy(_eai_platform_bootstrap.Qt.StrongFocus)
        self._chat_panel: _eai_platform_bootstrap.Optional["ChatPanelWidget"] = None
        self._ime_composing = False
        self._ime_guard_until = 0.0

    def focusInEvent(self, event) -> None:  # type: ignore[override]
        super().focusInEvent(event)
        pass
        _eai_platform_bootstrap._IME_LAST_TEXT_WIDGET = self
        _eai_ui_widgets_inputs._notify_ime_widget(self)

    def inputMethodEvent(self, event) -> None:  # type: ignore[override]
        # 必须在此跟踪组字；用 eventFilter 拦截 Enter 容易抢掉 fcitx 上屏键
        try:
            preedit = bool(event.preeditString())
            commit = bool(event.commitString())
            self._ime_composing = preedit
            if preedit:
                self._ime_guard_until = _eai_platform_bootstrap.time.monotonic() + 0.6
            elif commit:
                # 提交后短保护，避免紧随其后的 Enter 被当成「发送」
                self._ime_guard_until = _eai_platform_bootstrap.time.monotonic() + 0.2
        except Exception:
            pass
        super().inputMethodEvent(event)

    def keyPressEvent(self, event) -> None:  # type: ignore[override]
        if event.key() in (_eai_platform_bootstrap.Qt.Key_Return, _eai_platform_bootstrap.Qt.Key_Enter) and not (
            event.modifiers() & _eai_platform_bootstrap.Qt.ShiftModifier
        ):
            if self._ime_composing or _eai_platform_bootstrap.time.monotonic() < self._ime_guard_until:
                super().keyPressEvent(event)
                return
            try:
                app = _eai_platform_bootstrap.QApplication.instance()
                im = app.inputMethod() if app is not None else None
                if im is not None and im.isVisible():
                    super().keyPressEvent(event)
                    return
            except Exception:
                pass
            panel = self._chat_panel
            if panel is not None:
                if _eai_platform_bootstrap.time.monotonic() < getattr(panel, "_suppress_send_until", 0.0):
                    # 历史加载后短暂不发送，交给文本框（换行/IME）
                    super().keyPressEvent(event)
                    return
                panel._on_send_clicked()
                event.accept()
                return
        super().keyPressEvent(event)

    def _notify_input_method(self, *, full_query: bool = False) -> None:
        _eai_ui_widgets_inputs._notify_ime_widget(self, full_query=full_query)

