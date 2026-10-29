"""ui.widgets.tab bar — extracted application responsibilities."""
from __future__ import annotations
from eai_app.paths import PROJECT_ROOT
from eai_app.infrastructure.module_ref import module_ref
_eai_configuration_tabs = module_ref('eai_app.configuration.tabs')
_eai_platform_bootstrap = module_ref('eai_app.platform.bootstrap')

class NestableControlTabBar(_eai_platform_bootstrap.QTabBar):
    """控制区顶栏 /「其他」子栏：拖拽排序，并可跨栏拖入、拖出「其他」。"""

    tabsReordered = _eai_platform_bootstrap.pyqtSignal()
    tabNestToMisc = _eai_platform_bootstrap.pyqtSignal(str)
    tabPopOut = _eai_platform_bootstrap.pyqtSignal(str, int)  # title, insert_at（-1=追加到「其他」旁）

    def __init__(self, role: str, parent=None) -> None:
        super().__init__(parent)
        self._role = "misc" if role == "misc" else "outer"
        self.setMovable(False)
        self.setAcceptDrops(True)
        self.setElideMode(_eai_platform_bootstrap.Qt.ElideRight)
        self._drag_start_pos: _eai_platform_bootstrap.Optional[_eai_platform_bootstrap.QPoint] = None
        self._drag_index = -1
        self.setContextMenuPolicy(_eai_platform_bootstrap.Qt.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)

    def mousePressEvent(self, event: _eai_platform_bootstrap.QMouseEvent) -> None:  # type: ignore[override]
        if event.button() == _eai_platform_bootstrap.Qt.LeftButton:
            self._drag_start_pos = _eai_platform_bootstrap.QPoint(event.pos())
            self._drag_index = self.tabAt(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: _eai_platform_bootstrap.QMouseEvent) -> None:  # type: ignore[override]
        if not (event.buttons() & _eai_platform_bootstrap.Qt.LeftButton):
            super().mouseMoveEvent(event)
            return
        if self._drag_start_pos is None or self._drag_index < 0:
            super().mouseMoveEvent(event)
            return
        if (
            (event.pos() - self._drag_start_pos).manhattanLength()
            < _eai_platform_bootstrap.QApplication.startDragDistance()
        ):
            super().mouseMoveEvent(event)
            return
        title = str(self.tabText(self._drag_index) or "").strip()
        if not title or title == "…":
            super().mouseMoveEvent(event)
            return
        drag = _eai_platform_bootstrap.QDrag(self)
        mime = _eai_platform_bootstrap.QMimeData()
        mime.setData(_eai_configuration_tabs._EAI_CONTROL_TAB_MIME, title.encode("utf-8"))
        mime.setData(_eai_configuration_tabs._EAI_CONTROL_TAB_SRC_MIME, self._role.encode("utf-8"))
        mime.setText(title)
        drag.setMimeData(mime)
        drag.exec_(_eai_platform_bootstrap.Qt.MoveAction)
        self._drag_start_pos = None
        self._drag_index = -1

    def mouseReleaseEvent(self, event: _eai_platform_bootstrap.QMouseEvent) -> None:  # type: ignore[override]
        self._drag_start_pos = None
        self._drag_index = -1
        super().mouseReleaseEvent(event)

    def dragEnterEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasFormat(_eai_configuration_tabs._EAI_CONTROL_TAB_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dragMoveEvent(self, event) -> None:  # type: ignore[override]
        if event.mimeData().hasFormat(_eai_configuration_tabs._EAI_CONTROL_TAB_MIME):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:  # type: ignore[override]
        mime = event.mimeData()
        if not mime.hasFormat(_eai_configuration_tabs._EAI_CONTROL_TAB_MIME):
            event.ignore()
            return
        title = bytes(mime.data(_eai_configuration_tabs._EAI_CONTROL_TAB_MIME)).decode("utf-8", errors="replace").strip()
        src = bytes(mime.data(_eai_configuration_tabs._EAI_CONTROL_TAB_SRC_MIME)).decode("utf-8", errors="replace").strip()
        if not title:
            event.ignore()
            return
        onto = self.tabAt(event.pos())
        onto_title = str(self.tabText(onto) or "").strip() if onto >= 0 else ""

        if self._role == "outer":
            if (
                onto_title == _eai_configuration_tabs.CONTROL_MISC_TAB_TITLE
                and title != _eai_configuration_tabs.CONTROL_MISC_TAB_TITLE
                and onto >= 0
            ):
                # 落在「其他」中部 → 收入；落在边缘 → 按排序插入
                rect = self.tabRect(onto)
                x = event.pos().x()
                margin = max(6, rect.width() // 4)
                if rect.left() + margin <= x <= rect.right() - margin:
                    self.tabNestToMisc.emit(title)
                    event.acceptProposedAction()
                    return
            insert_at = onto if onto >= 0 else self.count()
            if src == "misc":
                self.tabPopOut.emit(title, insert_at)
                event.acceptProposedAction()
                return
            if src == "outer":
                from_idx = self._index_of_title(title)
                if from_idx < 0:
                    event.ignore()
                    return
                to_idx = insert_at
                if to_idx > from_idx:
                    to_idx -= 1
                if to_idx != from_idx and 0 <= to_idx < self.count():
                    self.moveTab(from_idx, to_idx)
                    self.tabsReordered.emit()
                event.acceptProposedAction()
                return

        # misc 子栏
        if title == _eai_configuration_tabs.CONTROL_MISC_TAB_TITLE:
            event.ignore()
            return
        if src == "outer":
            self.tabNestToMisc.emit(title)
            event.acceptProposedAction()
            return
        if src == "misc":
            from_idx = self._index_of_title(title)
            if from_idx < 0:
                event.ignore()
                return
            to_idx = onto if onto >= 0 else self.count()
            if to_idx > from_idx:
                to_idx -= 1
            if to_idx != from_idx and 0 <= to_idx < self.count():
                self.moveTab(from_idx, to_idx)
                self.tabsReordered.emit()
            event.acceptProposedAction()
            return
        event.ignore()

    def _index_of_title(self, title: str) -> int:
        for i in range(self.count()):
            if self.tabText(i) == title:
                return i
        return -1

    def _on_context_menu(self, pos: _eai_platform_bootstrap.QPoint) -> None:
        idx = self.tabAt(pos)
        if idx < 0:
            return
        title = str(self.tabText(idx) or "").strip()
        if not title or title == "…":
            return
        menu = _eai_platform_bootstrap.QMenu(self)
        if self._role == "outer" and title != _eai_configuration_tabs.CONTROL_MISC_TAB_TITLE:
            act = menu.addAction(f"移入「{_eai_configuration_tabs.CONTROL_MISC_TAB_TITLE}」")
            chosen = menu.exec_(self.mapToGlobal(pos))
            if chosen is act:
                self.tabNestToMisc.emit(title)
            return
        if self._role == "misc":
            act = menu.addAction("移回顶栏")
            chosen = menu.exec_(self.mapToGlobal(pos))
            if chosen is act:
                self.tabPopOut.emit(title, -1)

