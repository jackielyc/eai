"""GUI-owned code watcher: runs between event callbacks, never in a robot worker."""
from PyQt5.QtCore import QObject, QTimer
from PyQt5.QtWidgets import QAction, QMessageBox
from eai_app.paths import PROJECT_ROOT
from eai_app.development.reloader import SourceReloader

_controller = None


def assert_sources_current():
    """Reject a new plan/execution when disk and the running GUI disagree."""
    if _controller is not None and _controller.reloader.changed_paths():
        raise ValueError('代码已修改但尚未应用，请查看「开发 → 热更新状态」；完成更新或重启后重新规划')


class DevelopmentController(QObject):
    def __init__(self, window):
        super().__init__(window)
        self.window = window
        self.reloader = SourceReloader(PROJECT_ROOT / 'eai_app', self.busy_reason)
        self.message = '支持兼容方法体更新；布局、签名、导入及 ROS 生命周期修改需要重启。'
        self._previous = None
        self._pending_signature = None
        menu = window.menuBar().addMenu('开发')
        self.automatic = QAction('空闲时自动应用代码更改', self, checkable=True)
        self.automatic.setChecked(True)
        menu.addAction(self.automatic)
        action = menu.addAction('立即检查/应用代码更改')
        action.setShortcut('Ctrl+Shift+R')
        action.triggered.connect(lambda: self.check(manual=True))
        menu.addAction('热更新状态').triggered.connect(self.show_status)
        self.timer = QTimer(self)
        self.timer.setInterval(1500)
        self.timer.timeout.connect(self.poll)
        self.timer.start()

    def busy_reason(self):
        panel = getattr(self.window, 'grasp_execution_panel', None)
        if panel:
            for name in ('runner', 'hand_runner', 'process', '_program_active', '_restoring',
                         '_workflow_auto_running', '_workflow_auto_paused', '_workflow_waiting',
                         '_sim_reserved', '_unverified_grips', 'pending_attachment',
                         '_workflow_confirmation', '_perception_pending'):
                if getattr(panel, name, False):
                    return '抓取/规划/恢复或暂停中的任务仍占用控制'
            if getattr(getattr(panel, 'sequence', None), 'state', 'idle') in (
                    'executing', 'stopping', 'stop_unconfirmed', 'planning', 'await_open',
                    'await_grasp', 'opened', 'at_pregrasp', 'at_grasp', 'grasp_confirmed'):
                return '分段任务尚未结束或停止尚未确认'
            client = getattr(getattr(panel, 'phone_program_panel', None), 'client', None)
            if client and client.process.poll() is None:
                return '装配仿真进程仍在运行，请先取消并等待退出'
            if getattr(getattr(panel, 'world', None), 'attached', {}):
                return '手中仍有已绑定物体'
            if panel.other_busy():
                return '手动移动、文本队列或遥控仍在运行'
        node = getattr(self.window, 'node', None)
        if node and (getattr(node, '_grasp_execution_reserved', False) or node.is_slow_motion_busy()):
            return '机器人控制仍被占用'
        for name, value in vars(self.window).items():
            if name.endswith('_busy') and value is True:
                return f'后台任务 {name} 尚未结束'
        return ''

    def poll(self):
        if not self.automatic.isChecked():
            return
        # Two identical scans debounce editors that truncate before rewriting.
        try:
            paths = self.reloader.changed_paths()
            signature = tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) if p.exists()
                              else (str(p), None, None) for p in paths)
            if signature and signature == self._pending_signature:
                self.check()
            self._pending_signature = signature
        except OSError as exc:
            self.message = str(exc)

    def check(self, manual=False):
        try:
            status, message = self.reloader.check()
        except Exception as exc:
            status, message = 'error', str(exc)
        labels = {'applied': '已应用', 'restart': '需要重启', 'error': '更新失败，保留原代码',
                  'deferred': '更新待处理', 'unchanged': '无变化'}
        self.message = labels[status] + '：' + message
        if status == 'applied':
            panel = getattr(self.window, 'grasp_execution_panel', None)
            if panel:
                panel.invalidate_preview()
                panel.message = '代码已更新，执行前请重新预览/规划。'
                panel.refresh()
            self.window.update()
        if self.message != self._previous and status != 'unchanged':
            self.window.statusBar().showMessage(self.message, 15000)
            self._previous = self.message
        if manual:
            self.show_status()

    def show_status(self):
        QMessageBox.information(self.window, '代码热更新', self.message)


def install_development_tools(window):
    global _controller
    existing = getattr(window, '_development_controller', None)
    if existing is None:
        existing = DevelopmentController(window)
        window._development_controller = existing
    _controller = existing
    from eai_app.development.source_guard import install_check
    install_check(assert_sources_current)
    return existing
