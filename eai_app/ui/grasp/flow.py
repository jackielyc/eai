"""Task flow display; domain task definitions live separately."""
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QWidget
class FlowStepLabel(QLabel):
    clicked = pyqtSignal(int)

    def __init__(self, index, text):
        super().__init__(text)
        self.index = int(index)
        self.setCursor(Qt.PointingHandCursor)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.index)
        super().mousePressEvent(event)


class TaskFlowWidget(QScrollArea):
    step_selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setMinimumHeight(70)
        self.setMaximumHeight(82)
        self._content = QWidget()
        self._layout = QHBoxLayout(self._content)
        self._layout.setContentsMargins(6, 5, 6, 5)
        self._layout.setSpacing(4)
        self.setWidget(self._content)
        self._labels = []
        self._steps = ()
        self._current = 0
        self._completed_steps = None

    def set_task(self, task):
        self._runtime_status_key = None
        while self._layout.count():
            item = self._layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._steps = tuple(task['steps'])
        self._labels = []
        for index, step in enumerate(self._steps):
            label = FlowStepLabel(index, f"{index+1}. {step['label']}")
            label.clicked.connect(self.step_selected.emit)
            label.setAlignment(Qt.AlignCenter)
            label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
            label.setMinimumWidth(max(82, len(step['label'])*16+24))
            label.setMinimumHeight(38)
            label.setToolTip('点击跳转到该流程步骤；执行期间不能跳转')
            self._layout.addWidget(label)
            self._labels.append(label)
            if index < len(self._steps)-1:
                arrow = QLabel('→')
                arrow.setStyleSheet('color:#8a8f98;font-size:18px;')
                self._layout.addWidget(arrow)
        self._layout.addStretch(1)
        self.set_current(0)

    def set_current(self, index):
        self._current = max(0, min(int(index), len(self._labels)))
        for i, label in enumerate(self._labels):
            if i != self._current and (i in self._completed_steps if self._completed_steps is not None else i < self._current):
                style = 'background:#245f3b;color:#dff7e7;border:1px solid #43a565;'
            elif i == self._current:
                style = 'background:#7a4b12;color:#fff2d6;border:2px solid #ffb84d;font-weight:600;'
            else:
                style = 'background:#292b30;color:#aeb3bb;border:1px solid #555b66;'
            label.setStyleSheet('QLabel{'+style+'border-radius:5px;padding:4px 8px;}')

    @property
    def current_step(self):
        return self._steps[self._current] if self._steps and self._current < len(self._steps) else None

    def set_completed_steps(self, indices):
        completed = set(indices)
        if completed != self._completed_steps:
            self._completed_steps = completed
            self._runtime_status_key = None

    def set_runtime_status(self, text, state='active'):
        key = (self._current, text, state)
        if getattr(self, '_runtime_status_key', None) == key:
            return
        self._runtime_status_key = key
        self.setToolTip(text)
        self.set_current(self._current)
        if self._current >= len(self._labels):
            return
        label = self._labels[self._current]
        label.setToolTip(text)
        styles = {
            'failed': 'background:#702c2c;color:#ffe1e1;border:2px solid #ff6969;',
            'paused': 'background:#494050;color:#f1e7ff;border:2px solid #bc9bdd;',
            'confirm': 'background:#194e69;color:#e1f5ff;border:2px solid #61c7ff;',
        }
        if state in styles:
            label.setStyleSheet('QLabel{'+styles[state]+'border-radius:5px;padding:4px 8px;}')

