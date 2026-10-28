"""Persistent task definitions and a compact highlighted flow."""
import copy
import json
from pathlib import Path
import uuid
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QScrollArea, QSizePolicy, QWidget


TASK_CONFIG_PATH = Path(__file__).with_name('config') / 'grasp_tasks.json'

DEFAULT_TASKS = {
    'phone_box_assembly': {
        'name': '手机盒装配',
        'builtin': True,
        'roles': (('phone', '手机'), ('box', '手机盒'), ('lid', '手机盒盖')),
        'steps': (
            {'label': '拿起手机', 'kind': 'sequence', 'actions': (
                {'label': '选择手机', 'kind': 'target', 'role': 'phone'},
                {'label': '预抓取手机', 'kind': 'motion', 'role': 'phone', 'stage': 'pregrasp'},
                {'label': '接近手机', 'kind': 'motion', 'role': 'phone', 'stage': 'approach'},
                {'label': '闭合抓取', 'kind': 'hand', 'role': 'phone', 'open': False},
                {'label': '确认夹持', 'kind': 'confirm'},
                {'label': '抬起手机', 'kind': 'motion', 'role': 'phone', 'stage': 'lift'},
            )},
            {'label': '放入手机盒', 'kind': 'sequence', 'actions': (
                {'label': '移动至盒上方', 'kind': 'motion', 'role': 'box', 'stage': 'pregrasp'},
                {'label': '放入盒内', 'kind': 'motion', 'role': 'box', 'stage': 'approach'},
                {'label': '张开释放手机', 'kind': 'hand', 'role': 'box', 'open': True},
                {'label': '撤离手机盒', 'kind': 'motion', 'role': 'box', 'stage': 'lift'},
            )},
            {'label': '盖上手机盒', 'kind': 'sequence', 'actions': (
                {'label': '选择盒盖', 'kind': 'target', 'role': 'lid'},
                {'label': '预抓取盒盖', 'kind': 'motion', 'role': 'lid', 'stage': 'pregrasp'},
                {'label': '接近盒盖', 'kind': 'motion', 'role': 'lid', 'stage': 'approach'},
                {'label': '闭合抓取盒盖', 'kind': 'hand', 'role': 'lid', 'open': False},
                {'label': '抬起盒盖', 'kind': 'motion', 'role': 'lid', 'stage': 'lift'},
                {'label': '移动盒盖至盒上方', 'kind': 'motion', 'role': 'box', 'stage': 'pregrasp'},
                {'label': '放下盒盖', 'kind': 'motion', 'role': 'box', 'stage': 'approach'},
                {'label': '张开释放盒盖', 'kind': 'hand', 'role': 'box', 'open': True},
                {'label': '撤离盒盖', 'kind': 'motion', 'role': 'box', 'stage': 'lift'},
            )},
        ),
    },
}

VALID_KINDS = {'target', 'motion', 'hand', 'confirm', 'complete', 'sequence'}
VALID_STAGES = {'pregrasp', 'approach', 'lift'}


def validate_task(task):
    name = str(task.get('name') or '').strip()
    if not name:
        raise ValueError('任务名称不能为空')
    roles = tuple((str(key).strip(), str(label).strip()) for key, label in task.get('roles') or ())
    if not roles or any(not key or not label for key, label in roles):
        raise ValueError('任务至少需要一个有效目标角色')
    role_keys = {key for key, _ in roles}
    if len(role_keys) != len(roles):
        raise ValueError('目标角色标识不能重复')
    def normalize_step(raw, *, nested=False):
        step = dict(raw)
        step['label'] = str(step.get('label') or '').strip()
        step['kind'] = str(step.get('kind') or '').strip()
        if not step['label'] or step['kind'] not in VALID_KINDS:
            raise ValueError('流程步骤缺少名称或动作类型无效')
        if nested and step['kind'] == 'sequence':
            raise ValueError('组合步骤不能继续嵌套组合步骤')
        role = str(step.get('role') or '')
        if role and role not in role_keys:
            raise ValueError(f"步骤引用未知角色：{role}")
        if step['kind'] in ('target', 'motion', 'hand') and not role:
            raise ValueError(f"步骤“{step['label']}”需要目标角色")
        if step['kind'] == 'motion' and step.get('stage') not in VALID_STAGES:
            raise ValueError(f"步骤“{step['label']}”的运动阶段无效")
        if step['kind'] == 'hand':
            step['open'] = bool(step.get('open'))
        if step['kind'] == 'sequence':
            actions = tuple(normalize_step(action, nested=True) for action in step.get('actions') or ())
            if not actions:
                raise ValueError(f"组合步骤“{step['label']}”没有子动作")
            step['actions'] = actions
        return step
    steps = [normalize_step(raw) for raw in task.get('steps') or ()]
    if not steps:
        raise ValueError('任务至少需要一个流程步骤')
    return {'name': name, 'builtin': bool(task.get('builtin', False)),
            'roles': roles, 'steps': tuple(steps)}


def load_tasks(path=TASK_CONFIG_PATH):
    path = Path(path)
    tasks = copy.deepcopy(DEFAULT_TASKS)
    if path.is_file():
        data = json.loads(path.read_text(encoding='utf-8'))
        for key, task in (data.get('tasks') or {}).items():
            if key in DEFAULT_TASKS and DEFAULT_TASKS[key].get('builtin'):
                continue
            tasks[str(key)] = validate_task(task)
    return tasks


def save_tasks(tasks, path=TASK_CONFIG_PATH):
    normalized = {str(key): validate_task(task) for key, task in tasks.items()}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps({'version': 1, 'tasks': normalized}, ensure_ascii=False, indent=2)+'\n',
                         encoding='utf-8')
    temporary.replace(path)
    return path


def new_task_key():
    return 'task_' + uuid.uuid4().hex[:10]


def parse_roles(text):
    roles = []
    for index, token in enumerate(str(text).replace('，', ',').split(','), 1):
        token = token.strip()
        if not token:
            continue
        if '=' in token:
            key, label = (part.strip() for part in token.split('=', 1))
        else:
            key, label = f'role{index}', token
        roles.append((key, label))
    return tuple(roles)


def parse_steps(text):
    steps = []
    for line in str(text).splitlines():
        parts = [part.strip() for part in line.split('|')]
        if not parts or not parts[0]:
            continue
        kind = parts[1] if len(parts) > 1 and parts[1] else 'confirm'
        step = {'label': parts[0], 'kind': kind}
        if len(parts) > 2 and parts[2]:
            step['role'] = parts[2]
        if kind == 'motion' and len(parts) > 3:
            step['stage'] = parts[3]
        if kind == 'hand' and len(parts) > 3:
            step['open'] = parts[3].lower() in ('open', 'true', '1', '张开')
        steps.append(step)
    return tuple(steps)


def format_steps(task):
    lines = []
    for step in task['steps']:
        parts = [step['label'], step['kind'], str(step.get('role') or '')]
        if step['kind'] == 'motion':
            parts.append(step['stage'])
        elif step['kind'] == 'hand':
            parts.append('open' if step.get('open') else 'close')
        lines.append('|'.join(parts).rstrip('|'))
    return '\n'.join(lines)


TASKS = load_tasks()


def reload_tasks(path=TASK_CONFIG_PATH):
    TASKS.clear()
    TASKS.update(load_tasks(path))
    return TASKS


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

    def set_task(self, task):
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
            if i < self._current:
                style = 'background:#245f3b;color:#dff7e7;border:1px solid #43a565;'
            elif i == self._current:
                style = 'background:#7a4b12;color:#fff2d6;border:2px solid #ffb84d;font-weight:600;'
            else:
                style = 'background:#292b30;color:#aeb3bb;border:1px solid #555b66;'
            label.setStyleSheet('QLabel{'+style+'border-radius:5px;padding:4px 8px;}')

    @property
    def current_step(self):
        return self._steps[self._current] if self._steps and self._current < len(self._steps) else None

