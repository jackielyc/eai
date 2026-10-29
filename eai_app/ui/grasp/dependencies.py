"""Qt grasp control panel: explicit stages, no execution on pose arrival."""
from __future__ import annotations

from eai_app.paths import PROJECT_ROOT

import json
import copy
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
from PyQt5.QtCore import QProcess, QProcessEnvironment, QTimer, Qt
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QTextEdit, QVBoxLayout, QWidget, QGridLayout,
                             QMessageBox, QDialog, QDialogButtonBox, QFormLayout, QInputDialog)

from eai_app.domain.grasp.execution import (Sequence, JointRunner, evaluate_wrench, load_profile,
                             make_preview, read_feedback, manual_candidate)
from eai_app.domain.grasp.hand_execution import HandRunner
from eai_app.infrastructure.run_log import TaskRunLog
from eai_app.ui.grasp.runtime import GraspRuntimeMixin
from eai_app.ui.grasp.workflow import (TASKS, TaskFlowWidget, format_steps, new_task_key,
                                 parse_roles, parse_steps, reload_tasks, save_tasks,
                                 validate_task, reconcile_settings)


class TaskEditorDialog(QDialog):
    def __init__(self, source=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle('新建抓取任务')
        self.resize(650, 500)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.name_edit = QLineEdit('' if source is None else source['name']+' 副本')
        self.roles_edit = QLineEdit(
            'object=目标' if source is None else
            ', '.join(f'{key}={label}' for key, label in source['roles']))
        self.steps_edit = QTextEdit(
            '选择目标|target|object\n确认操作|confirm\n完成|complete'
            if source is None else format_steps(source))
        form.addRow('任务名称', self.name_edit)
        form.addRow('目标角色', self.roles_edit)
        form.addRow('流程步骤', self.steps_edit)
        layout.addLayout(form)
        hint = QLabel(
            '角色格式：role=显示名，多个用逗号分隔。\n'
            '步骤每行：名称|动作|角色|参数。动作支持 target、motion、hand、confirm、complete；'
            'motion 参数为 pregrasp/approach/lift，hand 参数为 open/close。'
            '高层动作支持 pick/place/align/insert/verify，自动展开为可确认的步骤。')
        hint.setWordWrap(True)
        layout.addWidget(hint)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def definition(self):
        return validate_task({'name': self.name_edit.text().strip(), 'builtin': False,
                              'roles': parse_roles(self.roles_edit.text()),
                              'steps': parse_steps(self.steps_edit.toPlainText())})


