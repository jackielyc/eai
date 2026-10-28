import os
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import unittest
import tempfile
from pathlib import Path
from PyQt5.QtWidgets import QApplication
from grasp_task_workflow import (TASKS, TaskFlowWidget, load_tasks, parse_roles,
                                 parse_steps, save_tasks, validate_task)


class WorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): cls.app=QApplication.instance() or QApplication([])

    def test_phone_box_flow_order_and_highlight(self):
        task=TASKS['phone_box_assembly']
        self.assertEqual(task['name'],'手机盒装配')
        self.assertEqual([s['label'] for s in task['steps']],
                         ['拿起手机','放入手机盒','盖上手机盒'])
        self.assertEqual([label for _,label in task['roles']],['手机','手机盒','手机盒盖'])
        self.assertEqual(len(task['steps'][0]['actions']),6)
        widget=TaskFlowWidget();widget.set_task(task);widget.set_current(1)
        self.assertEqual(widget.current_step['label'],'放入手机盒')
        self.assertIn('ffb84d',widget._labels[1].styleSheet())
        self.assertIn('245f3b',widget._labels[0].styleSheet())
        self.assertIn('292b30',widget._labels[2].styleSheet())
        selected=[];widget.step_selected.connect(selected.append)
        widget._labels[2].clicked.emit(2)
        self.assertEqual(selected,[2])
        widget.deleteLater()

    def test_custom_task_definition_roundtrip(self):
        task=validate_task({'name':'搬运测试','roles':parse_roles('item=物品, bin=容器'),
            'steps':parse_steps('选择物品|target|item\n移动|motion|item|pregrasp\n'
                                '闭合|hand|item|close\n确认|confirm\n完成|complete')})
        self.assertEqual(task['roles'],(('item','物品'),('bin','容器')))
        self.assertFalse(task['steps'][2]['open'])
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'tasks.json'
            save_tasks({'custom':task},path)
            loaded=load_tasks(path)
        self.assertIn('phone_box_assembly',loaded)
        self.assertEqual(loaded['custom']['name'],'搬运测试')
        self.assertEqual(loaded['custom']['steps'][1]['stage'],'pregrasp')

    def test_invalid_role_reference_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'未知角色'):
            validate_task({'name':'坏任务','roles':(('a','A'),),
                           'steps':({'label':'移动','kind':'motion','role':'b','stage':'pregrasp'},)})


if __name__=='__main__': unittest.main()
