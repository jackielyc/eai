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
        self.assertEqual(len(task['steps'][0]['actions']),7)
        self.assertEqual(task['steps'][0]['actions'][1]['kind'],'hand')
        self.assertEqual(task['steps'][1]['actions'][-1]['kind'],'confirm')
        self.assertEqual(task['steps'][2]['actions'][-1]['kind'],'confirm')
        widget=TaskFlowWidget();widget.set_task(task);widget.set_current(1)
        self.assertEqual(widget.current_step['label'],'放入手机盒')
        self.assertIn('ffb84d',widget._labels[1].styleSheet())
        self.assertIn('245f3b',widget._labels[0].styleSheet())
        self.assertIn('292b30',widget._labels[2].styleSheet())
        selected=[];widget.step_selected.connect(selected.append)
        widget._labels[2].clicked.emit(2)
        self.assertEqual(selected,[2])
        widget.deleteLater()

    def test_sequence_editor_roundtrip_and_settings_remap_after_insert(self):
        from grasp_task_workflow import format_steps,reconcile_settings
        task=TASKS['phone_box_assembly']
        restored=validate_task(dict(task,steps=parse_steps(format_steps(task))))
        self.assertEqual(restored['steps'],task['steps'])
        motion=task['steps'][0]['actions'][2]
        settings={'0:1':{'step':motion,'arm_side':'left'},'0:2':{'arm_side':'right'}}
        mapped=reconcile_settings(settings,task,allow_legacy=False)
        self.assertEqual(list(mapped),['0:2'])
        self.assertEqual(mapped['0:2']['arm_side'],'left')

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

    def test_fold_box_task_has_only_paper_box_target(self):
        task=next(task for task in TASKS.values() if task['name']=='叠纸盒')
        self.assertEqual(task['roles'],(('paper_box','纸盒'),))
        self.assertNotIn('phone',{role for role,_ in task['roles']})
        self.assertNotIn('box',{role for role,_ in task['roles']})
        referenced={step.get('role') for step in task['steps'] if step.get('role')}
        nested_referenced={action.get('role') for step in task['steps']
                           for action in step.get('actions',()) if action.get('role')}
        self.assertEqual(referenced,set())
        self.assertEqual(nested_referenced,{'paper_box'})
        self.assertEqual([step['label'] for step in task['steps']],
                         ['叠后面','叠前面','叠左面','叠右面'])
        self.assertEqual(task['definition_revision'],3)

    def test_cake_packaging_task_roles_and_flow(self):
        task=TASKS['cake_packaging']
        self.assertEqual(task['name'],'蛋糕打包')
        self.assertEqual(task['roles'],(('cake','蛋糕'),('box','蛋糕盒'),('lid','盒盖')))
        self.assertEqual([step['label'] for step in task['steps']],
                         ['装入蛋糕','盖上盒盖','确认打包'])
        self.assertEqual(len(task['steps'][0]['actions']),10)
        self.assertEqual(len(task['steps'][1]['actions']),10)
        self.assertEqual(task['steps'][2]['actions'][-1]['kind'],'complete')

    def test_test_tube_experiment_roles_and_flow(self):
        task=TASKS['test_tube_experiment']
        self.assertEqual(task['name'],'试管实验')
        self.assertEqual(task['roles'],(('tube','试管'),('station','实验位置'),('rack','试管架')))
        self.assertEqual([step['label'] for step in task['steps']],
                         ['取出试管','移至实验位置','放回试管架','确认完成'])
        self.assertEqual(task['steps'][0]['actions'][-1]['stage'],'lift')
        self.assertEqual(task['steps'][1]['actions'][2]['kind'],'confirm')
        self.assertTrue(task['steps'][2]['actions'][2]['open'])
        self.assertEqual(task['steps'][3]['actions'][-1]['kind'],'complete')

    def test_sichuan_mahjong_roles_and_flow(self):
        task=TASKS['sichuan_mahjong']
        self.assertEqual(task['name'],'四川麻将')
        self.assertEqual(task['roles'],(
            ('draw_tile','待摸牌'),('hand_slot','手牌放置位'),
            ('discard_tile','待打牌'),('discard_area','牌河位置')))
        self.assertEqual([step['label'] for step in task['steps']],
                         ['摸牌','整理与判断','打牌','确认回合'])
        self.assertEqual(len(task['steps'][0]['actions']),10)
        self.assertIn('定缺',task['steps'][1]['actions'][1]['label'])
        self.assertEqual(len(task['steps'][2]['actions']),10)
        self.assertEqual(task['steps'][3]['actions'][-1]['kind'],'complete')

    def test_optical_module_installation_roles_and_flow(self):
        task=TASKS['optical_module_installation']
        self.assertEqual(task['name'],'光模块安装')
        self.assertEqual(task['roles'],(('module','光模块'),('port','安装端口')))
        self.assertEqual([step['label'] for step in task['steps']],
                         ['准备安装位','拿起光模块','对准并插入','验证安装'])
        self.assertIn('无异物',task['steps'][0]['actions'][1]['label'])
        self.assertEqual(task['steps'][1]['actions'][-1]['stage'],'lift')
        self.assertTrue(task['steps'][2]['actions'][3]['open'])
        self.assertEqual(task['steps'][3]['actions'][-1]['kind'],'complete')

    def test_clothes_folding_roles_and_flow(self):
        task=TASKS['clothes_folding']
        self.assertEqual(task['name'],'叠衣服')
        self.assertEqual(task['roles'],(
            ('garment_center','衣服中心'),('left_edge','左侧衣边'),
            ('right_edge','右侧衣边'),('lower_edge','衣服下摆')))
        self.assertEqual([step['label'] for step in task['steps']],
                         ['展平定位','折左侧','折右侧','对折完成'])
        self.assertEqual(len(task['steps'][1]['actions']),10)
        self.assertEqual(len(task['steps'][2]['actions']),10)
        self.assertIn('重新估计',task['steps'][1]['actions'][-1]['label'])
        self.assertEqual(task['steps'][3]['actions'][-1]['kind'],'complete')


if __name__=='__main__': unittest.main()
