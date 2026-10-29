"""Persistent task definitions and a compact highlighted flow."""

from eai_app.paths import PROJECT_ROOT
import copy
import json
from pathlib import Path
import uuid


TASK_CONFIG_PATH = (PROJECT_ROOT / 'config') / 'grasp_tasks.json'

DEFAULT_TASKS = {
    'phone_box_assembly': {
        'name': '手机盒装配',
        'builtin': True,
        'definition_revision': 4,
        'program_type': 'phone_assembly',
        'roles': (('phone', '手机'), ('box', '手机盒'), ('lid', '手机盒盖')),
        'steps': (
            {'label': '拿起手机', 'kind': 'sequence', 'actions': (
                {'label': '选择手机', 'kind': 'target', 'role': 'phone'},
                {'label': '张开手准备抓取手机', 'kind': 'hand', 'role': 'phone', 'open': True},
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
                {'label': '确认手机已正确放入盒内', 'kind': 'confirm'},
            )},
            {'label': '盖上手机盒', 'kind': 'sequence', 'actions': (
                {'label': '选择盒盖', 'kind': 'target', 'role': 'lid'},
                {'label': '张开手准备抓取盒盖', 'kind': 'hand', 'role': 'lid', 'open': True},
                {'label': '预抓取盒盖', 'kind': 'motion', 'role': 'lid', 'stage': 'pregrasp'},
                {'label': '接近盒盖', 'kind': 'motion', 'role': 'lid', 'stage': 'approach'},
                {'label': '闭合抓取盒盖', 'kind': 'hand', 'role': 'lid', 'open': False},
                {'label': '确认盒盖夹持稳定', 'kind': 'confirm'},
                {'label': '抬起盒盖', 'kind': 'motion', 'role': 'lid', 'stage': 'lift'},
                {'label': '移动盒盖至盒上方', 'kind': 'motion', 'role': 'box', 'stage': 'pregrasp'},
                {'label': '放下盒盖', 'kind': 'motion', 'role': 'box', 'stage': 'approach'},
                {'label': '张开释放盒盖', 'kind': 'hand', 'role': 'box', 'open': True},
                {'label': '撤离盒盖', 'kind': 'motion', 'role': 'box', 'stage': 'lift'},
                {'label': '确认盒盖已对齐并盖合', 'kind': 'confirm'},
            )},
        ),
    },
    'cake_packaging': {
        'name': '蛋糕打包',
        'builtin': True,
        'definition_revision': 1,
        'roles': (('cake', '蛋糕'), ('box', '蛋糕盒'), ('lid', '盒盖')),
        'steps': (
            {'label': '装入蛋糕', 'kind': 'sequence', 'actions': (
                {'label': '选择蛋糕', 'kind': 'target', 'role': 'cake'},
                {'label': '预抓取蛋糕', 'kind': 'motion', 'role': 'cake', 'stage': 'pregrasp'},
                {'label': '接近蛋糕', 'kind': 'motion', 'role': 'cake', 'stage': 'approach'},
                {'label': '闭合抓取蛋糕', 'kind': 'hand', 'role': 'cake', 'open': False},
                {'label': '确认蛋糕夹持稳定', 'kind': 'confirm'},
                {'label': '抬起蛋糕', 'kind': 'motion', 'role': 'cake', 'stage': 'lift'},
                {'label': '移动到蛋糕盒上方', 'kind': 'motion', 'role': 'box', 'stage': 'pregrasp'},
                {'label': '放入蛋糕盒', 'kind': 'motion', 'role': 'box', 'stage': 'approach'},
                {'label': '张开释放蛋糕', 'kind': 'hand', 'role': 'box', 'open': True},
                {'label': '撤离蛋糕盒', 'kind': 'motion', 'role': 'box', 'stage': 'lift'},
            )},
            {'label': '盖上盒盖', 'kind': 'sequence', 'actions': (
                {'label': '选择盒盖', 'kind': 'target', 'role': 'lid'},
                {'label': '预抓取盒盖', 'kind': 'motion', 'role': 'lid', 'stage': 'pregrasp'},
                {'label': '接近盒盖', 'kind': 'motion', 'role': 'lid', 'stage': 'approach'},
                {'label': '闭合抓取盒盖', 'kind': 'hand', 'role': 'lid', 'open': False},
                {'label': '确认盒盖夹持稳定', 'kind': 'confirm'},
                {'label': '抬起盒盖', 'kind': 'motion', 'role': 'lid', 'stage': 'lift'},
                {'label': '移动盒盖至包装盒上方', 'kind': 'motion', 'role': 'box', 'stage': 'pregrasp'},
                {'label': '放下盒盖', 'kind': 'motion', 'role': 'box', 'stage': 'approach'},
                {'label': '张开释放盒盖', 'kind': 'hand', 'role': 'box', 'open': True},
                {'label': '撤离盒盖', 'kind': 'motion', 'role': 'box', 'stage': 'lift'},
            )},
            {'label': '确认打包', 'kind': 'sequence', 'actions': (
                {'label': '检查蛋糕位于盒内', 'kind': 'confirm'},
                {'label': '检查盒盖已对齐', 'kind': 'confirm'},
                {'label': '确认蛋糕打包完成', 'kind': 'complete'},
            )},
        ),
    },
    'test_tube_experiment': {
        'name': '试管实验',
        'builtin': True,
        'definition_revision': 1,
        'roles': (('tube', '试管'), ('station', '实验位置'), ('rack', '试管架')),
        'steps': (
            {'label': '取出试管', 'kind': 'sequence', 'actions': (
                {'label': '选择试管', 'kind': 'target', 'role': 'tube'},
                {'label': '预抓取试管', 'kind': 'motion', 'role': 'tube', 'stage': 'pregrasp'},
                {'label': '接近试管', 'kind': 'motion', 'role': 'tube', 'stage': 'approach'},
                {'label': '闭合抓取试管', 'kind': 'hand', 'role': 'tube', 'open': False},
                {'label': '确认试管夹持稳定', 'kind': 'confirm'},
                {'label': '垂直抬起试管', 'kind': 'motion', 'role': 'tube', 'stage': 'lift'},
            )},
            {'label': '移至实验位置', 'kind': 'sequence', 'actions': (
                {'label': '移动到实验位置上方', 'kind': 'motion', 'role': 'station', 'stage': 'pregrasp'},
                {'label': '下降到实验位置', 'kind': 'motion', 'role': 'station', 'stage': 'approach'},
                {'label': '人工完成加样或观察', 'kind': 'confirm'},
                {'label': '离开实验位置', 'kind': 'motion', 'role': 'station', 'stage': 'lift'},
            )},
            {'label': '放回试管架', 'kind': 'sequence', 'actions': (
                {'label': '移动到试管架上方', 'kind': 'motion', 'role': 'rack', 'stage': 'pregrasp'},
                {'label': '插回试管架', 'kind': 'motion', 'role': 'rack', 'stage': 'approach'},
                {'label': '张开释放试管', 'kind': 'hand', 'role': 'rack', 'open': True},
                {'label': '撤离试管架', 'kind': 'motion', 'role': 'rack', 'stage': 'lift'},
            )},
            {'label': '确认完成', 'kind': 'sequence', 'actions': (
                {'label': '确认试管已稳定放置', 'kind': 'confirm'},
                {'label': '确认实验区域无遗留物', 'kind': 'confirm'},
                {'label': '确认试管实验完成', 'kind': 'complete'},
            )},
        ),
    },
    'sichuan_mahjong': {
        'name': '四川麻将',
        'builtin': True,
        'definition_revision': 1,
        'roles': (('draw_tile', '待摸牌'), ('hand_slot', '手牌放置位'),
                  ('discard_tile', '待打牌'), ('discard_area', '牌河位置')),
        'steps': (
            {'label': '摸牌', 'kind': 'sequence', 'actions': (
                {'label': '选择待摸牌', 'kind': 'target', 'role': 'draw_tile'},
                {'label': '预抓取待摸牌', 'kind': 'motion', 'role': 'draw_tile', 'stage': 'pregrasp'},
                {'label': '接近待摸牌', 'kind': 'motion', 'role': 'draw_tile', 'stage': 'approach'},
                {'label': '闭合夹取麻将牌', 'kind': 'hand', 'role': 'draw_tile', 'open': False},
                {'label': '确认麻将牌夹持稳定', 'kind': 'confirm'},
                {'label': '抬起麻将牌', 'kind': 'motion', 'role': 'draw_tile', 'stage': 'lift'},
                {'label': '移动到手牌位置上方', 'kind': 'motion', 'role': 'hand_slot', 'stage': 'pregrasp'},
                {'label': '放到手牌位置', 'kind': 'motion', 'role': 'hand_slot', 'stage': 'approach'},
                {'label': '张开释放麻将牌', 'kind': 'hand', 'role': 'hand_slot', 'open': True},
                {'label': '撤离手牌位置', 'kind': 'motion', 'role': 'hand_slot', 'stage': 'lift'},
            )},
            {'label': '整理与判断', 'kind': 'sequence', 'actions': (
                {'label': '确认新牌已进入手牌区', 'kind': 'confirm'},
                {'label': '人工判断定缺、碰、杠或胡', 'kind': 'confirm'},
                {'label': '确认本轮待打牌', 'kind': 'confirm'},
            )},
            {'label': '打牌', 'kind': 'sequence', 'actions': (
                {'label': '选择待打牌', 'kind': 'target', 'role': 'discard_tile'},
                {'label': '预抓取待打牌', 'kind': 'motion', 'role': 'discard_tile', 'stage': 'pregrasp'},
                {'label': '接近待打牌', 'kind': 'motion', 'role': 'discard_tile', 'stage': 'approach'},
                {'label': '闭合夹取待打牌', 'kind': 'hand', 'role': 'discard_tile', 'open': False},
                {'label': '确认待打牌夹持稳定', 'kind': 'confirm'},
                {'label': '抬起待打牌', 'kind': 'motion', 'role': 'discard_tile', 'stage': 'lift'},
                {'label': '移动到牌河上方', 'kind': 'motion', 'role': 'discard_area', 'stage': 'pregrasp'},
                {'label': '放入牌河', 'kind': 'motion', 'role': 'discard_area', 'stage': 'approach'},
                {'label': '张开释放待打牌', 'kind': 'hand', 'role': 'discard_area', 'open': True},
                {'label': '撤离牌河', 'kind': 'motion', 'role': 'discard_area', 'stage': 'lift'},
            )},
            {'label': '确认回合', 'kind': 'sequence', 'actions': (
                {'label': '确认手牌数量和排列', 'kind': 'confirm'},
                {'label': '确认出牌位于正确牌河位置', 'kind': 'confirm'},
                {'label': '确认本回合操作完成', 'kind': 'complete'},
            )},
        ),
    },
    'optical_module_installation': {
        'name': '光模块安装',
        'builtin': True,
        'definition_revision': 1,
        'roles': (('module', '光模块'), ('port', '安装端口')),
        'steps': (
            {'label': '准备安装位', 'kind': 'sequence', 'actions': (
                {'label': '选择安装端口', 'kind': 'target', 'role': 'port'},
                {'label': '确认端口无异物且方向正确', 'kind': 'confirm'},
                {'label': '确认光模块保护帽和拉环状态', 'kind': 'confirm'},
            )},
            {'label': '拿起光模块', 'kind': 'sequence', 'actions': (
                {'label': '选择光模块', 'kind': 'target', 'role': 'module'},
                {'label': '预抓取光模块', 'kind': 'motion', 'role': 'module', 'stage': 'pregrasp'},
                {'label': '接近光模块', 'kind': 'motion', 'role': 'module', 'stage': 'approach'},
                {'label': '闭合夹取光模块', 'kind': 'hand', 'role': 'module', 'open': False},
                {'label': '确认光模块夹持稳定', 'kind': 'confirm'},
                {'label': '抬起光模块', 'kind': 'motion', 'role': 'module', 'stage': 'lift'},
            )},
            {'label': '对准并插入', 'kind': 'sequence', 'actions': (
                {'label': '移动到安装端口前方', 'kind': 'motion', 'role': 'port', 'stage': 'pregrasp'},
                {'label': '沿端口方向插入光模块', 'kind': 'motion', 'role': 'port', 'stage': 'approach'},
                {'label': '确认光模块完全到位', 'kind': 'confirm'},
                {'label': '张开释放光模块', 'kind': 'hand', 'role': 'port', 'open': True},
                {'label': '撤离安装端口', 'kind': 'motion', 'role': 'port', 'stage': 'lift'},
            )},
            {'label': '验证安装', 'kind': 'sequence', 'actions': (
                {'label': '确认锁扣或拉环已锁定', 'kind': 'confirm'},
                {'label': '确认模块无倾斜和松动', 'kind': 'confirm'},
                {'label': '确认光模块安装完成', 'kind': 'complete'},
            )},
        ),
    },
    'clothes_folding': {
        'name': '叠衣服',
        'builtin': True,
        'definition_revision': 1,
        'roles': (('garment_center', '衣服中心'), ('left_edge', '左侧衣边'),
                  ('right_edge', '右侧衣边'), ('lower_edge', '衣服下摆')),
        'steps': (
            {'label': '展平定位', 'kind': 'sequence', 'actions': (
                {'label': '选择衣服中心', 'kind': 'target', 'role': 'garment_center'},
                {'label': '确认衣服已展平且无遮挡', 'kind': 'confirm'},
                {'label': '确认左右衣边和下摆目标', 'kind': 'confirm'},
            )},
            {'label': '折左侧', 'kind': 'sequence', 'actions': (
                {'label': '选择左侧衣边', 'kind': 'target', 'role': 'left_edge'},
                {'label': '预抓取左侧衣边', 'kind': 'motion', 'role': 'left_edge', 'stage': 'pregrasp'},
                {'label': '接近左侧衣边', 'kind': 'motion', 'role': 'left_edge', 'stage': 'approach'},
                {'label': '闭合夹取左侧衣边', 'kind': 'hand', 'role': 'left_edge', 'open': False},
                {'label': '抬起左侧衣边', 'kind': 'motion', 'role': 'left_edge', 'stage': 'lift'},
                {'label': '移动到衣服中心上方', 'kind': 'motion', 'role': 'garment_center', 'stage': 'pregrasp'},
                {'label': '放下左侧衣边', 'kind': 'motion', 'role': 'garment_center', 'stage': 'approach'},
                {'label': '张开释放左侧衣边', 'kind': 'hand', 'role': 'garment_center', 'open': True},
                {'label': '撤离并确认左侧折叠', 'kind': 'motion', 'role': 'garment_center', 'stage': 'lift'},
                {'label': '确认衣服形状并按需重新估计', 'kind': 'confirm'},
            )},
            {'label': '折右侧', 'kind': 'sequence', 'actions': (
                {'label': '选择右侧衣边', 'kind': 'target', 'role': 'right_edge'},
                {'label': '预抓取右侧衣边', 'kind': 'motion', 'role': 'right_edge', 'stage': 'pregrasp'},
                {'label': '接近右侧衣边', 'kind': 'motion', 'role': 'right_edge', 'stage': 'approach'},
                {'label': '闭合夹取右侧衣边', 'kind': 'hand', 'role': 'right_edge', 'open': False},
                {'label': '抬起右侧衣边', 'kind': 'motion', 'role': 'right_edge', 'stage': 'lift'},
                {'label': '移动到衣服中心上方', 'kind': 'motion', 'role': 'garment_center', 'stage': 'pregrasp'},
                {'label': '放下右侧衣边', 'kind': 'motion', 'role': 'garment_center', 'stage': 'approach'},
                {'label': '张开释放右侧衣边', 'kind': 'hand', 'role': 'garment_center', 'open': True},
                {'label': '撤离并确认右侧折叠', 'kind': 'motion', 'role': 'garment_center', 'stage': 'lift'},
                {'label': '确认衣服形状并按需重新估计', 'kind': 'confirm'},
            )},
            {'label': '对折完成', 'kind': 'sequence', 'actions': (
                {'label': '选择衣服下摆', 'kind': 'target', 'role': 'lower_edge'},
                {'label': '预抓取衣服下摆', 'kind': 'motion', 'role': 'lower_edge', 'stage': 'pregrasp'},
                {'label': '接近衣服下摆', 'kind': 'motion', 'role': 'lower_edge', 'stage': 'approach'},
                {'label': '闭合夹取衣服下摆', 'kind': 'hand', 'role': 'lower_edge', 'open': False},
                {'label': '抬起衣服下摆', 'kind': 'motion', 'role': 'lower_edge', 'stage': 'lift'},
                {'label': '移动到衣服中心上方', 'kind': 'motion', 'role': 'garment_center', 'stage': 'pregrasp'},
                {'label': '放下衣服下摆', 'kind': 'motion', 'role': 'garment_center', 'stage': 'approach'},
                {'label': '张开释放衣服', 'kind': 'hand', 'role': 'garment_center', 'open': True},
                {'label': '撤离衣服', 'kind': 'motion', 'role': 'garment_center', 'stage': 'lift'},
                {'label': '确认衣服折叠完成', 'kind': 'complete'},
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
        from eai_app.domain.tasks.world import expand_skill
        step = expand_skill(dict(raw))
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
    result={'name': name, 'builtin': bool(task.get('builtin', False)),
            'definition_revision': max(1, int(task.get('definition_revision', 1))),
            'roles': roles, 'steps': tuple(steps)}
    if task.get('program_type'):
        if task['program_type']!='phone_assembly' or not {'phone','box','lid'}.issubset(role_keys) or len(steps)!=3:
            raise ValueError('手机装配程序需要phone/box/lid角色和三个高层流程')
        result['program_type']=task['program_type']
    return result


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


def reconcile_settings(settings, task, allow_legacy=True):
    """Match saved action semantics, not shifted ordinal positions after edits."""
    fields=('label','kind','role','stage','open','operation')
    def signature(step):return json.dumps({k:step[k] for k in fields if k in step},sort_keys=True)
    locations={}
    for index,top in enumerate(task['steps']):
        for substep,action in enumerate(top.get('actions') or (top,)):
            locations.setdefault(signature(action),[]).append(f'{index}:{substep}')
    result={}
    for key,value in settings.items():
        old=value.get('step')
        if old:
            matches=locations.get(signature(old),[])
            if len(matches)==1:result[matches[0]]=copy.deepcopy(value)
        elif allow_legacy:result[key]=copy.deepcopy(value)
    return result


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
        nested=line.lstrip().startswith('>')
        if nested:line=line.lstrip()[1:]
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
        if len(parts)>4 and parts[4]:step['operation']=parts[4]
        if nested:
            if not steps or steps[-1]['kind']!='sequence':raise ValueError('子步骤必须位于 sequence 后面')
            steps[-1].setdefault('actions',[]).append(step)
        else:steps.append(step)
    return tuple(steps)


def format_steps(task):
    lines = []
    def emit(step,prefix=''):
        kind=step.get('skill',step['kind'])
        parts = [step['label'], kind, str(step.get('role') or '')]
        if step['kind'] == 'motion':
            parts.append(step['stage'])
        elif step['kind'] == 'hand':
            parts.append('open' if step.get('open') else 'close')
        if step.get('operation'):
            while len(parts)<4:parts.append('')
            parts.append(step['operation'])
        lines.append(prefix+'|'.join(parts).rstrip('|'))
        if kind=='sequence':
            for action in step['actions']:emit(action,'> ')
    for step in task['steps']:emit(step)
    return '\n'.join(lines)


TASKS = load_tasks()


def reload_tasks(path=TASK_CONFIG_PATH):
    TASKS.clear()
    TASKS.update(load_tasks(path))
    return TASKS


