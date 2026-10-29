"""Versioned collision scene and full robot geometry, using Pinocchio/Coal."""

from eai_app.paths import PROJECT_ROOT
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import xml.etree.ElementTree as ET
import numpy as np

from eai_app.domain.geometry.transforms import validate_transform
from eai_app.domain.grasp.execution import fingerprint


def load_scene(path, hardware=False):
    scene = json.loads(Path(path).read_text())
    if scene.get('version') != 1 or scene.get('base_frame') != 'base_link':
        raise ValueError('碰撞场景必须是版本1、base_link坐标系')
    if hardware and scene.get('calibrated') is not True:
        raise ValueError('桌面/环境碰撞场景待标定，请在场景配置中填写实测几何并确认')
    names = set()
    for item in scene.get('world', []):
        name = str(item['id'])
        if not name or name in names:
            raise ValueError('场景物体ID为空或重复')
        names.add(name)
        validate_transform(item['T_base_object'])
        size = np.asarray(item['size_m'], float)
        if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0):
            raise ValueError('场景包围盒尺寸必须是3个有限正数')
    for rule in scene.get('contact_rules',[]):
        if not 0<float(rule.get('max_penetration_m',0))<=.005 or not 0<=float(rule.get('start_fraction',1))<=1:
            raise ValueError('接触规则穿透阈值需在0~5mm，路径起点比例需在0~1')
        if not rule.get('links') or any(not name.startswith(('hand1_','hand2_')) for name in rule['links']):
            raise ValueError('目标接触规则只能指定手部连杆，不能忽略手臂/相机/支架碰撞')
    return scene


def scene_digest(scene):
    return fingerprint(scene)


def geometry_digest(xml):
    digest=hashlib.sha256(xml.encode())
    for filename in sorted({mesh.get('filename') for mesh in ET.fromstring(xml).findall('.//collision//mesh')}):
        digest.update(filename.encode())
        with open(filename,'rb') as stream:
            while True:
                block=stream.read(1024*1024)
                if not block:break
                digest.update(block)
    return digest.hexdigest()


def augmented_urdf(urdf_path, accessory_path):
    """Use the same accessory chains as the viewer; no extra wrist DOF."""
    root = ET.parse(urdf_path).getroot()
    # Match the viewer: these are obsolete duplicate wrist shells, not real parts.
    for link in root.findall('link'):
        if link.get('name') in ('Link_l_ext','Link_r_ext'):
            for geometry in list(link.findall('collision'))+list(link.findall('visual')):link.remove(geometry)
    package_root = Path(urdf_path).resolve().parents[3]
    for mesh in root.findall('.//mesh'):
        name = mesh.get('filename')
        mesh.set('filename', str(package_root/name[len('package://'):]) if name.startswith('package://')
                 else str(Path(urdf_path).parent/name))
    if accessory_path is False:
        return ET.tostring(root,encoding='unicode')
    accessory = ET.parse(accessory_path).getroot()
    prefixes = ('hand1_', 'hand2_', 'wrist_connector_', 'wrist_camera_')
    for link in accessory.findall('link'):
        if link.get('name').startswith(prefixes):
            link = copy.deepcopy(link)
            for mesh in link.findall('.//mesh'):
                mesh.set('filename', str(Path(accessory_path).parent/mesh.get('filename')))
            root.append(link)
    for joint in accessory.findall('joint'):
        if joint.find('child').get('link').startswith(prefixes):
            joint = copy.deepcopy(joint)
            parent = joint.find('parent')
            parent.set('link', {'arm1_link_7':'Link7_l','arm2_link_7':'Link7_r'}.get(parent.get('link'),parent.get('link')))
            root.append(joint)
    return ET.tostring(root, encoding='unicode')


class CollisionScene:
    def __init__(self, urdf_path, scene, hands=None, attachments=(), accessory_path=None, stage=''):
        from eai_app.planning.arm_ik import _JOINT_NAMES  # selects the matching Pinocchio ABI
        import pinocchio as pin
        import hppfcl as fcl
        self.pin, self.fcl, self.scene = pin, fcl, copy.deepcopy(scene)
        self.stage=stage;self.fraction=0.
        if accessory_path is None:accessory_path=PROJECT_ROOT.parent/'psi-policy/assets/A2D_RuiYan_D405/A2D_RuiYan_D405.urdf'
        xml = augmented_urdf(urdf_path, accessory_path)
        self.geometry_hash = geometry_digest(xml)
        with tempfile.NamedTemporaryFile(suffix='.urdf',mode='w') as stream:
            stream.write(xml); stream.flush()
            self.model = pin.buildModelFromUrdf(stream.name)
            self.geometry = pin.buildGeomFromUrdf(self.model,stream.name,pin.GeometryType.COLLISION)
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        self.hand_indices=[]
        self.indices = [self.model.joints[self.model.getJointId(n)].idx_q for n in _JOINT_NAMES]
        motor_names = {'1_1':0,'1_2':1,'1_3':1,'2_1':2,'2_2':2,'3_1':3,'3_2':3,'4_1':4,'4_2':4,'5_1':5,'5_2':5}
        for joint in ET.fromstring(xml).findall('joint'):
            child = joint.find('child').get('link')
            if not child.startswith(('hand1_', 'hand2_')) or joint.get('type')=='fixed':
                continue
            suffix = child.rsplit('link_',1)[-1]
            if suffix not in motor_names:
                raise ValueError('未知手部关节映射：'+child)
            side = 'left' if child.startswith('hand1_') else 'right'
            values = (hands or {}).get(side)
            if values is None or len(values)!=6:
                raise ValueError('碰撞检查缺少双手实际姿态')
            value = float(values[motor_names[suffix]])
            if not np.isfinite(value) or not 0<=value<=1:
                raise ValueError('手部实际姿态无效')
            index = self.model.joints[self.model.getJointId(joint.get('name'))].idx_q
            self.hand_indices.append((index,side,motor_names[suffix]))
            self.q[index] = self.model.lowerPositionLimit[index]+value*(self.model.upperPositionLimit[index]-self.model.lowerPositionLimit[index])
        self.world_ids, self.attachment_ids = set(), set()
        self.touch = {}
        for item in scene.get('world', []):
            T = validate_transform(item['T_base_object'])
            obj = pin.GeometryObject('world:'+item['id'],0,pin.SE3(T[:3,:3],T[:3,3]),fcl.Box(*item['size_m']))
            self.world_ids.add(self.geometry.addGeometryObject(obj))
        for item in attachments:
            frame = self.model.getFrameId(item['link'])
            if frame >= len(self.model.frames):
                raise ValueError('附着物连接帧不存在')
            parent = self.model.frames[frame]
            parent_joint=parent.parentJoint if hasattr(parent,'parentJoint') else parent.parent
            T = validate_transform(item['T_link_object'])
            obj = pin.GeometryObject('attached:'+item['id'],parent_joint,
                parent.placement*pin.SE3(T[:3,:3],T[:3,3]),fcl.Box(*item['size_m']))
            index = self.geometry.addGeometryObject(obj)
            self.attachment_ids.add(index)
            self.touch[index] = set(item.get('touch_links', []))
        allowed = {frozenset(pair) for pair in scene.get('allowed_pairs', [])}
        for a in range(len(self.geometry.geometryObjects)):
            ga = self.geometry.geometryObjects[a]
            for b in range(a+1,len(self.geometry.geometryObjects)):
                gb = self.geometry.geometryObjects[b]
                if a in self.world_ids and b in self.world_ids:
                    continue
                names = frozenset((ga.name,gb.name))
                if names in allowed:
                    continue
                if not ({a,b}&(self.world_ids|self.attachment_ids)):
                    ja,jb=ga.parentJoint,gb.parentJoint
                    if ja==jb or self.model.parents[ja]==jb or self.model.parents[jb]==ja:
                        continue  # mechanically connected bodies
                if any(other.name.rsplit('_',1)[0] in self.touch.get(att,set())
                       for att,other in ((a,gb),(b,ga))):
                    continue
                self.geometry.addCollisionPair(pin.CollisionPair(a,b))
        self.geometry_data = pin.GeometryData(self.geometry)
        for request in self.geometry_data.collisionRequests:
            request.enable_contact=True;request.num_max_contacts=100

    def set_hands(self,hands):
        for index,side,motor in self.hand_indices:
            value=float(hands[side][motor])
            if not np.isfinite(value) or not 0<=value<=1:raise ValueError('手部姿态无效')
            self.q[index]=self.model.lowerPositionLimit[index]+value*(self.model.upperPositionLimit[index]-self.model.lowerPositionLimit[index])

    def check(self, q16):
        q16 = np.asarray(q16,float)
        if q16.shape!=(16,) or not np.isfinite(q16).all():
            raise ValueError('碰撞检查关节输入无效')
        self.q[self.indices] = q16
        self.pin.computeCollisions(self.model,self.data,self.geometry,self.geometry_data,self.q,False)
        for pair,result in zip(self.geometry.collisionPairs,self.geometry_data.collisionResults):
            if result.isCollision():
                a,b = (self.geometry.geometryObjects[i].name for i in (pair.first,pair.second))
                allowed=False
                for rule in self.scene.get('contact_rules',[]):
                    if self.stage not in rule.get('stages',[]) or self.fraction<float(rule.get('start_fraction',1)):
                        continue
                    world='world:'+str(rule.get('object_id'))
                    other=b if a==world else (a if b==world else '')
                    if other.rsplit('_',1)[0] in rule.get('links',[]):
                        contacts=result.getContacts()
                        allowed=bool(contacts) and all(0<=c.penetration_depth<=rule['max_penetration_m'] for c in contacts)
                        if allowed:break
                if allowed:continue
                raise ValueError('碰撞检查失败：'+a+' / '+b)

    def validate_path(self, rows):
        checked = 0
        for index,(a,b) in enumerate(zip(rows,rows[1:])):
            steps = max(1,int(np.ceil(np.max(np.abs(np.array(b)-a)/np.array([.001]+[.01]*15)))))
            for fraction in np.linspace(0,1,steps+1):
                self.fraction=(index+fraction)/max(1,len(rows)-1)
                self.check(np.asarray(a)*(1-fraction)+np.asarray(b)*fraction); checked += 1
        return checked
