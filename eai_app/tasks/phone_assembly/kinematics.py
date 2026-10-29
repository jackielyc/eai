"""A2D IK mapping for the nominal contact proxy, kept separate from real calibration."""
import time
import numpy as np
from eai_app.planning.arm_ik import HostArmIK,_JOINT_NAMES


class A2DMapping:
    def __init__(self,urdf_path,initial_tool_pose,config=None):
        self.ik=HostArmIK(urdf_path);self.pin=self.ik._pin
        R=np.array([[-.643,.447,.622],[.763,.443,.471],[-.065,.777,-.626]])
        u,_,v=np.linalg.svd(R);R=u@v
        self.T_link_tool=np.eye(4);self.T_link_tool[:3,:3]=R.T
        self.T_link_tool[:3,3]=R.T@np.array([0,0,-.08])
        # A folded idle arm leaves room to compensate waist motion without
        # moving its TCP; an almost extended idle arm can make the task infeasible.
        self.q=np.zeros(16);self.q[0]=.2;self.q[3]=.5;self.q[5]=-1.3
        target=self.link_goal(initial_tool_pose)
        q=self.ik.solve_targets(self.q,[(self.ik._right_fid,target)],_JOINT_NAMES[9:],
            max_iters=500,attempts=5,deadline=time.monotonic()+5)
        if q is None:raise ValueError('仿真初始工具位姿无法映射到A2D右臂')
        self.q=np.asarray(q)
        self.ik.ee_pose(self.q,'left');self.held=self.ik.data.oMf[self.ik._left_fid].copy()
        self.max_position_error=0.;self.max_rotation_error=0.
        self.collision=None;self.collision_samples=0;self.dynamic_objects={};self.config=config
        if config is not None:
            if config.get('source')!='simulation_example':raise ValueError('代理工具碰撞模型仅用于仿真')
            from eai_app.planning.scene import CollisionScene
            table=np.eye(4);table[:3,3]=[.55,-.1,config['table_z']-.025]
            box=np.eye(4);box[:3,3]=[*config['box_xy'],config['table_z']+(config['wall']+config['box_inner_size'][2])/2]
            scene={'world':[{'id':'table','size_m':[.56,.84,.05],'T_base_object':table.tolist()},
                {'id':'box','size_m':[config['box_inner_size'][0]+2*config['wall'],config['box_inner_size'][1]+2*config['wall'],
                                    config['wall']+config['box_inner_size'][2]],'T_base_object':box.tolist()}]}
            for role in ('phone','lid'):
                T=np.eye(4);T[:3,3]=[*config[role+'_xy'],config['table_z']+.01]
                size=config['phone_size'] if role=='phone' else [config['lid_inner_size'][0]+2*config['lid_wall'],
                    config['lid_inner_size'][1]+2*config['lid_wall'],config['lid_skirt']+config['lid_wall']]
                scene['world'].append({'id':role,'size_m':size,'T_base_object':T.tolist()})
            self.collision=CollisionScene(urdf_path,scene,accessory_path=False)
            self.dynamic_objects={role:self.collision.geometry.getGeometryId('world:'+role) for role in ('phone','lid')}
            self.collision.check(self.q)
    def link_goal(self,tool_pose):
        T=np.asarray(tool_pose)@np.linalg.inv(self.T_link_tool)
        return self.pin.SE3(T[:3,:3],T[:3,3])
    def update(self,tool_pose,objects=None):
        for role,index in self.dynamic_objects.items():
            if objects and role in objects:
                T=np.asarray(objects[role],float).copy()
                if role=='lid':T[:3,3]+=T[:3,:3]@np.array([0,0,-self.config['lid_skirt']/2])
                self.collision.geometry.geometryObjects[index].placement=self.pin.SE3(T[:3,:3],T[:3,3])
        target=self.link_goal(tool_pose)
        q=self.ik.solve_coordinated(self.q,moving_side='right',target=target,allow_waist=True,
            held_target=self.held,pos_tol=.001,ori_tol=.01,max_iters=250,attempts=2,
            deadline=time.monotonic()+2)
        if q is None:raise ValueError('当前接触工具位姿无法映射到A2D；停止仿真任务')
        q=np.asarray(q)
        if self.collision:self.collision_samples+=self.collision.validate_path([self.q.tolist(),q.tolist()])
        self.q=q
        self.ik.ee_pose(self.q,'right');actual=self.ik.data.oMf[self.ik._right_fid]
        self.max_position_error=max(self.max_position_error,float(np.linalg.norm(actual.translation-target.translation)))
        self.max_rotation_error=max(self.max_rotation_error,float(np.linalg.norm(self.pin.log3(actual.rotation.T@target.rotation))))
        return self.q.tolist()
