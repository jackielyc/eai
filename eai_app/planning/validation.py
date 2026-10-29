"""Shared final trajectory checks for restoration and alternate planners."""
import numpy as np
from eai_app.domain.motion.trajectory import parameterize,limits_from_profile,sample
from eai_app.planning.scene import CollisionScene,scene_digest


def certify_plan(plan,profile,inputs):
    plan=dict(plan)
    timing=parameterize(plan['joint_waypoints'],limits_from_profile(profile))
    plan.update(timing)
    if inputs.get('collision_required'):
        scene=inputs['scene'];checker=CollisionScene(profile['urdf_path'],scene,
            inputs['hands'],inputs.get('attachments',[]))
        samples=[sample(plan,float(t)).tolist() for t in np.linspace(0,plan['times_s'][-1],
            max(2,int(np.ceil(plan['times_s'][-1]/.05))+1))]
        count=checker.validate_path(samples)
        plan.update(collision_checked=True,collision_method='sampled_full_geometry',collision_samples=count,
            geometry_sha256=checker.geometry_hash,scene_sha256=scene_digest(scene),
            scene_file_sha256=inputs['scene_file_sha256'],scene_path=inputs['scene_path'],hands=inputs['hands'])
    plan['assembly_file_sha256']=inputs.get('assembly_file_sha256')
    plan['profile_sha256']=inputs.get('profile_sha256')
    return plan
