"""Unit-aware limits and C2 quintic time parameterization (lift in metres)."""
import numpy as np


def joint_vector(rotary, linear):
    return np.array([linear] + [rotary]*15, dtype=float)


ERROR_LIMITS = {
    'start': joint_vector(.02, .005),
    'hold': joint_vector(.005, .002),
    'tracking': joint_vector(.15, .02),
    'fixed': joint_vector(.03, .003),
    'tick': joint_vector(.03, .003),
    'goal': joint_vector(.02, .005),
}


def exceeds(actual, target, kind, indices=None):
    error = np.abs(np.asarray(actual)-np.asarray(target))
    limits = ERROR_LIMITS[kind]
    if error.shape != (16,) or not np.isfinite(error).all():
        return True
    if indices is not None:
        error, limits = error[indices], limits[indices]
    return bool(np.any(error > limits))


def limits_from_profile(profile):
    limits = {
        'velocity': joint_vector(profile['joint_velocity_rad_s'], profile.get('lift_velocity_m_s', .03)),
        'acceleration': joint_vector(profile['joint_acceleration_rad_s2'], profile.get('lift_acceleration_m_s2', .05)),
        'jerk': joint_vector(profile.get('joint_jerk_rad_s3', 2.), profile.get('lift_jerk_m_s3', .2)),
    }
    for name, vector in limits.items():
        configured = profile.get('joint_limits', {}).get(name)
        if configured is not None:
            override = np.asarray(configured, dtype=float)
            if override.shape != (16,):
                raise ValueError(name+'需要16个逐关节限值，首项单位为米')
            limits[name] = np.minimum(vector, override)
        if not np.isfinite(limits[name]).all() or np.any(limits[name] <= 0):
            raise ValueError(name+'限值必须为有限正数')
    return limits


def extrema(coefficients, derivative=0):
    coeff = np.asarray(coefficients, dtype=float)
    for _ in range(derivative):
        coeff = np.arange(1, len(coeff))[:, None]*coeff[1:]
    result = []
    for column in coeff.T:
        critical = np.polynomial.polynomial.polyroots(np.arange(1,len(column))*column[1:])
        points = [0., 1.] + [float(x.real) for x in critical if abs(x.imag)<1e-8 and 0<x.real<1]
        values = np.polynomial.polynomial.polyval(points, column)
        result.append((min(values), max(values)))
    return np.array(result)


def parameterize(rows, limits, blend=True):
    rows = np.asarray(rows, dtype=float)
    if rows.ndim != 2 or rows.shape[1] != 16 or len(rows)<2 or not np.isfinite(rows).all():
        raise ValueError('轨迹需要至少两个完整关节点')
    delta = np.diff(rows, axis=0)
    dt = np.maximum(.08, np.max(1.875*np.abs(delta)/limits['velocity'], axis=1))
    dt = np.maximum(dt, np.max(np.sqrt(5.78*np.abs(delta)/limits['acceleration']), axis=1))
    dt = np.maximum(dt, np.max(np.cbrt(60*np.abs(delta)/limits['jerk']), axis=1))
    velocity = np.zeros_like(rows)
    if blend and len(rows)>2:
        slopes = delta/dt[:, None]
        same = slopes[:-1]*slopes[1:] > 0
        velocity[1:-1] = np.where(same, .5*np.sign(slopes[:-1])*np.minimum(abs(slopes[:-1]), abs(slopes[1:])), 0.)
    coefficients = []
    scale = 1.
    for i, duration in enumerate(dt):
        c = np.zeros((6,16)); c[0] = rows[i]; c[1] = velocity[i]*duration
        c[3:] = np.linalg.solve(np.array([[1,1,1],[3,4,5],[6,12,20]], float),
            np.array([delta[i]-c[1], velocity[i+1]*duration-c[1], np.zeros(16)]))
        bounds = extrema(c)
        if np.any(bounds[:,0]<np.minimum(rows[i],rows[i+1])-1e-10) or np.any(bounds[:,1]>np.maximum(rows[i],rows[i+1])+1e-10):
            if blend:
                return parameterize(rows, limits, blend=False)
            raise ValueError('时间参数化造成关节越界')
        for derivative, name in enumerate(('velocity','acceleration','jerk'),1):
            peak = np.max(abs(extrema(c, derivative)),axis=1)/duration**derivative
            scale = max(scale, float(np.max(peak/limits[name]))**(1/derivative))
        coefficients.append(c.tolist())
    dt *= scale*1.001
    return {'times_s': np.r_[0.,np.cumsum(dt)].tolist(),
            'trajectory_coefficients': coefficients, 'trajectory_blended': bool(blend),
            'motion_limits': {k:v.tolist() for k,v in limits.items()}}


def sample(plan, elapsed):
    times = np.asarray(plan['times_s'])
    if elapsed >= times[-1]:
        return np.asarray(plan['joint_waypoints'][-1], dtype=float)
    index = max(0,int(np.searchsorted(times,elapsed,side='right'))-1)
    u = np.clip((elapsed-times[index])/(times[index+1]-times[index]),0.,1.)
    coefficients = np.asarray(plan['trajectory_coefficients'][index])
    return np.polynomial.polynomial.polyval(u, coefficients)
