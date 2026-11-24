"""Finite-force pickup with a rising, rotating wrist. No soft-state assignment."""
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from motion import minimum_jerk
from postlift_orientation import _serial, continuous_rate_bounds
from run import GateFailure


def upright_pickup(exp):
    option = exp.config['task']['pickup_uprighting']
    state = exp.frames[-1]
    initial_p, initial_r = exp.planner.fk('right', np.asarray(state['q_right'])[:7])
    command_p, command_r = exp.planner.fk('right', exp.qs['right'])
    if not np.isfinite(float(state['time'])) or abs(float(state['time'])-exp.time) > 1e-8:
        raise GateFailure('Uprighting requires the fresh completed closure state')
    initial_pe=float(np.linalg.norm(command_p-initial_p))
    initial_ae=float((command_r*initial_r.inv()).magnitude())
    if not np.isfinite([initial_pe,initial_ae]).all() or initial_pe > .0005 or initial_ae > .002:
        raise GateFailure('Uprighting command and actual start disagree')
    target = initial_p + np.array([0., 0., float(option['rise_m'])])
    final_r = Rotation.from_euler('x', float(option['degrees']), degrees=True) * initial_r
    observed = {s:float(np.clip(np.asarray(state['q_'+s])[7:].mean(), 0., .04)) for s in exp.robots}
    speed = float(exp.config['task'].get('motion_speed_scale', 1.))
    plan = exp.planner.trajectory(exp.qs, {'right':target}, rotations={'right':final_r},
        duration=float(option['duration_s'])/speed, samples=int(option['samples']),
        finger_start=observed, finger_end=observed, clearance_margin=.008,
        max_joint_velocity=.55*speed, max_joint_acceleration=speed**2, validate_samples=401)
    # Use continuous polynomial bounds, preserving the original movement limits.
    rates = continuous_rate_bounds(plan)
    v = max(float(np.max(row['velocity_upper_bound_rad_s'])) for row in rates.values())
    a = max(float(np.max(row['acceleration_upper_bound_rad_s2'])) for row in rates.values())
    plan.duration *= max(1., v/(.55*speed), np.sqrt(a/(speed**2))) * 1.002
    rates = continuous_rate_bounds(plan)
    if any(np.max(row['velocity_upper_bound_rad_s']) > .55*speed or
           np.max(row['acceleration_upper_bound_rad_s2']) > speed**2 for row in rates.values()):
        raise GateFailure('Uprighting exceeds original continuous rate bounds')
    expected = Slerp([0., 1.], Rotation.from_quat([command_r.as_quat(), final_r.as_quat()]))
    min_rigid, min_table = float('inf'), float('inf')
    max_pe, max_ae = 0., 0.
    rows = []
    for t in np.linspace(0., plan.duration, 401):
        sample = plan.sample(float(t))
        fraction = float(minimum_jerk(t/plan.duration)[0])
        p, r = exp.planner.fk('right', sample['q']['right'])
        pe = float(np.linalg.norm(p - ((1.-fraction)*command_p + fraction*target)))
        ae = float((r*expected(fraction).inv()).magnitude())
        max_pe, max_ae = max(max_pe,pe), max(max_ae,ae)
        if not np.isfinite([pe,ae]).all() or pe > .0005 or ae > .002:
            raise GateFailure('Uprighting reference exceeds original FK tolerance')
        if not np.array_equal(sample['q']['left'], exp.qs['left']):
            raise GateFailure('Uprighting moved the left command')
        for left_q in (exp.qs['left'], np.asarray(state['q_left'])[:7]):
            clearances = exp.planner.clearance(dict(sample['q'], left=left_q), observed)
            gap = min(float(row['distance']) for row in clearances.values())
            min_rigid = min(min_rigid,gap)
            if not np.isfinite(gap) or gap < .008:
                raise GateFailure('Uprighting lacks original rigid clearance')
        frozen = (r*initial_r.inv()).apply(state['bear']-initial_p)+p
        z = float(frozen[:,2].min())
        min_table = min(min_table,z)
        rows.append([float(t),z])
        if not np.isfinite(z) or z < exp.config['table']['height']:
            raise GateFailure('Uprighting frozen carry would push the teddy through the table')
    evidence = {'scope':'401-sample frozen carry forecast and robot reference only, not continuous soft-body clearance proof; native contact and deformation guards remain active.',
        'option':option,'actual_source_time_s':float(state['time']), 'target_tcp_m':target,
        'target_rotation_xyzw':final_r.as_quat(),'duration_s':plan.duration,
        'continuous_rate_bounds':rates,'sample_count':401,'minimum_rigid_clearance_m':min_rigid,
        'minimum_frozen_table_z_m':min_table,'frozen_minimum_height_trace':rows,
        'maximum_position_error_m':max_pe,'maximum_angle_error_rad':max_ae}
    (exp.out/'pickup_uprighting_reference.json').write_text(json.dumps(_serial(evidence),indent=2,allow_nan=False)+'\n')
    np.savez_compressed(exp.out/'pickup_uprighting_plan.npz',duration=plan.duration,
        left_x=plan.curves['left'].x,left_c=plan.curves['left'].c,
        right_x=plan.curves['right'].x,right_c=plan.curves['right'].c)
    # Geometric checking uses observed opening; the finite motor target persists.
    plan.finger_start=exp.finger.copy()
    plan.finger_end=exp.finger.copy()
    exp.execute('pickup_teddy_upright',plan)
    exp.rotations['right']=final_r
    actual_p,actual_r=exp.planner.fk('right',np.asarray(exp.frames[-1]['q_right'])[:7])
    pe=float(np.linalg.norm(actual_p-target));ae=float((actual_r*final_r.inv()).magnitude())
    endpoint={'actual_time_s':exp.time,'position_error_m':pe,'angle_error_rad':ae,
              'full_workflow_success':False}
    (exp.out/'pickup_uprighting_endpoint.json').write_text(json.dumps(endpoint,indent=2,allow_nan=False)+'\n')
    if not np.isfinite([pe,ae]).all() or pe > .0005 or ae > .002:
        raise GateFailure('Uprighting actual endpoint exceeds original tracking tolerance')
