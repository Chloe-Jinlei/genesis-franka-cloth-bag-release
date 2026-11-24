"""Physical one-handle pickup from a flat tabletop bag; no soft-state edits."""
from __future__ import annotations
import json
import numpy as np
from scipy.spatial.transform import Rotation
from contact_grasp import cloth_grasp_frame
from motion import FINGERTIP_BELOW_TCP
from scene import positions


def enabled(config):
    settings = config.get('task', {}).get('flat_start', {})
    if not isinstance(settings, dict):
        raise ValueError('task.flat_start must be an object')
    value = settings.get('enabled', False)
    if type(value) is not bool:
        raise ValueError('task.flat_start.enabled must be a boolean')
    return value


def validate_task(config, task):
    flat = enabled(config)
    if flat and task not in ('settle', 'open', 'lift', 'full'):
        raise ValueError('Flat-start mode supports settle, open, lift, or full only')
    if not flat and task == 'lift':
        raise ValueError('The single-handle lift diagnostic requires flat-start mode')


def initial_control_state(config, planner):
    settings = config['task']['flat_start']
    qs, fingers, rotations = {}, {}, {}
    for side in ('left', 'right'):
        q = np.asarray(settings['initial_q'][side], float)
        if q.shape not in ((7,), (9,)) or not np.isfinite(q).all():
            raise ValueError('Flat-start initial joint state must be seven or nine finite values')
        qs[side] = q[:7].copy()
        fingers[side] = float(settings.get('initial_fingers', {}).get(side, .04))
        if not .038 <= fingers[side] <= .04:
            raise ValueError('Both hands must begin visibly open and ungrasped')
        rotations[side] = planner.fk(side, qs[side])[1]
    clearance = planner.clearance(qs, fingers)
    if min(row['distance'] for row in clearance.values()) < .008:
        raise ValueError('Flat-start initial robot pose fails rigid clearance')
    evidence = {'mode': 'flat_tabletop_single_handle', 'initial_grasp': False,
                'initial_finger_half_openings_m': fingers.copy(),
                'initial_rigid_clearance': clearance}
    return qs, fingers, rotations, evidence


def run_flat_pickup(exp):
    from run import GateFailure, tcp
    settings = exp.config['task']['flat_start']
    exp.report['flat_start_execution'] = {'one_handle_only': True,
        'right_arm_role': 'wait open, then grasp teddy',
        'soft_state_mutations': False, 'attachments_or_fixed_cloth_nodes': False}
    exp.hold('flat_settle', float(settings.get('initial_settle_seconds', .05)))
    bag = positions(exp.bag)
    frame = cloth_grasp_frame(bag, exp.rest['bag_faces'], exp.left_grasp_material_ids,
        shell_radius=exp.config['bag']['thickness'],
        preferred_approach=settings.get('preferred_approach', [0.,0.,-1.]),
        preferred_closing=settings.get('preferred_closing', [0.,-1.,0.]))
    offset=np.asarray(settings.get('grasp_tangent_offset_hand_m',[0.,0.,0.]),float)
    if offset.shape!=(3,) or not np.isfinite(offset).all() or abs(offset[1])>1e-9 or np.linalg.norm(offset)>.008:
        raise GateFailure('Invalid physical grasp offset within the observed ribbon patch')
    target = frame.position + frame.rotation.apply(offset)
    exp.report['flat_start_execution']['observed_grasp_frame'] = frame.as_dict()
    exp.report['flat_start_execution']['observed_at_s'] = exp.time
    exp.report['flat_start_execution']['grasp_target_m'] = target.tolist()
    exp.report['flat_start_execution']['grasp_tangent_offset_hand_m'] = offset.tolist()
    (exp.out/'flat_grasp_observation.json').write_text(json.dumps(exp.report['flat_start_execution'],indent=2)+'\n')
    # The planner checks the complete finger meshes against the table. This
    # inexpensive check also rejects a collapsed tip over the table footprint.
    half=np.asarray(exp.config['table']['size'][:2])/2
    if np.all(np.abs(target[:2])<half) and target[2] < exp.config['table']['height']+FINGERTIP_BELOW_TCP+.008:
        raise GateFailure('Observed flat strap collapsed too close to tabletop for a collision-free downward pinch')

    def move(label, target, duration, rotation):
        plan=exp.planner.trajectory(exp.qs, {'left':np.asarray(target)},
            rotations={'left':rotation}, duration=duration,
            finger_start=exp.finger, finger_end=exp.finger,
            max_joint_velocity=.55,max_joint_acceleration=1.,
            clearance_margin=.008,validate_samples=151)
        exp.execute(label,plan)
        exp.rotations['left']=rotation

    move('approach_single_handle', target, float(settings.get('approach_duration_s',1.2)), frame.rotation)
    exp.hold('close_single_handle',float(settings.get('handle_close_seconds',.35)),left=0.)
    proximity=float(np.linalg.norm(positions(exp.bag)[exp.left_grasp_material_ids]-tcp(exp.robots['left']),axis=1).min())
    exp.report['flat_start_execution']['named_handle_distance_after_close_m']=proximity
    if proximity>.018:
        raise GateFailure('Single strap not retained at closure; refusing a fictitious lift')
    # Read closure contact without recording a second physics sample at this
    # same time. Every later scene.step() supplies its own monitored sample.
    ids=exp.handle_coupling_indices['left']
    data=exp.handle_coupling_data
    reactions=np.einsum('ij,ij->i',data.out_forces[0,ids],data.aim_transforms[0,ids,:3,1])
    exp.report['flat_start_execution']['single_handle_contact_at_closure']={
        'time_s':exp.time,'left_material_distance_m':proximity,
        'left_opening_reactions_n':reactions.tolist()}
    if not np.isfinite(reactions).all() or np.min(reactions)<=.1:
        raise GateFailure('Single strap lacks two-sided native contact at closure')
    exp.must_hold_bag=True
    targets=settings['lift_targets']
    for i,stage in enumerate(targets):
        label='lift_single_handle' if i==len(targets)-1 else f'lift_single_handle_{i+1}'
        rotation=Rotation.from_quat(stage.get('rotation_xyzw',frame.rotation.as_quat()))
        move(label,stage['tcp_m'],float(stage['duration_s']),rotation)
    exp.hold('settle_single_handle',float(settings.get('settle_seconds',1.)))
    from validate import mouth_structure, containment, signed_volumes
    body,rim,topology=mouth_structure(exp.rest['bag'],exp.rest['bag_faces'],exp.config)
    state=exp.frames[-1]
    weights=np.abs(signed_volumes(exp.rest['bear'],exp.rest['bear_tets']))
    opening=containment(state['bag'],state['bear'],body,rim,weights,exp.rest['bear_tets'])
    exp.report['single_handle_opening']={'topology':topology,**opening,
        'actual_time_s':exp.time,'scope':'Measured free cloth opening; not a completed insertion.'}
    (exp.out/'single_handle_opening.json').write_text(json.dumps(exp.report['single_handle_opening'],indent=2)+'\n')
    exp.report['single_handle_lift_completed']=True
