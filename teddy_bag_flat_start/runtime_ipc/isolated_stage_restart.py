"""Isolated approximate stage restart, preserving native reference meshes/materials.

This is NOT an exact checkpoint continuation. Velocities are finite differences
of saved frames and joint samples. IPC contact/Newton/BDF warm history is rebuilt.
No setter is used during the resumed rollout; this adapter only establishes a
new native world's initial current state, separate from its original rest state.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import hashlib, json
import numpy as np


def _hash(a):
    a=np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()


def load_saved_frame(source_run, time_s):
    run=Path(source_run).resolve()
    with np.load(run/'states.npz',allow_pickle=False) as archive:
        times=np.asarray(archive['time'],float); index=int(np.argmin(abs(times-float(time_s))))
        if abs(times[index]-float(time_s))>.026:
            raise ValueError('No saved complete frame within 26 ms of requested time')
        if index<1: raise ValueError('Need a previous soft frame to estimate velocity')
        current={k:np.asarray(archive[k][index]).copy() for k in archive.files}
        dt=float(times[index]-times[index-1]); assert dt>0
        velocities={k:(current[k]-np.asarray(archive[k][index-1]))/dt for k in ('bag','bear')}
        previous_time=float(times[index-1])
    with np.load(run/'rest.npz',allow_pickle=False) as archive:
        rest={k:np.asarray(archive[k]).copy() for k in archive.files}
    trace=np.load(run/'joint_trace.npy',allow_pickle=False)
    j=int(np.argmin(abs(trace[:,0]-float(current['time']))))
    if j<1 or abs(trace[j,0]-float(current['time']))>1e-7:
        raise ValueError('Saved complete frame must have an exact joint-trace time')
    joint_dt=float(trace[j,0]-trace[j-1,0]); assert joint_dt>0
    dq=(trace[j,1:]-trace[j-1,1:])/joint_dt
    for side,sl in [('left',slice(0,9)),('right',slice(9,18))]:
        if not np.allclose(current['q_'+side].reshape(-1),trace[j,1:][sl],rtol=0,atol=1e-9):
            raise ValueError('Joint snapshot and trace disagree')
        current['q_'+side]=current['q_'+side].reshape(-1)
        current['dq_'+side]=dq[sl]
    current.update(rest=rest,bag_velocity=velocities['bag'],bear_velocity=velocities['bear'],source_run=str(run))
    current['restart_metadata']={
        'mode':'approximate_saved_frame_native_reinitialization','exact_continuation':False,
        'requested_time_s':float(time_s),'saved_time_s':float(current['time']),
        'soft_velocity_method':'backward difference of successive saved complete soft frames',
        'soft_velocity_previous_time_s':previous_time,'soft_velocity_dt_s':dt,
        'joint_velocity_method':'backward difference of original per-step joint_trace',
        'joint_velocity_dt_s':joint_dt,'original_rest_geometry_required':True,
        'contact_solver_history_restored':False,'BDF_history_restored':False,
        'source_array_sha256':{k:_hash(current[k]) for k in ('bag','bear','q_left','q_right','bag_velocity','bear_velocity','dq_left','dq_right')},
        'rest_array_sha256':{k:_hash(v) for k,v in rest.items()},
        'config_sha256':hashlib.sha256((run/'config.json').read_bytes()).hexdigest()}
    return current


def _array(value):
    if hasattr(value,'detach'):return value.detach().cpu().numpy()
    return np.asarray(value)


@contextmanager
def native_restart_context(payload):
    """Wrap one scene.build(initial_q=payload q) in an isolated process.

    Original per-instance initialization, shell-shear, STC and rigid-contact hooks
    stay active. The class method is restored before returning to caller. Cannot
    be used concurrently with another scene build in the same process.
    """
    import uipc
    from uipc.backend import SceneVisitor
    from uipc.geometry import SimplicialComplexSlot
    from genesis.engine.couplers.ipc_coupler.coupler import IPCCoupler
    from genesis.engine.couplers.ipc_coupler.utils import read_ipc_geometry_metadata
    original=IPCCoupler._finalize_ipc
    report=dict(payload['restart_metadata'],status='pending_build',entities=[],rigid_velocity_initialization='estimated qdot mapped through rigid forward velocity to IPC affine transform derivative')
    used=False
    def finalize(coupler):
        nonlocal used
        if used:raise RuntimeError('Restart adapter may initialize exactly one IPC world')
        used=True
        visitor=SceneVisitor(coupler._ipc_scene)
        rest_slots={int(slot.id()):slot for slot in visitor.rest_geometries()}
        rest_hashes={i:hashlib.sha256(json.dumps(s.geometry().to_json(),sort_keys=True).encode()).hexdigest() for i,s in rest_slots.items()}
        # Existing initialize_before_ipc already established q before geometry build.
        rigid=coupler.rigid_solver
        for side in ('left','right'):
            robot=next(e for e in rigid.entities if e.name==side+'_franka')
            actual=_array(rigid.get_qpos(np.arange(robot.q_start,robot.q_end))).reshape(-1)
            if not np.allclose(actual,payload['q_'+side],rtol=0,atol=1e-8):
                raise ValueError('Caller must pass saved q as build(initial_q=...)')
            rigid.set_dofs_velocity(payload['dq_'+side],dofs_idx=np.arange(robot.dof_start,robot.dof_end))
        matched=set()
        for slot in visitor.geometries():
            if not isinstance(slot,SimplicialComplexSlot):continue
            geo=slot.geometry();meta=read_ipc_geometry_metadata(geo)
            if meta is None or meta[0] not in ('cloth','fem'):continue
            kind,env,index=meta
            if env!=0:raise ValueError('This restart supports only one environment')
            key='bag' if kind=='cloth' else 'bear';entity=coupler.fem_solver.entities[index]
            current=uipc.view(geo.positions()).reshape(-1,3)
            native_rest=uipc.view(rest_slots[int(slot.id())].geometry().positions()).reshape(-1,3)
            expected=np.asarray(payload['rest'][key],float)
            if current.shape!=expected.shape or not np.allclose(current,expected,rtol=0,atol=1e-8) or not np.allclose(native_rest,expected,rtol=0,atol=1e-8):
                raise ValueError(f'{key}: native original mesh order/coordinates do not match saved rest')
            cells=np.asarray(entity.elems)
            saved=payload['rest']['bag_faces' if key=='bag' else 'bear_tets']
            if not np.array_equal(cells,saved):raise ValueError(f'{key}: original topology differs')
            velocity=geo.vertices().find(uipc.builtin.velocity)
            if velocity is None:velocity=geo.vertices().create(uipc.builtin.velocity,np.zeros(3,dtype=np.float64))
            current[:]=np.asarray(payload[key],float)
            uipc.view(velocity).reshape(-1,3)[:]=payload[key+'_velocity']
            if not np.isfinite(current).all() or not np.isfinite(payload[key+'_velocity']).all():
                raise ValueError('Nonfinite initial state/velocity')
            matched.add(key);report['entities'].append({'name':key,'geometry_id':int(slot.id()),'vertex_count':len(current),'reference_positions_max_error_m':float(np.max(np.abs(native_rest-expected))),'reference_topology_identical':True,'estimated_velocity_max_m_s':float(np.linalg.norm(payload[key+'_velocity'],axis=1).max())})
        if matched!={'bag','bear'}:raise RuntimeError('Did not initialize both original soft geometries')
        # IPC ABD velocity is dT/dt, not a spatial twist. Preserve fixed objects.
        for link,slots in coupler._abd_slots_by_link.items():
            if link.entity.name not in ('left_franka','right_franka'):continue
            linear=_array(rigid.get_links_vel(link.idx)).reshape(-1,3)[0]
            angular=_array(rigid.get_links_ang(link.idx)).reshape(-1,3)[0]
            wx,wy,wz=angular;skew=np.array([[0.,-wz,wy],[wz,0.,-wx],[-wy,wx,0.]])
            for slot in slots:
                geo=slot.geometry();trans=uipc.view(geo.transforms()).reshape(-1,4,4)[0]
                derivative=np.zeros((4,4));derivative[:3,:3]=skew@trans[:3,:3];derivative[:3,3]=linear
                vel=geo.instances().find(uipc.builtin.velocity)
                if vel is None:vel=geo.instances().create(uipc.builtin.velocity,np.zeros((4,4),dtype=np.float64))
                uipc.view(vel).reshape(-1,4,4)[:]=derivative
        after={i:hashlib.sha256(json.dumps(s.geometry().to_json(),sort_keys=True).encode()).hexdigest() for i,s in rest_slots.items()}
        if rest_hashes!=after:raise RuntimeError('Restart touched original rest geometry/material attributes')
        report['all_rest_geometry_and_material_json_unchanged']=True
        original(coupler)
        report['status']='native_initialized_approximate_state';report['initial_native_frame']=int(coupler._ipc_world.frame())
    IPCCoupler._finalize_ipc=finalize
    try:
        yield report
        if not used:raise RuntimeError('No native IPC world was built in restart context')
    finally:
        IPCCoupler._finalize_ipc=original


def sync_genesis_soft_views(scene,payload):
    """Retrieve actual new IPC coordinates into Genesis after build; no advance."""
    coupler=scene.sim.coupler
    coupler._ipc_world.retrieve();coupler._retrieve_fem_states()
    errors={}
    for entity in coupler.fem_solver.entities:
        key='bag' if entity.name=='open_canvas_tote' else 'bear' if entity.name=='free_soft_teddy' else None
        if key:
            p=_array(entity.get_state().pos).reshape(-1,3)
            errors[key]=float(np.max(np.abs(p-payload[key])))
            if errors[key]>1e-7:raise RuntimeError(f'{key}: native state differs from requested saved state')
    return {'native_retrieval_max_position_errors_m':errors,'native_steps_executed':0,'exact_continuation':False}
