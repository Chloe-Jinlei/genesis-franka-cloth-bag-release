"""Recover a copied native IPC dump; never modify its live source workspace.

IPC q/q_prev/q_v and dumped contact history are restored by World.recover. The
separate Genesis rigid q/dq and commands are restored, but its unsaved rigid
solver warmstart and previous coupling force/aim buffers are not. Consequently
this is native checkpoint recovery, not certified exact whole-scene replay.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import hashlib
import json
import shutil
import struct
import tempfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEM = Path('dump/cuda/finite_element/finite_element_method.cu')
ABD = Path('dump/cuda/affine_body/affine_body_dynamics.cu')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _manifest(path):
    return {str(p.relative_to(path)):_sha(p) for p in sorted(Path(path).rglob('*')) if p.is_file()}


def _native_array(path):
    data=Path(path).read_bytes()
    if len(data)<16 or struct.unpack('<Q',data[8:16])[0]!=len(data)-16:
        raise ValueError(f'Invalid native buffer header: {path}')
    return np.frombuffer(data[16:],dtype='<f8').copy()


def load_native_checkpoint(live_run, workspace=None):
    """Freeze an already paused live run and its matching native checkpoint.

    Returns the same bag/bear/rest/q/dq keys as load_saved_frame, plus the actual
    controller command fields and immutable source-copy metadata. The selected
    dump must match the controller's actual soft positions, frame and time.
    """
    run=Path(live_run).resolve()
    status=json.loads((run/'status.json').read_text())
    if status['status'] not in ('paused_after_error','paused','ready','finished','stopped','release_completed'):
        raise ValueError('Checkpoint source must be paused or finished')
    control_hash=_sha(run/'live_control_state.npz')
    with np.load(run/'live_control_state.npz',allow_pickle=False) as z:
        p={k:np.array(z[k],copy=True) for k in z.files}
    with np.load(run/'rest.npz',allow_pickle=False) as z:
        p['rest']={k:np.array(z[k],copy=True) for k in z.files}
    config=json.loads((run/'config.json').read_text())
    dt=float(config['simulation']['dt'])/int(config['simulation']['substeps'])
    # Current runner's time is local to this native world; require an exact frame.
    saved_native=status.get('native_dump',{}).get('native_frame')
    frame=int(saved_native) if saved_native is not None else round(float(p['time'])/dt)
    if (saved_native is None and abs(frame*dt-float(p['time']))>1e-8) or abs(float(status['time_s'])-float(p['time']))>1e-8:
        raise ValueError('Controller status time and native frame disagree')
    actual=np.concatenate([p['bag'],p['bear']],axis=0)
    # Published evidence keeps the checkpoint beside its controller state.
    # Historical status files retain their original absolute paths as evidence.
    bundled_workspace=run/'native_workspace'
    workspace=workspace or (str(bundled_workspace) if bundled_workspace.is_dir()
                            else status.get('native_dump',{}).get('workspace'))
    candidates=[Path(workspace)] if workspace else list(Path(tempfile.gettempdir()).glob('genesis_ipc_*'))
    matches=[]
    for candidate in candidates:
        marker=candidate/f'dump/common/sim_engine.cpp/state.{frame}.json'
        qfile=candidate/FEM/f'q.{frame}'
        if not marker.exists() or not qfile.exists():continue
        # Later diagnostic/recovery worlds may contain the same frame and pose.
        # The original checkpoint was dumped before its controller snapshot.
        if workspace is None and marker.stat().st_mtime_ns > (run/'live_control_state.npz').stat().st_mtime_ns:continue
        state=json.loads(marker.read_text())
        q=_native_array(qfile)
        if state.get('frame')==frame and q.size==actual.size and np.array_equal(q.reshape(actual.shape),actual):
            matches.append(candidate.resolve())
    if len(matches)!=1:
        raise RuntimeError(f'Expected one exact matching native dump, found {len(matches)}')
    original=matches[0]
    before=_manifest(original)
    snapshot_parent=ROOT/'runtime_ipc/native_checkpoint_snapshots'
    snapshot_parent.mkdir(exist_ok=True)
    frozen=Path(tempfile.mkdtemp(prefix=f'{run.name}_frame{frame}_',dir=snapshot_parent))
    shutil.copytree(original,frozen,dirs_exist_ok=True)
    if before!=_manifest(original) or before!=_manifest(frozen) or control_hash!=_sha(run/'live_control_state.npz'):
        raise RuntimeError('Source changed while checkpoint was copied')
    velocities=_native_array(frozen/FEM/f'q_v.{frame}').reshape(-1,3)
    n=len(p['bag']);p['bag_velocity']=velocities[:n].copy();p['bear_velocity']=velocities[n:].copy()
    p.update(source_run=str(run),native_source_workspace=str(original),native_snapshot_workspace=str(frozen),native_frame=frame,
             source_config=config,source_status=status,native_dump_manifest=before)
    p['restart_metadata']={
        'mode':'copied_native_ipc_checkpoint_recovery','exact_continuation':False,
        'native_dump_frame':frame,'saved_time_s':float(p['time']),
        'source_run':str(run),'source_workspace':str(original),'frozen_workspace_copy':str(frozen),
        'source_controller_sha256':control_hash,'source_config_sha256':_sha(run/'config.json'),
        'native_velocity_method':'exact native q_v buffer; no finite differences',
        'joint_velocity_method':'exact saved Genesis get_dofs_velocity result',
        'original_rest_geometry_required':True,'source_approximate_restart_lineage':status.get('restart'),
        'native_solver_history_restored':False,
        'whole_scene_exact_resume_limitation':'Genesis rigid solver warmstart, prior coupling aim/force buffers, and nonsaved sensor buffers are rebuilt; full-scene bitwise continuation is not certified.',
        'old_process_or_workspace_modified':False}
    return p


@contextmanager
def native_checkpoint_context(payload):
    """Build original scene at saved q, then recover into its new IPC workspace.

    Use with scene.build(initial_q={side:payload['q_'+side] ...}). No simulation
    advance occurs. Existing original initialization and material hooks run first.
    """
    from genesis.engine.couplers.ipc_coupler.coupler import IPCCoupler
    from genesis.engine.couplers.ipc_coupler.utils import read_ipc_geometry_metadata
    from uipc.backend import SceneVisitor
    import uipc
    original=IPCCoupler._finalize_ipc
    report=dict(payload['restart_metadata'],status='pending_build')
    holder=[]
    rest_hashes={}
    def finalize(coupler):
        if holder:raise RuntimeError('Exactly one IPC world is supported per recovery context')
        holder.append(coupler)
        visitor=SceneVisitor(coupler._ipc_scene)
        rests={int(s.id()):s.geometry() for s in visitor.rest_geometries()}
        for slot in visitor.geometries():
            geo=slot.geometry();meta=read_ipc_geometry_metadata(geo)
            if meta is None or meta[0] not in ('cloth','fem'):continue
            key='bag' if meta[0]=='cloth' else 'bear'
            rest=rests[int(slot.id())]
            expected=payload['rest'][key]
            native=uipc.view(rest.positions()).reshape(-1,3)
            if not np.array_equal(native,expected):raise ValueError(f'{key} original rest positions changed')
            entity=coupler.fem_solver.entities[meta[2]]
            cells=payload['rest']['bag_faces' if key=='bag' else 'bear_tets']
            if not np.array_equal(np.asarray(entity.elems),cells):raise ValueError(f'{key} original rest topology changed')
        rest_hashes.update({i:hashlib.sha256(json.dumps(g.to_json(),sort_keys=True).encode()).hexdigest() for i,g in rests.items()})
        original(coupler)
    IPCCoupler._finalize_ipc=finalize
    try:
        yield report
        if len(holder)!=1:raise RuntimeError('Build exactly one IPC scene inside context')
        c=holder[0];frame=int(payload['native_frame'])
        target=Path(c._ipc_engine.workspace()).resolve()
        source=Path(payload['native_snapshot_workspace']).resolve()
        if target in (source,Path(payload['native_source_workspace']).resolve()):raise RuntimeError('Recovery destination must be a new workspace')
        shutil.copytree(source,target,dirs_exist_ok=True)
        if not c._ipc_world.recover(frame):raise RuntimeError(f'World.recover({frame}) returned false')
        if int(c._ipc_world.frame())!=frame:raise RuntimeError('Recovered frame mismatch')
        c._ipc_world.retrieve()
        for side in ('left','right'):
            robot=next(e for e in c.rigid_solver.entities if e.name==side+'_franka')
            actual=robot.get_qpos().detach().cpu().numpy().reshape(-1)
            if not np.array_equal(actual,payload['q_'+side]):raise RuntimeError('Caller did not build at saved actual q')
            robot.set_dofs_velocity(payload['dq_'+side])
        c._store_gs_rigid_states();c._retrieve_fem_states();c._retrieve_rigid_states()
        after={int(s.id()):hashlib.sha256(json.dumps(s.geometry().to_json(),sort_keys=True).encode()).hexdigest() for s in SceneVisitor(c._ipc_scene).rest_geometries()}
        if rest_hashes!=after:raise RuntimeError('Native recovery changed original rest geometry/material attributes')
        # Dumping the recovered world only touches its new workspace. Exact bytes
        # verify q, previous q, velocities, BDF and the dumped contact histories.
        expected={str(p.relative_to(source)):_sha(p) for p in source.rglob('*') if p.is_file() and (p.name.endswith(f'.{frame}') or p.name==f'state.{frame}.json')}
        if not c._ipc_world.dump():raise RuntimeError('Recovered world dump verification failed')
        different=[name for name,digest in expected.items() if not (target/name).exists() or _sha(target/name)!=digest]
        essential=[str(folder/f'{name}.{frame}') for folder in (FEM,ABD) for name in ('q','q_prev','q_v')]
        if any(name in different for name in essential):raise RuntimeError(f'Native state did not round-trip exactly: {different}')
        if different:raise RuntimeError(f'Native dumped history did not round-trip exactly: {different}')
        report.update(status='native_checkpoint_recovered_no_steps',native_solver_history_restored=True,
            native_all_dumped_buffers_roundtrip_identical=True,native_dumped_buffers_verified=len(expected),
            native_dumped_buffer_names=sorted(expected),new_workspace=str(target),
            all_rest_geometry_and_material_json_unchanged=True,native_steps_executed=0)
        payload['native_recovery_report']=report
    finally:
        IPCCoupler._finalize_ipc=original


def sync_current_native_views(scene,payload):
    """Synchronize views and verify exact actual positions/native velocities."""
    import uipc
    from uipc.core import FiniteElementStateAccessorFeature
    c=scene.sim.coupler;c._ipc_world.retrieve();c._retrieve_fem_states();c._retrieve_rigid_states()
    feature=c._ipc_world.features().find(FiniteElementStateAccessorFeature)
    state=feature.create_geometry()
    state.vertices().create(uipc.builtin.velocity,np.zeros(3,dtype=np.float64))
    feature.copy_to(state)
    native_velocity=uipc.view(state.vertices().find(uipc.builtin.velocity)).reshape(-1,3)
    expected_velocity=np.concatenate([payload['bag_velocity'],payload['bear_velocity']],axis=0)
    if not np.array_equal(native_velocity,expected_velocity):raise RuntimeError('Recovered native velocities mismatch')
    errors={}
    for entity in c.fem_solver.entities:
        key='bag' if entity.name=='open_canvas_tote' else 'bear' if entity.name=='free_soft_teddy' else None
        if not key:continue
        actual=entity.get_state().pos.detach().cpu().numpy().reshape(-1,3)
        errors[key]=float(np.max(np.abs(actual-payload[key])))
        if not np.array_equal(actual,payload[key]):raise RuntimeError(f'Recovered {key} positions mismatch')
    return {'native_frame':int(c._ipc_world.frame()),'native_positions_max_error_m':errors,
            'native_velocity_max_error_m_s':float(np.max(np.abs(native_velocity-expected_velocity))),
            'native_steps_executed':0,'whole_scene_exact_continuation':False}
