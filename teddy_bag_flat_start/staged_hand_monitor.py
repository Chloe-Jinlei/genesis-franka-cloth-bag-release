"""Read-only height separation from current native IPC hand collision geometry.

ABD positions are link-local reference vertices: Genesis builds each merged
collision mesh in link coordinates and supplies its separate world transform.
The monitor preserves the complete native affine transform. It does not set
positions, joint targets, transforms, contacts, or forces.
"""
from importlib.metadata import version
from pathlib import Path
import hashlib
import numpy as np


def _palm_vertices_in_native_link(entity, target_link, native_points):
    """Reproduce the coupler's fixed-hand merge, excluding the link7 arm mesh."""
    import genesis.utils.geom as gu
    from genesis.engine.couplers.ipc_coupler.utils import (
        compute_link_to_link_transform, find_target_link_for_fixed_merge)
    from scipy.spatial import cKDTree
    palms=[link for link in entity.links if link.name=='hand']
    if len(palms)!=1 or find_target_link_for_fixed_merge(palms[0]) is not target_link:
        raise RuntimeError('Cannot resolve the fixed hand collision geometry into native link7')
    palm=palms[0];merge=compute_link_to_link_transform(palm,target_link)
    pieces=[];geom_ids=[]
    for geom in palm.geoms:
        if geom.n_verts:
            points=gu.transform_by_trans_quat(geom.init_verts,geom.init_pos,geom.init_quat)
            points=gu.transform_by_trans_quat(points,*merge)
            pieces.append(np.asarray(points,dtype=float));geom_ids.append(int(geom.idx))
    if not pieces:
        raise RuntimeError('Reviewed fixed palm has no collision vertices')
    points=np.concatenate(pieces)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise RuntimeError('Nonfinite or malformed fixed-palm collision vertices')
    errors,ids=cKDTree(native_points).query(points)
    if float(np.max(errors))>1e-7:
        raise RuntimeError('Fixed-palm vertices do not match the merged native link7 surface')
    # Use the exact coordinates stored by IPC, after verifying source identity.
    points=native_points[np.unique(ids)].copy()
    return points,dict(collision_body='hand',source_geom_indices=geom_ids,
        merge_target='link7',native_subset_max_error_m=float(np.max(errors)),
        native_slot_vertex_count=len(native_points),native_subset_vertex_indices=np.unique(ids).tolist())


class NativeHandHeightMonitor:
    names = ('link7', 'left_finger', 'right_finger')

    def __init__(self, scene, *, minimum_gap_m=.005):
        found={name:version(name) for name in ('genesis-world','pyuipc')}
        if found!={'genesis-world':'1.4.3','pyuipc':'0.0.29'}:
            raise RuntimeError(f'Unsupported native height-monitor versions: {found}')
        self.scene=scene;self.coupler=scene.sim.coupler
        if type(self.coupler).__name__!='IPCCoupler' or scene.sim.substeps!=1 or scene.sim._B!=1:
            raise RuntimeError('Native height monitor requires one IPC environment and one physics substep')
        self.minimum_gap_m=float(minimum_gap_m)
        if not np.isfinite(self.minimum_gap_m) or self.minimum_gap_m<=0:
            raise ValueError('Native height allowance must be finite and positive')
        self.data=self.coupler._coupling_data
        if self.data is None:
            raise RuntimeError('Build the scene before constructing the height monitor')
        self.links=[];self.indices=[];self.vertices=[];self.geometry_metadata=[]
        entities={id(link.entity):link.entity for link in self.data.links if link.entity.name=='left_franka'}
        if len(entities)!=1:
            raise RuntimeError('Expected exactly one coupled left_franka entity')
        entity=next(iter(entities.values()))
        if entity.material.coup_type!='two_way_soft_constraint':
            raise RuntimeError('Left hand must use the reviewed two-way native coupling')
        if any(link.entity is entity and link.name=='hand' for link in self.data.links):
            raise RuntimeError('Expected hand collision meshes welded into link7, not a separate hand ABD')
        # Same checked Panda fixed-chain mapping used by the force monitor.
        from contact_guard import _model_mapping
        self.model,_=_model_mapping(entity.morph.file)
        for name in self.names:
            matches=[(i,link) for i,link in enumerate(self.data.links) if link.entity is entity and link.name==name]
            if len(matches)!=1:
                raise RuntimeError(f'Missing or ambiguous native left {name}')
            index,link=matches[0];slots=self.coupler._abd_slots_by_link.get(link,[])
            if len(slots)!=1:
                raise RuntimeError(f'Expected one native geometry slot for left {name}')
            geometry=slots[0].geometry()
            if geometry.dim()!=2:
                raise RuntimeError(f'Expected a native collision triangle surface for left {name}')
            points=np.asarray(geometry.positions().view(),dtype=float).reshape(-1,3).copy()
            if len(points)<4 or not np.isfinite(points).all():
                raise RuntimeError(f'Invalid native collision vertices for left {name}')
            selection=dict(collision_body=name,native_slot_vertex_count=len(points))
            if name=='link7':
                points,selection=_palm_vertices_in_native_link(entity,link,points)
                if len(points)<4:
                    raise RuntimeError('Incomplete native palm collision surface')
            points.setflags(write=False)
            self.links.append(link);self.indices.append(index);self.vertices.append(points)
            self.geometry_metadata.append(dict(link=name,coupling_index=index,vertices=len(points),selection=selection,
                local_bounds_m=[points.min(0).tolist(),points.max(0).tolist()],
                local_vertices_sha256=hashlib.sha256(points.astype('<f8').tobytes()).hexdigest()))
        bears=[entity for entity in scene.sim.fem_solver.entities if entity.name=='free_soft_teddy']
        if len(bears)!=1 or int(bears[0].n_vertices)<4:
            raise RuntimeError('Expected exactly one complete native FEM teddy')
        self.bear_vertex_count=int(bears[0].n_vertices)
        self._last_step=None;self._last_time=None

    def reference_geometry(self):
        """Save once as NPZ for independent reconstruction of every trace row."""
        return {'left_'+name:points.copy() for name,points in zip(self.names,self.vertices)}

    def measure(self,time_s,bear_positions):
        timestamp=float(time_s);step=int(self.scene.sim.cur_step_global)
        if not np.isfinite(timestamp) or timestamp<0 or step<=0:
            raise RuntimeError('Height monitor needs a finite time after a completed physics step')
        if self._last_step is not None and (step<self._last_step or timestamp<self._last_time-1e-10 or
                (step==self._last_step and abs(timestamp-self._last_time)>1e-10)):
            raise RuntimeError('Stale or reversed native height sample')
        if self.coupler._coupling_data is not self.data or any(
                self.data.links[i] is not link for i,link in zip(self.indices,self.links)):
            raise RuntimeError('Native coupling buffer or link mapping changed')
        points=np.asarray(bear_positions,dtype=float)
        if points.ndim!=2 or points.shape!=(self.bear_vertex_count,3) or not np.isfinite(points).all():
            raise ValueError('Need the complete finite current bear position array')
        transforms=np.asarray(self.data.ipc_transforms[0,self.indices],dtype=float).copy()
        if transforms.shape!=(3,4,4) or not np.isfinite(transforms).all() or not np.allclose(
                transforms[:,3,:],np.tile([0.,0.,0.,1.],(3,1)),rtol=0,atol=1e-10):
            raise RuntimeError('Invalid current native affine transforms')
        if np.any(np.linalg.det(transforms[:,:3,:3])<=0):
            raise RuntimeError('Inverted native affine collision body')
        minima=np.asarray([float(np.min(local@matrix[2,:3]+matrix[2,3]))
                            for local,matrix in zip(self.vertices,transforms)])
        bear_top=float(np.max(points[:,2]));hand_bottom=float(minima.min());gap=hand_bottom-bear_top
        row=dict(time_s=timestamp,physics_step=step,left_link_names=list(self.names),
            left_distal_ipc_transforms=transforms.tolist(),left_link_minimum_z_m=minima.tolist(),
            left_distal_minimum_z_m=hand_bottom,bear_vertex_count=len(points),bear_maximum_z_m=bear_top,
            highest_bear_vertex_index=int(np.argmax(points[:,2])),vertical_gap_m=gap,
            minimum_gap_m=self.minimum_gap_m,scope='Current native IPC affine collision surfaces versus all current bear vertices')
        reason='Complete teddy is not below native left palm and both fingers by the required height allowance' if gap<self.minimum_gap_m else None
        row['stop_reason']=reason;self._last_step=step;self._last_time=timestamp
        return row,reason

    def report(self):
        return dict(kind='native_left_distal_hand_height_monitor',minimum_gap_m=self.minimum_gap_m,
            model=self.model,cached_reference_geometry=self.geometry_metadata,bear_vertex_count=self.bear_vertex_count,
            native_transform_source='IPCCouplingData.ipc_transforms populated by AffineBodyStateAccessorFeature after completed IPC advance',
            vertex_source='Verified native vertex subset for the fixed hand alone inside link7; all native vertices for each finger. Arm link7 geometry is excluded.',
            native_affine_preserved=True,modifies_state_or_forces=False,
            scope='Separation of complete teddy below the left distal hand only; other arm links, rigid robot paths, cloth passage and grip need their existing guards.',
            limitations=['Recorded integration states only; no independent continuous-time guarantee between samples.',
                         'Requires reference_geometry() to be saved with the trace for independent per-row reconstruction.',
                         'A valid height gap does not establish teddy containment or successful cloth passage.'])
