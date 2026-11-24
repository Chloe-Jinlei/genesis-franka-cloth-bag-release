"""Generate an initially tabletop-flat, two-panel cloth tote, without supports.

This creates NEW files only. The native mesh itself defines the undeformed
material metric, avoiding an upright-rest mesh force-flattened into prestrain.
Body topology and sewn edges are inherited from the previous accepted tote.
The upper strap has a small loose twist/curl at its tip for finite-size pads.
No particle is fixed, scripted, kinematically attached, or magnetically held.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import brentq
import trimesh
from assets_builder import topology, write_obj
from contact_grasp import cloth_grasp_frame

HERE = Path(__file__).resolve().parent
ASSETS = HERE / 'assets'
SOURCE = HERE
STEM = 'flat_two_panel_bag_coarse'
METADATA = 'metadata_flat_two_panel_bag_coarse.json'
LAYER_GAP = .0024
CURL_HEIGHT = .025
WIDTH, HEIGHT = .46, .40
POSITION = [-.13, .05, .742]


def handle_path(extra, *, curled):
    # Same 37 transverse rows and 25 mm width as the original long handles.
    r = .12
    arc = np.linspace(np.pi, 0., 19)
    xy = np.r_[np.column_stack([np.full(9, -r), np.linspace(0., extra, 10)[:-1]]),
               np.column_stack([r*np.cos(arc), extra+r*np.sin(arc)]),
               np.column_stack([np.full(9, r), np.linspace(extra, 0., 10)[1:]])]
    planar_s = np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    phase = np.pi * planar_s / planar_s[-1]
    amplitude = np.sin(phase)**2 if curled else np.zeros(len(phase))
    amplitude[[0, -1]] = 0.
    z = (LAYER_GAP if curled else 0.) + CURL_HEIGHT * amplitude
    return np.column_stack([xy, z]), amplitude


def main():
    outputs = [ASSETS / (STEM+s) for s in ('.obj','.mtl','_collision.obj','.npz')]
    outputs += [ASSETS/METADATA, ASSETS/'flat_bag_geometry_validation.json']
    if any(p.exists() for p in outputs):
        raise FileExistsError('New flat assets already exist; choose another stem to preserve them.')
    source_file = SOURCE/'assets/two_panel_bag_wide_tall_long_handles_coarse.npz'
    old = dict(np.load(source_file, allow_pickle=False))
    source_metadata = json.loads((SOURCE/'assets/metadata_two_panel_wide_tall_long_handles_coarse.json').read_text())
    p = old['points'].copy()
    faces = old['faces'].copy()
    mx, mh = old['material_x'], old['material_height']
    # U seam is one shared line; 2.4 mm is temporary layer clearance, NOT a
    # sewn side/bottom panel or volumetric gusset. No extra triangles are added.
    for name, upper in [('front', True), ('back', False)]:
        grid = old['panel_'+name+'_grid']
        for j, h in enumerate(mh):
            for i, x in enumerate(mx):
                seam = j == 0 or i in (0, len(mx)-1)
                z = LAYER_GAP/2 if seam else (LAYER_GAP if upper else 0.)
                p[grid[j,i]] = [x, h-HEIGHT, z]
    handle_info = {}
    handle_grids = {}
    for name, curled in [('front', True), ('back', False)]:
        root0 = old['handle_'+name+'_root_left']
        root1 = old['handle_'+name+'_root_right'][::-1]
        interior = np.setdiff1d(old['handle_'+name], np.r_[root0, root1])
        strip = np.vstack([root0, interior.reshape(-1, len(root0)), root1])
        length = source_metadata['bag']['handles'][name]['centerline_length_m']
        extra = brentq(lambda e: np.linalg.norm(np.diff(handle_path(e,curled=curled)[0],axis=0),axis=1).sum()-length, .10,.22)
        path, amplitude = handle_path(extra,curled=curled)
        assert strip.shape[0] == len(path)
        tangent = np.gradient(path, axis=0)
        tangent[0] = [0,1,0]; tangent[-1] = [0,-1,0]
        tangent /= np.linalg.norm(tangent,axis=1)[:,None]
        flat_width = np.column_stack([tangent[:,1],-tangent[:,0],np.zeros(len(path))])
        flat_width /= np.linalg.norm(flat_width,axis=1)[:,None]
        orthogonal = np.cross(flat_width,tangent)
        orthogonal /= np.linalg.norm(orthogonal,axis=1)[:,None]
        twist = np.pi/2 * amplitude**2
        width_axis = np.cos(twist)[:,None]*flat_width + np.sin(twist)[:,None]*orthogonal
        cross = mx[np.abs(mx+.12)<=.0125+1e-9]+.12
        for j in range(1,len(path)-1):
            p[strip[j]] = path[j]+cross[:,None]*width_axis[j]
        handle_grids['handle_'+name+'_grid'] = strip.astype(np.int32)
        handle_info[name] = {'root_center_left_m':p[root0].mean(axis=0).tolist(),
          'root_center_right_m':p[root1].mean(axis=0).tolist(),
          'centerline_length_m':float(np.linalg.norm(np.diff(path,axis=0),axis=1).sum()),
          'flat_leg_extra_m':float(extra),'cross_samples':len(cross),
          'width_m':.025,'tip_center_height_m':float(path[len(path)//2,2]),
          'tip_width_twist_degrees':float(np.degrees(twist.max())),
          'fixed_vertices':0,'role':'loose curl for contact-only pinch' if curled else 'laid flat under upper handle'}
    # Preserve physical face ordering and named vertex groups, with additional
    # explicit row grids for geometric validation and contact planning.
    new = {**old, **handle_grids, 'points':p, 'faces':faces}
    np.savez_compressed(ASSETS/(STEM+'.npz'), **new)
    uv = np.column_stack([(p[:,0]+WIDTH/2)/WIDTH, (p[:,1]+HEIGHT)/.72])
    write_obj(ASSETS/(STEM+'.obj'),p,faces,uv,'cotton_canvas.png')
    write_obj(ASSETS/(STEM+'_collision.obj'),p,faces)
    stats = topology(p,faces)
    body_stats = topology(p[old['body']],faces[old['body_face_indices']])
    groups = {k:v.tolist() for k,v in old.items() if v.ndim==1 and np.issubdtype(v.dtype,np.integer) and k!='body_face_indices'}
    body = p[old['body']]
    panel_metrics = {}
    for name in ('front','back'):
        grid = p[old['panel_'+name+'_grid']]
        row = np.linalg.norm(np.diff(grid,axis=1),axis=-1).sum(axis=1)
        col = np.linalg.norm(np.diff(grid,axis=0),axis=-1).sum(axis=0)
        panel_metrics[name] = {'row_length_range_m':[float(row.min()),float(row.max())],
          'column_length_range_m':[float(col.min()),float(col.max())],
          'maximum_row_excess_fraction':float(np.max(row/WIDTH-1)),
          'maximum_column_excess_fraction':float(np.max(col/HEIGHT-1)),
          'nominal_flat_panel_area_m2':WIDTH*HEIGHT}
    frame = cloth_grasp_frame(p,faces,new['grasp_front_center'],preferred_approach=(0,0,-1),shell_radius=.0003)
    bfront,bback = old['panel_front_grid'],old['panel_back_grid']
    corresponding_gap = np.linalg.norm(p[bfront[1:,1:-1]]-p[bback[1:,1:-1]],axis=-1)
    checks = {'same_vertex_and_face_count':p.shape==old['points'].shape and np.array_equal(faces,old['faces']),
      'all_original_named_group_indices_preserved':all(np.array_equal(new[k],old[k]) for k in groups),
      'two_panel_shared_U_seam':all(np.array_equal(bfront[sl],bback[sl]) for sl in ((0,slice(None)),(slice(None),0),(slice(None),-1))),
      'only_two_panels_in_body':np.array_equal(np.sort(np.r_[old['panel_front_face_indices'],old['panel_back_face_indices']]),old['body_face_indices']),
      'body_lies_within_2p4mm_tabletop_slab':bool(np.ptp(body[:,2])<=LAYER_GAP+1e-12),
      'zero_bottom_seam_width_or_area':bool(np.ptp(p[old['bottom_seam'],1])==0 and np.ptp(p[old['bottom_seam'],2])==0),
      'one_connected_manifold_three_open_loops':stats['connected_components']==1 and stats['nonmanifold_edges']==0 and stats['boundary_loops']==3 and stats['consistent_winding'],
      'no_degenerate_triangles':stats['degenerate_triangles']==0,
      'no_local_vertices_below_table_plane':bool(p[:,2].min()>=-1e-12),
      'corresponding_nonseam_panel_gap_exceeds_2shellradii':bool(corresponding_gap.min()>.0006),
      'original_handle_centerline_length_preserved':all(abs(handle_info[s]['centerline_length_m']-source_metadata['bag']['handles'][s]['centerline_length_m'])<1e-10 for s in ('front','back')),
      'nominal_panel_dimensions_within_0p02percent':all(max(v['maximum_row_excess_fraction'],v['maximum_column_excess_fraction'])<.0002 for v in panel_metrics.values())}
    # Complete sampled geometric triangle-intersection / clearance requires
    # native IPC; this report intentionally does not claim its equivalent.
    report={'status':'CPU_geometry_only_native_pending','physical_success':False,
      'checks':checks,'all_checks_passed':all(checks.values()),'mesh':stats,'body_mesh':body_stats,
      'initial_body_world_z_range_m':(np.array(POSITION)[2]+np.array([body[:,2].min(),body[:,2].max()])).tolist(),
      'initial_total_world_bounds_m':(np.array(stats['bounds_m'])+np.array(POSITION)).tolist(),
      'panel_metrics':panel_metrics,'single_grasp_group':'grasp_front_center',
      'single_grasp_local_frame':frame.as_dict(),
      'suggested_bag_position_m':POSITION,
      'suggested_grasp_world_m':(frame.position+np.array(POSITION)).tolist(),
      'source_asset_sha256':hashlib.sha256(source_file.read_bytes()).hexdigest(),
      'rest_metric':'New mesh geometry itself is the material rest configuration. No upright-state mesh is force-flattened or used as a deformation target.',
      'limitations':['No native IPC initialization or dynamic retention proof.','Small tip curl is unconstrained; gravity can flatten it before approach. Contact timing must be validated.','One held handle does not guarantee a usable mouth: measure the actual deformed opening before bear insertion.','Corresponding-node separation is not a full nonadjacent triangle clearance proof.']}
    bag={'file':STEM+'.obj','npz_file':STEM+'.npz','collision_file':STEM+'_collision.obj',
      'construction':'Exactly two cloth panels U-sewn along left/bottom/right; no side panels, bottom panel, gusset, rigid support or fixed particles.',
      'width_m':WIDTH,'height_m':HEIGHT,'depth_m':0.,'top_opening_center_gap_m':LAYER_GAP,
      'bottom_seam_transverse_depth_m':0.,'bottom_seam_has_area':False,'strap_width_m':.025,
      'strap_peak_z_m':float(p[:,2].max()),'mesh':stats,'body_mesh':body_stats,'vertex_groups':groups,
      'body_face_indices':old['body_face_indices'].tolist(),'panel_metrics':panel_metrics,'handles':handle_info,
      'grasp_centers_local_m':{k:p[v].mean(axis=0).tolist() for k,v in groups.items() if k.startswith('grasp')},
      'initial_pose':'Body flat in horizontal XY plane; two loose layered handles; upper handle has 25 mm high centreline curl at tip.',
      'geometry':{'target_flat_width_m':WIDTH,'target_flat_height_m':HEIGHT,'body_thickness_envelope_m':LAYER_GAP,'initial_panel_normal_axis':'z','body_local_y_range_m':[-HEIGHT,0.]},
      'fixed_particles':0,'hidden_rigid_support':False,
      'suggested_material':{'name':'full40 cotton-linen effective material','E':1e7,'nu':.3,'rho':333.3333333333333,'thickness_radius_m':.0003,'bending_stiffness':1e-5,'friction_mu':.65,'shear_stiffness':50.,'handle_shear_stiffness':350.},
      'estimated_mass_kg':float(stats['triangle_area_m2']*.2)}
    (ASSETS/METADATA).write_text(json.dumps({'units':'metres','generator':'build_flat_bag.py','bag':bag,'bear':source_metadata['bear'],'validation':report},indent=2)+'\n')
    (ASSETS/'flat_bag_geometry_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    assert all(checks.values()), checks
    print(json.dumps({'files':[str(p) for p in outputs],'checks':checks,'suggested_grasp_world_m':report['suggested_grasp_world_m'],'single_grasp_local_frame':report['single_grasp_local_frame'],'bounds':stats['bounds_m'],'panel_metrics':panel_metrics},indent=2))

if __name__=='__main__':main()
