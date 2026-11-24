"""CPU geometry screening for a bounded native cloth-contact trial.

Frozen cloth is not treated as a rigid obstacle for bear vertices. Their segment
crossings are predicted contact diagnostics, never a cloth-safe certificate.
The mass COM must enter only through the analysis mouth cap, without crossing
any actual cloth triangle. Every candidate still needs robot/left-soft checks,
unchanged native physics guards, and a fresh actual release review.
"""
from __future__ import annotations

import numpy as np
import trimesh

from validate import containment, mesh_edges, mouth_structure, signed_volumes, winding

SCOPE = ("Frozen actual cloth and translated actual teddy geometry only. "
         "Vertex/COM segments do not cover all swept tetrahedral, edge or face interiors. "
         "Vertex cloth intersections predict contact, not permitted native penetration. "
         "No continuous cloth-safe certificate, native execution, release authorization "
         "or physical success is established.")
NUMERICAL_GUARD_M = 1e-7


def _intersections(origins, delta, vertices, faces):
    """Finite Moller-Trumbore hits plus conservative uncertainty flags.

    Parallel/coplanar, degenerate and endpoint/edge cases near the segment are
    flagged rather than silently counted as clear. For vertices these flags are
    diagnostic. The COM hard gate rejects any such uncertainty.
    """
    tri = vertices[faces]
    e1, e2 = tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    normal = np.cross(e1, e2)
    normal_length = np.linalg.norm(normal, axis=1)
    good = normal_length > 1e-16
    nhat = np.zeros_like(normal)
    nhat[good] = normal[good] / normal_length[good, None]
    denominator = nhat @ delta
    parallel = abs(denominator) <= 1e-12
    pvec = np.cross(np.broadcast_to(delta, e2.shape), e2)
    determinant = np.einsum('ij,ij->i', e1, pvec)
    valid = good & ~parallel
    inv = np.zeros_like(determinant)
    inv[valid] = 1/determinant[valid]
    count = np.zeros(len(origins), dtype=int)
    uncertain = np.zeros(len(origins), dtype=bool)
    first = np.full(len(origins), np.inf)
    hit_details = []
    for start in range(0, len(origins), 32):
        points = origins[start:start+32]
        relative = points[:, None] - tri[None, :, 0]
        u = np.einsum('bti,ti->bt', relative, pvec) * inv
        q = np.cross(relative, e1[None])
        vv = np.einsum('i,bti->bt', delta, q) * inv
        t = np.einsum('ti,bti->bt', e2, q) * inv
        bary_epsilon = 1e-9
        hits = (valid[None] & (u >= -bary_epsilon) & (vv >= -bary_epsilon)
                & (u+vv <= 1+bary_epsilon) & (t >= -1e-8) & (t <= 1+1e-8))
        count[start:start+len(points)] = hits.sum(1)
        first[start:start+len(points)] = np.where(hits, t, np.inf).min(1)
        ii, jj = np.where(hits)
        hit_details.extend((int(start+a), int(b), float(t[a, b])) for a, b in zip(ii, jj))
        seg_low = np.minimum(points, points+delta)[:, None] - NUMERICAL_GUARD_M
        seg_high = np.maximum(points, points+delta)[:, None] + NUMERICAL_GUARD_M
        aabb = ((tri.min(1)[None] <= seg_high).all(2)
                & (tri.max(1)[None] >= seg_low).all(2))
        distance_to_plane = abs(np.einsum('bti,ti->bt', relative, nhat))
        uncertain_near_plane = aabb & ((~good)[None] | (
            parallel[None] & (distance_to_plane <= NUMERICAL_GUARD_M)))
        edge_or_endpoint = hits & ((np.minimum(np.minimum(u, vv), 1-u-vv) < 1e-8)
                                  | (t < 1e-8) | (t > 1-1e-8))
        uncertain[start:start+len(points)] = (uncertain_near_plane | edge_or_endpoint).any(1)
    return dict(count=count, first=first, uncertain=uncertain, hits=hit_details)


def _com_path(center, delta, vertices, physical_faces, caps, source_inside):
    physical = _intersections(center[None], delta, vertices, physical_faces)
    cap = _intersections(center[None], delta, vertices, caps)
    # A line can cross a folded analysis cap several times. Reject ambiguous
    # repeated enter/exit patterns instead of inferring continuous inclusion.
    cap_t = sorted(t for _, _, t in cap['hits'])
    unique_t = []
    for t in cap_t:
        if not unique_t or abs(t-unique_t[-1]) > 1e-8:
            unique_t.append(t)
    accepted = bool(physical['count'][0] == 0 and not physical['uncertain'][0]
                    and not cap['uncertain'][0]
                    and len(unique_t) == (0 if source_inside else 1))
    return dict(admissible=accepted,
                source_COM_inside=bool(source_inside),
                physical_cloth_triangle_crossing_count=int(physical['count'][0]),
                physical_cloth_crossing=bool(physical['count'][0]),
                physical_cloth_intersection_indeterminate=bool(physical['uncertain'][0]),
                analysis_cap_unique_crossing_fractions=unique_t,
                analysis_cap_intersection_indeterminate=bool(cap['uncertain'][0]),
                expected_mode='remain_inside' if source_inside else 'enter_once_through_mouth_cap',
                includes_all_actual_cloth_triangles_and_handles=True)


def _node_contact_prediction(points, delta, vertices, body_faces, physical_faces,
                             caps, inside_start, inside_end):
    physical = _intersections(points, delta, vertices, physical_faces)
    body = _intersections(points, delta, vertices, body_faces)
    cap = _intersections(points, delta, vertices, caps)
    entering = ~inside_start & inside_end
    groups = {}
    for name, mask in [('newly_inside', entering), ('already_inside', inside_start),
                       ('all_nodes', np.ones(len(points), bool))]:
        groups[name] = dict(node_count=int(mask.sum()),
            physical_cloth_crossing_nodes=int((mask & (physical['count'] > 0)).sum()),
            body_crossing_nodes=int((mask & (body['count'] > 0)).sum()),
            cap_crossing_nodes=int((mask & (cap['count'] > 0)).sum()),
            cap_only_no_physical_cloth_crossing_nodes=int((mask & (cap['count'] > 0)
                                                        & (physical['count'] == 0)).sum()),
            physical_cloth_indeterminate_nodes=int((mask & physical['uncertain']).sum()))
    return dict(kind='frozen_vertex_segment_contact_prediction', groups=groups,
                newly_inside_physical_cloth_crossing_indices=np.flatnonzero(
                    entering & (physical['count'] > 0)).tolist(),
                predicted_cloth_contact=bool((physical['count'] > 0).any()),
                body_crossings_are_diagnostic_not_a_rigid_wall_rejection=True,
                continuous_cloth_safe_certificate=False, scope=SCOPE)


def rank_recovery_directions(rest, state, config, *, maximum_xy_m=.02,
                             maximum_down_m=.02):
    """Screen at most 32 exact command deltas; do not rescale returned evidence.

    Each radius/down list is {10 mm, supplied maximum}, with duplicates removed.
    Caller may pass 19 mm to reserve its 1 mm measured-motion tracking budget.
    This function does not know or reset prior cumulative native XY/down use.
    """
    for value in (maximum_xy_m, maximum_down_m):
        if isinstance(value, bool) or not np.isscalar(value) or not np.isfinite(value) \
                or not 0 < value <= .02:
            raise ValueError('Recovery geometry limits must be positive and at most 20 mm')
    bag, bear = np.asarray(state['bag'], float), np.asarray(state['bear'], float)
    tcp = np.asarray(state['tcp_right'], float)
    time = float(state['time'])
    if bag.shape != rest['bag'].shape or bear.shape != rest['bear'].shape \
            or tcp.shape != (3,) or not np.isfinite(np.concatenate((bag.ravel(), bear.ravel(), tcp, [time]))).all():
        raise ValueError('Nonfinite or mismatched actual recovery observation')
    body, rim, topology = mouth_structure(rest['bag'], rest['bag_faces'], config)
    if not len(rim) or not topology['two_panels_directly_sewn_along_u']:
        raise ValueError('Recovery requires verified actual two-panel body topology')
    tetra = np.asarray(rest['bear_tets'])
    weights = abs(signed_volumes(rest['bear'], tetra))
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('Invalid native rest mass weights')
    current = containment(bag, bear, body, rim, weights, tetra)
    if not current['virtual_closed_body_valid']:
        raise ValueError('Invalid actual body and nonplanar analysis cap')
    center = np.asarray(current['bear_center_of_mass_m'])
    vertices = np.vstack((bag, bag[rim].mean(0)))
    _, _, boundary = mesh_edges(body)
    caps = np.column_stack((boundary[:, 1], boundary[:, 0], np.full(len(boundary), len(bag))))
    closed_faces = np.vstack((body, caps))
    physical_faces = np.asarray(rest['bag_faces'])
    mesh = trimesh.Trimesh(vertices, closed_faces, process=False)
    inside0 = abs(winding(bear, vertices, closed_faces)) > .5
    threshold = float(config['task']['contact_trial']['gravity_release']['minimum_inside_vertex_fraction'])
    if not np.isfinite(threshold) or not .5 <= threshold <= 1.:
        raise ValueError('Recovery must keep original at-least-50-percent release threshold')
    from contact_insertion import _policy
    forecast_com_clearance = float(_policy(config)['com_boundary_clearance_m'])
    table_required = float(config['table']['height']) + float(config['task']['contact_trial']['bottom_clearance_m'])
    if not np.isfinite(table_required):
        raise ValueError('Invalid original table policy')
    radii = sorted(set([min(.01, float(maximum_xy_m)), float(maximum_xy_m)]))
    downs = sorted(set([min(.01, float(maximum_down_m)), float(maximum_down_m)]))
    candidates, rejected = [], []
    for radius in radii:
        for down in downs:
            for angle in np.arange(8)*np.pi/4:
                delta = np.array([radius*np.cos(angle), radius*np.sin(angle), -down])
                delta[abs(delta) < 1e-15] = 0.
                base = dict(delta_m=delta.tolist(), target_tcp_m=(tcp+delta).tolist(),
                            translation_length_m=float(np.linalg.norm(delta)),
                            observed_time_s=time, physical_success=False, release_authorized=False)
                table_margin = float(bear[:, 2].min()+delta[2]-.001-table_required)
                base['table_path_review'] = dict(original_required_min_z_m=table_required,
                    allowance_m=.001, conservative_margin_m=table_margin,
                    straight_frozen_translation_bound=True)
                if table_margin < 0:
                    rejected.append(dict(**base, reason='Original table clearance plus 1 mm allowance fails'))
                    continue
                path = _com_path(center, delta, vertices, physical_faces, caps,
                                 current['bear_center_of_mass_inside'])
                if not path['admissible']:
                    rejected.append(dict(**base, reason='COM path does not certify cap-only entry or continued inclusion',
                                         COM_path=path))
                    continue
                predicted = containment(bag, bear+delta, body, rim, weights, tetra)
                if not predicted['virtual_closed_body_valid'] or not predicted['bear_center_of_mass_inside'] \
                        or predicted['bear_vertex_inside_fraction'] < threshold \
                        or predicted['bear_vertex_inside_fraction'] <= current['bear_vertex_inside_fraction']:
                    rejected.append(dict(**base, reason='Actual-shape frozen endpoint does not improve and meet original capture gate',
                                         COM_path=path, containment=predicted))
                    continue
                inside1 = abs(winding(bear+delta, vertices, closed_faces)) > .5
                _, distance, _ = trimesh.proximity.closest_point_naive(mesh, (center+delta)[None])
                if float(distance[0]) < forecast_com_clearance:
                    rejected.append(dict(**base,
                        reason='Frozen COM endpoint is within original 2 mm boundary margin',
                        COM_path=path, containment=predicted,
                        COM_inside_boundary_distance_m=float(distance[0]),
                        required_forecast_COM_boundary_clearance_m=forecast_com_clearance))
                    continue
                contact = _node_contact_prediction(bear, delta, vertices, body, physical_faces, caps, inside0, inside1)
                contact['COM_path'] = path
                base.update(kind='bounded_native_contact_trial_geometry_candidate',
                    prediction=dict(containment=predicted, current_containment=current,
                                    capture_fraction_improvement=float(predicted['bear_vertex_inside_fraction']-current['bear_vertex_inside_fraction']),
                                    COM_inside_boundary_distance_m=float(distance[0]),
                                    required_forecast_COM_boundary_clearance_m=forecast_com_clearance,
                                    frozen_forecast_only=True, physical_success=False),
                    contact_prediction=contact, scope=SCOPE)
                candidates.append(base)
    candidates.sort(key=lambda x:(
        -x['prediction']['containment']['bear_vertex_inside_fraction'],
        -x['prediction']['COM_inside_boundary_distance_m'],
        round(x['translation_length_m'], 12)))
    return dict(kind='bounded_native_contact_trial_geometry_screen', observed_time_s=time,
                candidates=candidates, rejections=rejected, current_containment=current,
                evaluated_candidate_count=len(candidates)+len(rejected),
                maximum_command_xy_m=float(maximum_xy_m), maximum_command_down_m=float(maximum_down_m),
                required_forecast_COM_boundary_clearance_m=forecast_com_clearance,
                forecast_boundary_scope='3D closed-analysis-surface distance with the original 2 mm COM margin; actual release inside/fraction rules unchanged',
                ranking='greatest capture fraction after original 2 mm COM margin, then COM boundary distance, then shortest displacement',
                requires_external_original_robot_and_left_soft_checks=True,
                requires_unchanged_native_guards_and_fresh_actual_release_review=True,
                physical_success=False, release_authorized=False, scope=SCOPE)
