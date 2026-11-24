"""Independent saved-state audit. No Genesis import and no GPU work.

The virtual mouth cap below is an analysis surface only, never a simulation
constraint. Passing geometric samples is not a proof of continuous contact.
Task completion is never inferred from a successful solver exit.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent


def signed_volumes(points, tetra):
    return np.linalg.det(points[tetra[:, 1:]] - points[tetra[:, :1]]) / 6.0


def mesh_edges(faces):
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    unique, first, count = np.unique(np.sort(directed, axis=1), axis=0,
                                     return_index=True, return_counts=True)
    return unique, count, directed[first[count == 1]]


def boundary_loops(faces):
    _, counts, boundary = mesh_edges(faces)
    neighbours = {}
    for a, b in boundary:
        neighbours.setdefault(int(a), []).append(int(b))
        neighbours.setdefault(int(b), []).append(int(a))
    if not neighbours or any(len(v) != 2 for v in neighbours.values()):
        return [], False
    loops, visited = [], set()
    for start in neighbours:
        if start in visited:
            continue
        loop, previous, current = [], None, start
        while current not in visited:
            loop.append(current)
            visited.add(current)
            nxt = next(n for n in neighbours[current] if n != previous)
            previous, current = current, nxt
        if current != start:
            return [], False
        loops.append(np.asarray(loop, dtype=int))
    return loops, bool(np.all(counts <= 2))


def winding(points, vertices, faces, chunk=32):
    """Solid-angle winding of an oriented triangle mesh, in bounded batches."""
    tri = vertices[faces]
    result = np.zeros(len(points))
    for start in range(0, len(points), chunk):
        batch = points[start:start+chunk]
        a, b, c = (tri[None, :, k] - batch[:, None] for k in range(3))
        la, lb, lc = (np.linalg.norm(q, axis=-1) for q in (a, b, c))
        numerator = np.einsum("...i,...i->...", a, np.cross(b, c))
        denominator = (la*lb*lc + np.einsum("...i,...i->...", a,b)*lc
                       + np.einsum("...i,...i->...", b,c)*la
                       + np.einsum("...i,...i->...", c,a)*lb)
        result[start:start+len(batch)] = np.arctan2(numerator, denominator).sum(1)/(2*np.pi)
    return result


def tetra_boundary(points, tetra):
    face = np.concatenate([tetra[:, [1,2,3]], tetra[:, [0,3,2]],
                           tetra[:, [0,1,3]], tetra[:, [0,2,1]]])
    opposite = np.concatenate([tetra[:, k] for k in range(4)])
    _, inverse, counts = np.unique(np.sort(face,axis=1), axis=0,
                                   return_inverse=True, return_counts=True)
    mask = counts[inverse] == 1
    face, opposite = face[mask].copy(), opposite[mask]
    tri = points[face]
    inward = np.einsum("ij,ij->i", np.cross(tri[:,1]-tri[:,0], tri[:,2]-tri[:,0]),
                       points[opposite]-tri[:,0]) > 0
    face[inward] = face[inward][:, [0,2,1]]
    return face


def proper_segment_crossings(poly):
    count = 0
    def cross(a,b):
        return a[0]*b[1]-a[1]*b[0]
    n = len(poly)
    for i in range(n):
        a,b = poly[i],poly[(i+1)%n]
        for j in range(i+2,n):
            if i == 0 and j == n-1:
                continue
            c,d = poly[j],poly[(j+1)%n]
            if cross(b-a,c-a)*cross(b-a,d-a) < -1e-16 and cross(d-c,a-c)*cross(d-c,b-c) < -1e-16:
                count += 1
    return count


def _flat_start_enabled(config):
    value = config.get("task", {}).get("flat_start", {})
    if not isinstance(value, dict) or type(value.get("enabled", False)) is not bool:
        raise ValueError("task.flat_start must be an object with a boolean enabled flag")
    return value.get("enabled", False)


def _flat_source_topology(source, rest_bag, bag_faces, config):
    """Verify the two sewn grids without assigning meaning to world XYZ axes.

    Labels are evidence to check, not an assertion to trust: each grid must be
    exactly triangulated, the two body grids share precisely the U boundary,
    and the only remaining faces must triangulate the two rooted strap grids.
    This is a construction/topology certificate, not a self-contact test.
    """
    from scipy.spatial import cKDTree
    points = np.asarray(source["points"], float)
    faces = np.asarray(source["faces"])
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Invalid source points")
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in "iu" \
            or np.any(faces < 0) or np.any(faces >= len(points)):
        raise ValueError("Invalid source face indices")
    world = points + np.asarray(config["bag"]["position"], float)
    distances, mapping = cKDTree(rest_bag).query(world)
    if len(points) != len(rest_bag) or distances.max() > 1e-7 \
            or len(np.unique(mapping)) != len(points):
        raise ValueError("Source-to-rest mapping is not an exact vertex bijection")
    canonical = lambda f: {tuple(row) for row in np.sort(f, axis=1)}
    if len(canonical(faces)) != len(faces) or len(bag_faces) != len(faces) \
            or canonical(mapping[faces]) != canonical(bag_faces):
        raise ValueError("Source/imported triangle sets differ or contain duplicate triangles")
    triangle = rest_bag[mapping[faces]]
    if np.any(np.linalg.norm(np.cross(triangle[:, 1]-triangle[:, 0],
                                     triangle[:, 2]-triangle[:, 0]), axis=1) <= 1e-12):
        raise ValueError("Degenerate physical cloth triangle")

    def indices(name, upper=len(points), ndim=1):
        a = np.asarray(source[name])
        if a.dtype.kind not in "iu" or a.ndim != ndim or not a.size \
                or np.any(a < 0) or np.any(a >= upper) or len(np.unique(a)) != a.size:
            raise ValueError("Invalid or repeated indices in " + name)
        return a

    def grid_faces(grid, actual, name):
        if min(grid.shape) < 2:
            raise ValueError("Degenerate grid " + name)
        remaining = canonical(actual)
        for j in range(grid.shape[0]-1):
            for i in range(grid.shape[1]-1):
                a, b, c, d = grid[j, i], grid[j, i+1], grid[j+1, i], grid[j+1, i+1]
                options = (canonical([[a, b, d], [a, d, c]]),
                           canonical([[a, b, c], [b, d, c]]))
                found = next((pair for pair in options if pair <= remaining), None)
                if found is None:
                    raise ValueError("Missing or foreign triangle in " + name)
                remaining -= found
        if remaining or len(actual) != 2*(grid.shape[0]-1)*(grid.shape[1]-1):
            raise ValueError("Extra triangles in " + name)

    grids = {side: indices("panel_"+side+"_grid", ndim=2) for side in ("front", "back")}
    if grids["front"].shape != grids["back"].shape:
        raise ValueError("Body grids have different material dimensions")
    front, back = grids["front"], grids["back"]
    seams = {"bottom_seam": front[0], "seam_left": front[:, 0], "seam_right": front[:, -1]}
    for name, expected in seams.items():
        if not np.array_equal(indices(name), expected):
            raise ValueError("Incorrect material seam label " + name)
    if not all(np.array_equal(front[sl], back[sl]) for sl in
               ((0, slice(None)), (slice(None), 0), (slice(None), -1))):
        raise ValueError("Body panels do not share the same U-seam vertices")
    u = set(np.r_[front[0], front[:, 0], front[:, -1]])
    if set(front.ravel()) & set(back.ravel()) != u or set(indices("u_seam")) != u:
        raise ValueError("Body panels share vertices outside the U seam")
    body_ids = indices("body_face_indices", len(faces))
    panel_ids = {side: indices("panel_"+side+"_face_indices", len(faces)) for side in grids}
    if set(panel_ids["front"]) & set(panel_ids["back"]) \
            or set(np.r_[panel_ids["front"], panel_ids["back"]]) != set(body_ids):
        raise ValueError("Body faces are not exactly the two disjoint panel partitions")
    expected_body = set(front.ravel()) | set(back.ravel())
    if set(indices("body")) != expected_body:
        raise ValueError("Body vertex group disagrees with the two grids")
    panel_edges = {}
    for side, grid in grids.items():
        if set(indices("panel_"+side)) != set(grid.ravel()):
            raise ValueError("Panel vertex group disagrees with its grid")
        grid_faces(grid, faces[panel_ids[side]], side)
        panel_edges[side] = canonical(mesh_edges(faces[panel_ids[side]])[0])
    expected_u_edges = set()
    for chain in seams.values():
        expected_u_edges |= canonical(np.column_stack([chain[:-1], chain[1:]]))
    if panel_edges["front"] & panel_edges["back"] != expected_u_edges:
        raise ValueError("Panels share edges outside the direct U seam")
    ordered_rim = np.r_[front[-1], back[-1, -2:0:-1]]
    if set(indices("rim")) != set(ordered_rim) or len(np.unique(ordered_rim)) != len(ordered_rim):
        raise ValueError("Explicit rim does not match the two open top grid edges")
    _, _, body_boundary = mesh_edges(faces[body_ids])
    if canonical(body_boundary) != canonical(np.column_stack([ordered_rim, np.roll(ordered_rim, -1)])):
        raise ValueError("Body boundary contains a hole or an extra edge")

    # No unlabelled floor/side/support triangles can hide outside the body set.
    used_faces = set(body_ids)
    strap_vertices = set()
    for side in ("front", "back"):
        grid = indices("handle_"+side+"_grid", ndim=2)
        if strap_vertices & set(grid.ravel()):
            raise ValueError("The two separate straps share material vertices")
        strap_vertices |= set(grid.ravel())
        if set(indices("handle_"+side)) != set(grid.ravel()):
            raise ValueError("Strap vertex group disagrees with its grid")
        roots = [indices("handle_"+side+"_root_"+end) for end in ("left", "right")]
        if not np.array_equal(grid[0], roots[0]) or not np.array_equal(grid[-1], roots[1][::-1]):
            raise ValueError("Strap grid is not attached at the two declared roots")
        root_set = set(np.r_[*roots])
        if set(grid.ravel()) & expected_body != root_set or not root_set <= set(grids[side][-1]):
            raise ValueError("Strap connects to unintended body material")
        ids = np.flatnonzero(np.all(np.isin(faces, grid), axis=1))
        if used_faces & set(ids):
            raise ValueError("Strap/body face partitions overlap")
        grid_faces(grid, faces[ids], "handle_"+side)
        used_faces |= set(ids)
    if used_faces != set(range(len(faces))):
        raise ValueError("Faces outside the two body panels and the two straps")
    if expected_body | strap_vertices != set(range(len(points))):
        raise ValueError("Unlabelled or unused source vertices")
    mapped_body = mapping[faces[body_ids]]
    body_mesh = trimesh.Trimesh(rest_bag, mapped_body, process=False)
    full_mesh = trimesh.Trimesh(rest_bag, bag_faces, process=False)
    loops, manifold = boundary_loops(mapped_body)
    full_loops, full_manifold = boundary_loops(bag_faces)
    if len(loops) != 1 or not manifold or not full_manifold \
            or not body_mesh.is_winding_consistent or not full_mesh.is_winding_consistent \
            or len(full_mesh.split(only_watertight=False)) != 1:
        raise ValueError("Nonmanifold, disconnected or inconsistently oriented cloth")
    return mapped_body, mapping[ordered_rim], {
        "verified": True, "method": "Exact panel/strap grid triangulations, shared U edges, exhaustive face partitions and native vertex bijection",
        "source_to_rest_maximum_mapping_error_m": float(distances.max()),
        "panel_grid_shape": list(front.shape), "body_triangle_count": len(body_ids),
        "shared_u_seam_edge_count": len(expected_u_edges), "full_boundary_loop_count": len(full_loops),
        "independent_bottom_or_side_panel_faces": 0,
        "self_intersection_or_collision_clearance_certified": False}


def mouth_structure(rest_bag, bag_faces, config):
    if _flat_start_enabled(config):
        source_name = config.get("bag", {}).get("npz_file") or str(Path(config["bag"]["mesh"]).with_suffix(".npz"))
        source_path = Path(source_name)
        if not source_path.is_absolute():
            source_path = HERE/"assets"/source_path
        try:
            with np.load(source_path, allow_pickle=False) as source:
                body, rim, evidence = _flat_source_topology(source, rest_bag, bag_faces, config)
            valid, error = True, None
        except (OSError, ValueError, KeyError, IndexError, TypeError) as exc:
            body, rim = np.empty((0, 3), int), np.empty(0, int)
            valid, error, evidence = False, str(exc), {"verified": False}
        return body, rim, {
            "actual_open_body_boundary": valid, "body_boundary_loop_count": int(valid),
            "complete_bag_boundary_loop_count": evidence.get("full_boundary_loop_count"),
            "nonmanifold_edges": int((mesh_edges(bag_faces)[1] > 2).sum()),
            "connected_components": len(trimesh.Trimesh(rest_bag, bag_faces, process=False).split(only_watertight=False)),
            "body_triangle_count": len(body), "rim_rest_z": float(rest_bag[rim, 2].mean()) if valid else None,
            "body_face_selection": "Verified explicit material grids; no world-axis body-height fallback",
            "independent_flat_bottom_triangles": 0 if valid else None,
            "independent_flat_side_triangles": 0 if valid else None,
            "body_boundary_only_at_top": valid, "two_panels_directly_sewn_along_u": valid,
            "top_means_material_open_edge_not_world_z": True,
            "construction_evidence": evidence, "construction_error": error,
            "analysis_cap_is_not_simulated": True}
    metadata_path = HERE / "assets/metadata.json"
    bag_height = config.get("bag",{}).get("height_m")
    if bag_height is None:
        bag_height = json.loads(metadata_path.read_text())["bag"]["height_m"] if metadata_path.exists() else .32
    rim_z = float(np.min(rest_bag[:,2]) + bag_height)
    body_faces = bag_faces[np.all(rest_bag[bag_faces,2] <= rim_z+1e-7,axis=1)]
    body_source="geometric body-height separation"
    source_rim=None
    source_name=config.get("bag",{}).get("npz_file")
    if source_name is None:
        source_name=str(Path(config.get("bag",{}).get("mesh","bag.obj")).with_suffix(".npz"))
    source_path=Path(source_name)
    if not source_path.is_absolute():
        source_path=HERE/"assets"/source_path
    if source_path.exists():
        with np.load(source_path,allow_pickle=False) as source:
            if "body_face_indices" in source and "rim" in source:
                from scipy.spatial import cKDTree
                world=source["points"]+np.asarray(config["bag"]["position"])
                distances,mapping=cKDTree(rest_bag).query(world)
                if distances.max()>1e-7 or len(np.unique(mapping))!=len(world):
                    body_faces=np.empty((0,3),dtype=int)
                    body_source="source-to-imported mesh mapping failed; refusing geometric fallback"
                else:
                    candidate=mapping[source["faces"][source["body_face_indices"]]]
                    imported={tuple(v) for v in np.sort(bag_faces,axis=1)}
                    if all(tuple(v) in imported for v in np.sort(candidate,axis=1)):
                        body_faces=candidate
                        source_rim=mapping[source["rim"]]
                        rim_z=float(rest_bag[source_rim,2].mean())
                        body_source="explicit physical body-face labels mapped to imported rest mesh"
                    else:
                        body_faces=np.empty((0,3),dtype=int)
                        body_source="source body triangles absent from imported topology"
    body_loops, body_manifold = boundary_loops(body_faces)
    full_loops, full_manifold = boundary_loops(bag_faces)
    mesh = trimesh.Trimesh(rest_bag, bag_faces, process=False)
    body_mesh = trimesh.Trimesh(rest_bag, body_faces, process=False)
    valid = (len(body_loops) == 1 and body_manifold and full_manifold
             and mesh.is_winding_consistent and body_mesh.is_winding_consistent)
    tri=rest_bag[body_faces]
    bottom_panels=int(np.all(np.abs(tri[:,:,2]-rest_bag[:,2].min())<1e-7,axis=1).sum())
    side_panels=int((np.all(np.abs(tri[:,:,0]-rest_bag[:,0].min())<1e-7,axis=1)
                    |np.all(np.abs(tri[:,:,0]-rest_bag[:,0].max())<1e-7,axis=1)).sum())
    top_boundary=bool(valid and (set(body_loops[0])==set(source_rim) if source_rim is not None
                                else np.all(np.abs(rest_bag[body_loops[0],2]-rim_z)<1e-7)))
    two_panel=bool(valid and top_boundary and bottom_panels==0 and side_panels==0)
    report = {"actual_open_body_boundary":bool(valid),"body_boundary_loop_count":len(body_loops),
              "complete_bag_boundary_loop_count":len(full_loops),"nonmanifold_edges":int((mesh_edges(bag_faces)[1]>2).sum()),
              "connected_components":len(mesh.split(only_watertight=False)),
              "body_triangle_count":len(body_faces),"rim_rest_z":rim_z,
              "body_face_selection":body_source,
              "independent_flat_bottom_triangles":bottom_panels,"independent_flat_side_triangles":side_panels,
              "body_boundary_only_at_top":top_boundary,"two_panels_directly_sewn_along_u":two_panel,
              "analysis_cap_is_not_simulated":True}
    return body_faces, body_loops[0] if valid else np.empty(0,dtype=int), report


def containment(bag, bear, body_faces, rim, tetra_weights, tets):
    rim_points = bag[rim]
    center = rim_points.mean(0)
    _,_,basis = np.linalg.svd(rim_points-center,full_matrices=False)
    projected = (rim_points-center)@basis[:2].T
    area = abs(np.sum(projected[:,0]*np.roll(projected[:,1],-1)-projected[:,1]*np.roll(projected[:,0],-1)))/2
    crossings = proper_segment_crossings(projected)
    vertices = np.concatenate([bag,center[None]])
    _,_,directed_boundary = mesh_edges(body_faces)
    caps = np.column_stack([directed_boundary[:,1],directed_boundary[:,0],np.full(len(directed_boundary),len(bag))])
    faces = np.concatenate([body_faces,caps])
    surface = trimesh.Trimesh(vertices, faces, process=False)
    center_of_mass = np.sum(bear[tets].mean(1)*tetra_weights[:,None],axis=0)/tetra_weights.sum()
    valid = surface.is_watertight and surface.is_winding_consistent and crossings == 0 and area > 1e-5
    if valid:
        contained = np.abs(winding(np.vstack([bear,center_of_mass]),vertices,faces)) > .5
        fraction = float(contained[:-1].mean())
        com_inside = bool(contained[-1])
    else:
        fraction,com_inside = 0.,False
    return {"mouth_projected_area_m2":float(area),"mouth_projected_crossings":crossings,
            "mouth_projected_dimensions_m":np.ptp(projected,axis=0).tolist(),
            "mouth_nonplanarity_max_m":float(np.abs((rim_points-center)@basis[2]).max()),
            "virtual_closed_body_valid":bool(valid),"bear_vertex_inside_fraction":fraction,
            "bear_center_of_mass_m":center_of_mass.tolist(),"bear_center_of_mass_inside":com_inside,
            "all_bear_vertices_inside":bool(valid and fraction == 1. and com_inside)}


def final_contact_audit(bag, bear, faces, tets, tolerance=.001):
    """Check bag vertices/edge midpoints/face centroids against the bear volume."""
    edges,_,_ = mesh_edges(faces)
    samples = np.vstack([bag,bag[edges].mean(1),bag[faces].mean(1)])
    candidate = np.all((samples >= bear.min(0)-tolerance)&(samples <= bear.max(0)+tolerance),axis=1)
    samples = samples[candidate]
    surface = trimesh.Trimesh(bear,tetra_boundary(bear,tets),process=False)
    inside = np.abs(winding(samples,surface.vertices,surface.faces)) > .5
    depth = np.empty(0)
    if inside.any():
        _,depth,_ = trimesh.proximity.closest_point(surface,samples[inside])
    return {"method":"Discrete bag vertex, edge-midpoint and triangle-centroid winding/distance samples",
            "sample_count":len(candidate),"near_bear_samples":len(samples),
            "inside_samples":int(inside.sum()),"penetration_over_tolerance":int((depth>tolerance).sum()),
            "maximum_sampled_penetration_m":float(depth.max()) if len(depth) else 0.,
            "tolerance_m":tolerance,"continuous_collision_guarantee":False}


def bear_deformation_audit(rest_points, tetra, frames, time, q_right, events=None):
    """Report FEM inversion independently of whether the gripper lifted it.

    Principal stretches come from the actual deformation gradient. Percentiles
    are weighted by rest tet volume so small slivers cannot dominate the bulk.
    Stable Neo-Hookean and surface IPC contact do not prohibit tet inversion.
    """
    tetra = np.asarray(tetra, dtype=int)
    reference = np.asarray(rest_points)[tetra]
    dm_inv = np.linalg.inv(np.moveaxis(reference[:, 1:] - reference[:, :1], 1, 2))
    volume = np.abs(signed_volumes(np.asarray(rest_points), tetra))
    weights = volume / volume.sum()
    edge_length_sq = np.stack([np.sum((reference[:, j]-reference[:, i])**2, axis=1)
                               for i in range(4) for j in range(i+1, 4)], axis=1)
    quality = 12*(3*volume)**(2/3) / edge_length_sq.sum(axis=1)
    first_inversion, worst, final_quantiles = None, None, None
    max_stretch, max_stretched_fraction = 0., 0.
    samples = []
    for i, (points, timestamp) in enumerate(zip(frames, time)):
        deformed = points[tetra]
        gradient = np.moveaxis(deformed[:, 1:] - deformed[:, :1], 1, 2) @ dm_inv
        ratio = np.linalg.det(gradient)
        stretch = np.linalg.svd(gradient, compute_uv=False)
        count = int((ratio <= 0).sum())
        sample = {"time_s": float(timestamp), "minimum_volume_ratio": float(ratio.min()),
                  "inverted_tetrahedra": count,
                  "inverted_reference_volume_fraction": float(weights[ratio <= 0].sum()),
                  "right_finger_q_m": np.asarray(q_right[i])[-2:].tolist()}
        samples.append(sample)
        if count and first_inversion is None:
            first_inversion = sample
        if worst is None or sample["minimum_volume_ratio"] < worst["minimum_volume_ratio"]:
            k = int(np.argmin(ratio))
            worst = {**sample, "tetra_index": k,
                     "rest_center_m": reference[k].mean(axis=0).tolist(),
                     "rest_volume_m3": float(volume[k]),
                     "rest_mean_ratio_quality": float(quality[k]),
                     "rest_quality_percentile": float(100*(quality < quality[k]).mean())}
        max_stretch = max(max_stretch, float(stretch.max()))
        max_stretched_fraction = max(max_stretched_fraction, float(weights[stretch[:, 0] > 2.].sum()))
        if i == len(time)-1:
            final_quantiles = {}
            for name, values in (("smallest", stretch[:, -1]), ("largest", stretch[:, 0])):
                order = np.argsort(values)
                final_quantiles[name] = dict(zip(("p01", "p50", "p99"),
                    np.interp([.01, .5, .99], np.cumsum(weights[order]), values[order]).tolist()))
    phases = {}
    for event in events or []:
        start, end = event.get("start", 0), event.get("start", 0) + event.get("duration", 0)
        window = [s for s in samples if start-1e-8 <= s["time_s"] <= end+.011]
        if window:
            phases[event.get("phase", "unknown")] = {
                "minimum_volume_ratio": min(s["minimum_volume_ratio"] for s in window),
                "maximum_inverted_tetrahedra": max(s["inverted_tetrahedra"] for s in window)}
    return {"first_observed_inversion": first_inversion,
            "worst_volume_ratio_sample": worst,
            "maximum_principal_stretch": max_stretch,
            "maximum_reference_volume_fraction_with_principal_stretch_over_2": max_stretched_fraction,
            "final_volume_weighted_principal_stretch_quantiles": final_quantiles,
            "phase_deformation": phases,
            "positive_tetra_volume_in_all_saved_states": first_inversion is None,
            "rest_mean_ratio_quality_quantiles": dict(zip(("min", "p01", "p50", "p99"),
                np.quantile(quality, [0, .01, .5, .99]).tolist())),
            "material_realism_calibrated": False}


def audit_soft_metrics(metrics, frame_time, frame_metrics, joint_time, expected_dt):
    """Validate per-integration-step summaries against independently read frames."""
    if metrics is None:
        return {"provided": False, "valid": True,
                "sampling_scope": "Full soft geometry only at saved frame timestamps."}
    report = {"provided": True, "valid": False,
              "columns": ["time_s", "minimum_tetra_volume_ratio", "bag_table_penetration_m",
                          "bear_table_penetration_m", "maximum_bag_edge_stretch_ratio",
                          "all_soft_states_finite"],
              "sampling_scope": "Soft summary at each recorded integration step; full geometry at saved frames."}
    a = np.asarray(metrics, dtype=float)
    if a.ndim != 2 or a.shape[1] != 6 or len(a) < 3:
        return {**report, "errors": ["Expected at least three rows with six columns."]}
    errors = []
    report["rows"] = len(a)
    if not np.isfinite(a).all():
        errors.append("Nonfinite per-step metric; a nonfinite physical state must not pass.")
    if not np.all(a[:, 5] == 1.):
        errors.append("Not all per-step soft-state finite flags equal one.")
    time = a[:, 0]
    if not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        return {**report, "errors": errors + ["Per-step timestamps must be finite and strictly increasing."]}
    report["maximum_interval_s"] = float(np.diff(time).max())
    report["expected_integration_interval_s"] = float(expected_dt)
    if report["maximum_interval_s"] > expected_dt * 1.001 + 1e-9:
        errors.append("Per-step metric trace has an integration-step sampling gap.")
    if time[0] > frame_time[0] + 1e-8 or time[-1] < frame_time[-1] - 1e-8:
        errors.append("Per-step metrics do not cover the initial and final saved soft states.")

    def matching_indices(query):
        right = np.clip(np.searchsorted(time, query), 0, len(time)-1)
        left = np.maximum(0, right-1)
        return np.where(np.abs(time[right]-query) < np.abs(time[left]-query), right, left)

    frame_indices = matching_indices(frame_time)
    frames_match = bool(np.all(np.abs(time[frame_indices]-frame_time) <= 1e-8))
    joint_indices = matching_indices(joint_time)
    joints_match = bool(np.all(np.abs(time[joint_indices]-joint_time) <= 1e-8))
    if not frames_match:
        errors.append("One or more saved soft frames lack a matching per-step summary.")
    if not joints_match:
        errors.append("One or more actual joint-trace timestamps lack a matching per-step summary.")
    report["covers_all_saved_frame_timestamps"] = frames_match
    report["covers_all_joint_trace_timestamps"] = joints_match
    if frames_match:
        independent = np.asarray(frame_metrics)
        recorded = a[frame_indices, 1:5]
        agreement = np.isclose(recorded, independent, rtol=1e-5, atol=1e-7)
        report["matches_independent_saved_frame_recomputation"] = bool(agreement.all())
        report["maximum_shared_frame_absolute_error_by_metric"] = np.max(
            np.abs(recorded-independent), axis=0).tolist()
        if not agreement.all():
            errors.append("Recorded per-step metrics disagree with raw saved soft coordinates.")
    finite = np.isfinite(a[:, 1:5]).all(axis=1)
    if finite.any():
        b = a[finite]
        report["minimum_tetrahedron_volume_ratio"] = float(b[:, 1].min())
        report["maximum_table_penetration_m"] = {"bag": float(b[:, 2].max()), "bear": float(b[:, 3].max())}
        report["maximum_bag_edge_stretch_ratio"] = float(b[:, 4].max())
        report["worst_volume_ratio_time_s"] = float(b[np.argmin(b[:, 1]), 0])
        if np.any(b[:, 2:4] < 0) or np.any(b[:, 4] <= 0):
            errors.append("Negative penetration depths or nonpositive stretch metrics are invalid.")
    report["valid"] = not errors
    report["errors"] = errors
    return report


def audit_left_handle_hold(rest_bag, frames, config, events):
    """Track the named physical strap nodes after handoff, not any bag vertex.

    Negative PD is control evidence only. It is not a contact-force sensor and
    can also occur when fingers press each other. Named material proximity is
    therefore required independently; this still does not prove force closure.
    """
    flat = _flat_start_enabled(config)
    anchor_phase = "close_single_handle" if flat else "release_right_handle"
    release = next((e for e in events if e.get("phase") == anchor_phase), None)
    result = {"applicable": release is not None, "passed": False,
              "material_group_retained": False, "both_fingers_command_clamping": False,
              "distance_limit_m": .018, "pd_clamping_threshold_n": -.1,
              "distance_definition": "Nearest vertex belonging to the exact configured left grasp material group",
              "sampling_scope": "Every saved full-geometry frame after right-handle release starts",
              "force_evidence_scope": "Saved get_dofs_control_force PD values; not native contact force",
              "errors": []}
    if release is None:
        result["errors"].append("Single-left closure phase has not been recorded." if flat
                                else "Right-handle handoff phase has not been recorded.")
        return result
    time = np.asarray(frames["time"], float)
    anchor = float(release["start"])
    if flat:
        anchor += float(release["duration"])
        result.update(anchor_phase=anchor_phase, anchor_time_s=anchor,
                      sampling_scope="Every saved full-geometry frame from the end of single-left closure; includes actual lift and later hold")
        if not np.isfinite(anchor) or anchor > time[-1]+1e-9:
            result["applicable"] = False
            result["errors"].append("Single-left closure has not completed in the saved observation interval.")
            return result
    selected = np.flatnonzero(time >= anchor-1e-9)
    if not len(selected):
        result["errors"].append("No saved material state after handoff.")
        return result
    group = config.get("task",{}).get("handle_grasps",{}).get("left",{}).get("group", "grasp_front")
    source_name = config.get("bag",{}).get("npz_file")
    if source_name is None:
        source_name = str(Path(config.get("bag",{}).get("mesh","bag.obj")).with_suffix(".npz"))
    source_path = Path(source_name)
    if not source_path.is_absolute():
        source_path = HERE/"assets"/source_path
    result.update({"material_group": group, "source_npz": str(source_path),
                   "first_checked_time_s":float(time[selected[0]]),
                   "last_checked_time_s":float(time[selected[-1]]), "checked_frames":len(selected)})
    try:
        from scipy.spatial import cKDTree
        with np.load(source_path, allow_pickle=False) as source:
            points = np.asarray(source["points"], float)
            indices = np.asarray(source[group], int).reshape(-1)
            if not len(indices) or np.any(indices<0) or np.any(indices>=len(points)):
                raise ValueError("Invalid or empty named grasp group")
            world = points+np.asarray(config["bag"]["position"])
            distances, mapping = cKDTree(np.asarray(rest_bag)).query(world)
            if len(world)!=len(rest_bag) or distances.max()>1e-7 or len(np.unique(mapping))!=len(world):
                raise ValueError("Source-to-rest mapping is not an exact vertex bijection")
            mapped = np.unique(mapping[indices])
            result["source_to_rest_maximum_mapping_error_m"] = float(distances.max())
            result["mapped_material_vertex_indices"] = mapped.tolist()
            result["source_vertex_order_identical_to_rest"] = bool(np.array_equal(mapping,np.arange(len(mapping))))
        metadata_name = config.get("bag",{}).get("metadata")
        if metadata_name:
            metadata_path = Path(metadata_name)
            if not metadata_path.is_absolute():
                metadata_path = HERE/"assets"/metadata_path
            metadata = json.loads(metadata_path.read_text()).get("bag",{})
            if metadata.get("npz_file") and Path(metadata["npz_file"]).name != source_path.name:
                raise ValueError("Configured bag metadata names a different source NPZ")
            if metadata.get("file") and Path(metadata["file"]).name != Path(config["bag"]["mesh"]).name:
                raise ValueError("Configured bag metadata names a different mesh")
            result["metadata_file"] = str(metadata_path)
    except (OSError, ValueError, KeyError) as exc:
        result["errors"].append(f"Cannot verify named physical handle group: {exc}")
        return result
    material = np.asarray(frames["bag"],float)[selected][:,mapped]
    tcp = np.asarray(frames["tcp_left"],float)[selected]
    distances = np.linalg.norm(material-tcp[:,None,:],axis=2)
    nearest = distances.min(axis=1)
    centroid = np.linalg.norm(material.mean(axis=1)-tcp,axis=1)
    finite = np.isfinite(distances).all()
    result["maximum_nearest_group_distance_m"] = float(nearest.max())
    result["maximum_group_centroid_distance_m"] = float(centroid.max())
    result["maximum_farthest_group_vertex_distance_m"] = float(distances.max())
    result["material_group_retained"] = bool(finite and np.all(nearest <= result["distance_limit_m"]))
    violations = np.flatnonzero(~np.isfinite(nearest) | (nearest>result["distance_limit_m"]))
    result["first_material_distance_violation_s"] = float(time[selected[violations[0]]]) if len(violations) else None
    if "force_left" not in frames:
        result["errors"].append("Saved left-finger PD control forces are missing.")
    else:
        force = np.asarray(frames["force_left"],float)
        if force.ndim != 2 or force.shape != (len(time),9):
            result["errors"].append("Invalid saved left-finger PD control-force shape.")
        else:
            pd = force[selected,-2:]
            result["finger_pd_minimum_n"] = pd.min(axis=0).tolist()
            result["finger_pd_maximum_n"] = pd.max(axis=0).tolist()
            result["both_fingers_command_clamping"] = bool(np.isfinite(pd).all() and np.all(pd < result["pd_clamping_threshold_n"]))
            violations = np.flatnonzero(~np.isfinite(pd).all(axis=1) | np.any(pd>=result["pd_clamping_threshold_n"],axis=1))
            result["first_pd_clamping_violation_s"] = float(time[selected[violations[0]]]) if len(violations) else None
    result["passed"] = bool(result["material_group_retained"] and result["both_fingers_command_clamping"] and not result["errors"])
    return result


def audit_flat_start(rest_bag, body_faces, frames, config, events, joint_time, joint_values,
                     *, every_step_joint_trace):
    """Additional, explicitly scoped evidence for the single-left flat prefix.

    Open measured fingers and declared phases exclude the old right-hand
    clamping sequence; they do not establish zero incidental right contact.
    Native two-sided left reaction monitoring is a separate runtime record.
    """
    time = np.asarray(frames["time"], float)
    body = np.unique(body_faces)
    checks = {
        "flat_start_initial_body_flat_on_table": False,
        "flat_start_initial_hands_open": False,
        "flat_start_right_fingers_open_until_teddy_pregrasp": False,
        "flat_start_single_left_prefix_recorded": False,
        "flat_start_bag_lifted_before_teddy_grasp": False,
    }
    result = {"checks": checks, "body_flat_maximum_height_m": .012,
              "body_flat_maximum_thickness_extent_m": .012,
              "actual_body_minimum_rise_before_teddy_m": .04,
              "finger_open_minimum_m": .038, "errors": [],
              "right_hand_evidence_scope": "Actual open finger positions and phase records only; no proof of zero incidental cloth contact",
              "left_contact_force_evidence_scope": "Not available in these arrays; requires the separate every-step native reaction record",
              "open_mouth_or_insertability_proved_by_lift": False}
    if not len(body):
        result["errors"].append("Verified material body faces unavailable")
        return result
    initial = np.asarray(frames["bag"], float)[0, body]
    height = float(config["table"]["height"])
    maximum_height = float(initial[:, 2].max()-height)
    minimum_height = float(initial[:, 2].min()-height)
    extent = float(np.ptp(initial[:, 2]))
    initial_matches_rest = bool(abs(time[0]) <= 1e-9 and
        np.allclose(np.asarray(frames["bag"])[0], rest_bag, rtol=0, atol=1e-7))
    result.update(initial_body_maximum_height_above_table_m=maximum_height,
                  initial_body_minimum_height_above_table_m=minimum_height,
                  initial_body_z_extent_m=extent, initial_observation_is_reference_at_t0=initial_matches_rest)
    checks["flat_start_initial_body_flat_on_table"] = bool(initial_matches_rest and
        minimum_height >= -.002 and maximum_height <= .012 and extent <= .012)
    initial_fingers = np.r_[np.asarray(frames["q_left"])[0, -2:],
                            np.asarray(frames["q_right"])[0, -2:]]
    result["initial_actual_finger_positions_m"] = initial_fingers.tolist()
    checks["flat_start_initial_hands_open"] = bool(np.isfinite(initial_fingers).all() and
                                                  np.all(initial_fingers >= .038))
    names = [e.get("phase") for e in events]
    required = ("approach_single_handle", "close_single_handle", "lift_single_handle", "settle_single_handle")
    old = {"close_both_handles", "drape_under_gravity", "spread_bag", "release_right_handle", "right_clear_handle"}
    prefix_valid = all(names.count(p) == 1 for p in required) and not (old & set(names))
    by_phase = {e.get("phase"): e for e in events}
    try:
        if prefix_valid:
            starts = np.asarray([float(by_phase[p]["start"]) for p in required])
            durations = np.asarray([float(by_phase[p]["duration"]) for p in required])
            ends = starts+durations
            prefix_valid = bool(np.isfinite(starts).all() and np.isfinite(durations).all() and
                np.all(starts >= 0) and np.all(durations > 0) and
                np.all(np.diff([names.index(p) for p in required]) > 0) and
                np.all(starts[1:] >= ends[:-1]-1e-9) and ends[-1] <= time[-1]+1e-9)
            # Intermediate carry stages may only occur after closure and before
            # the final lift stage. They are actual recorded movements, not a
            # replacement for the completed lift/settle anchors.
            for event in events:
                if str(event.get("phase", "")).startswith("lift_single_handle_"):
                    start, duration = float(event["start"]), float(event["duration"])
                    prefix_valid &= bool(np.isfinite(start) and np.isfinite(duration) and duration > 0 and
                        start >= ends[1]-1e-9 and start+duration <= starts[2]+1e-9)
            result.update(left_material_hold_anchor_time_s=float(ends[1]),
                          left_stationary_anchor_time_s=float(starts[3]),
                          single_handle_settle_end_time_s=float(ends[3]))
        checks["flat_start_single_left_prefix_recorded"] = bool(prefix_valid)
        pregrasp = by_phase.get("teddy_pregrasp")
        limit = float(pregrasp["start"]) if pregrasp else float(time[-1])
        selected = np.asarray(joint_time) <= limit+1e-9
        right_fingers = np.asarray(joint_values["right"])[selected, -2:]
        coverage = bool(every_step_joint_trace and len(right_fingers) and
                        joint_time[0] <= float(config.get("simulation", {}).get("dt", .005))+1e-9 and
                        np.isfinite(limit) and joint_time[-1] >= limit-1e-9)
        checks["flat_start_right_fingers_open_until_teddy_pregrasp"] = bool(coverage and
            np.isfinite(right_fingers).all() and np.all(right_fingers >= .038))
        result.update(right_open_checked_through_s=limit, right_open_checked_samples=len(right_fingers),
                      right_open_uses_every_step_joint_trace=bool(every_step_joint_trace),
                      minimum_right_finger_position_before_teddy_m=float(right_fingers.min()) if len(right_fingers) else None)
        if prefix_valid:
            anchor = float(by_phase["settle_single_handle"]["start"])
            idx = int(np.argmin(np.abs(time-anchor)))
            frame_agrees = bool(abs(time[idx]-anchor) <= 1e-7)
            rise = float(np.asarray(frames["bag"])[idx, body, 2].max()-initial[:, 2].max())
            result.update(actual_body_maximum_z_rise_m=rise, actual_body_rise_frame_time_s=float(time[idx]),
                          lift_endpoint_saved_at_stationary_anchor=frame_agrees)
            teddy_after_settle = bool(pregrasp and float(pregrasp["start"]) >= ends[3]-1e-9 and
                                     names.index("teddy_pregrasp") > names.index("settle_single_handle"))
            result["lifted_and_settled_bag_before_any_teddy_pregrasp"] = teddy_after_settle
            # A partial open-only probe can report the real lift even though
            # the full-task sequence is deliberately still incomplete.
            checks["flat_start_bag_lifted_before_teddy_grasp"] = bool(frame_agrees and rise >= .04 and
                (teddy_after_settle if pregrasp else not any(p in names for p in
                    ("teddy_grasp_approach", "close_teddy_gripper", "lift_teddy"))))
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        result["errors"].append("Malformed flat-prefix event/evidence: " + str(exc))
        checks["flat_start_single_left_prefix_recorded"] = False
        checks["flat_start_bag_lifted_before_teddy_grasp"] = False
    result["passed"] = bool(all(checks.values()) and not result["errors"])
    return result


def audit_arrays(rest, frames, config, *, task_completed=False, hold_duration=.5, joint_trace=None, events=None,
                 soft_metrics=None):
    try:
        flat_start = _flat_start_enabled(config)
    except ValueError as exc:
        return {"status": "failed", "task_success": False, "reason": str(exc)}
    required=("time","bear","bag","q_left","q_right","tcp_left","tcp_right")
    missing=[k for k in required if k not in frames]
    if missing:
        return {"status":"incomplete","task_success":False,"missing_frame_arrays":missing}
    time=np.asarray(frames["time"],float).reshape(-1)
    if len(time)<3 or np.any(np.diff(time)<0):
        return {"status":"incomplete","task_success":False,"reason":"Need at least three strictly increasing saved timestamps."}
    # Phase snapshots may repeat the same last video timestamp. Keep the last
    # saved value at each timestamp, without silently sorting backwards time.
    keep=np.r_[np.diff(time)>1e-10,True]
    duplicate_count=int((~keep).sum())
    frames={k:np.asarray(frames[k])[keep] for k in frames
            if np.asarray(frames[k]).shape[:1] == (len(time),)}
    time=time[keep]
    if len(time)<3:
        return {"status":"incomplete","task_success":False,"reason":"Fewer than three distinct saved timestamps."}
    finite=all(np.isfinite(np.asarray(frames[k])).all() for k in required)
    if not finite:
        return {"status":"failed","task_success":False,"reason":"Nonfinite saved state."}
    bear0,bag0=np.asarray(rest["bear"]),np.asarray(rest["bag"])
    tets,faces=np.asarray(rest["bear_tets"],int),np.asarray(rest["bag_faces"],int)
    bears,bags=np.asarray(frames["bear"]),np.asarray(frames["bag"])
    if bears.shape[1:]!=bear0.shape or bags.shape[1:]!=bag0.shape or len(bears)!=len(time) or len(bags)!=len(time):
        return {"status":"failed","task_success":False,"reason":"Saved topology/state shapes disagree."}
    rest_vol=signed_volumes(bear0,tets)
    if np.any(np.abs(rest_vol)<1e-14):
        return {"status":"failed","task_success":False,"reason":"Degenerate reference tetrahedron."}
    weights=np.abs(rest_vol)
    tetra_ratio_min=1.
    inverted_volume_fraction=0.
    table_penetration={"bear":0.,"bag":0.}
    table_height=config["table"]["height"]
    half=np.asarray(config["table"]["size"][:2])/2
    edges,_,_=mesh_edges(faces)
    rest_length=np.linalg.norm(bag0[edges[:,1]]-bag0[edges[:,0]],axis=1)
    stretch_max=1.
    frame_metrics=[]
    for bear,bag in zip(bears,bags):
        ratios=signed_volumes(bear,tets)/rest_vol
        tetra_ratio_min=min(tetra_ratio_min,float(ratios.min()))
        inverted_volume_fraction=max(inverted_volume_fraction,float(weights[ratios<=0].sum()/weights.sum()))
        frame_penetration={"bear":0.,"bag":0.}
        for name,p in (("bear",bear),("bag",bag)):
            on_table=np.all(np.abs(p[:,:2])<=half,axis=1)
            if on_table.any():
                frame_penetration[name]=float(max(0.,table_height-p[on_table,2].min()))
                table_penetration[name]=max(table_penetration[name],frame_penetration[name])
        frame_stretch=float((np.linalg.norm(bag[edges[:,1]]-bag[edges[:,0]],axis=1)/rest_length).max())
        stretch_max=max(stretch_max,frame_stretch)
        frame_metrics.append([float(ratios.min()),frame_penetration["bag"],frame_penetration["bear"],frame_stretch])
    area=float(trimesh.Trimesh(bag0,faces,process=False).area)
    # pyuipc 0.0.29 defines the passed thickness as one-sided radius r.
    bag_mass=area*config["bag"]["rho"]*2*config["bag"]["thickness"]
    mass={"bag_kg":bag_mass,"bear_kg":float(weights.sum()*config["bear"]["rho"]),
          "bag_mass_formula":"rest area * density * 2 * IPC thickness radius",
          "derived_from_reference_geometry_and_config":True,"runtime_IPC_mass_buffer_checked":False}
    motion={}
    validation=config.get("validation",{})
    vmax=validation.get("max_joint_velocity",1.5)
    amax=validation.get("max_joint_acceleration",10.)
    joint_time=time
    joint_values={side:np.asarray(frames["q_"+side],float) for side in ("left","right")}
    motion_source="saved visualization states"
    if joint_trace is not None:
        trace=np.asarray(joint_trace,dtype=float)
        if trace.ndim!=2 or trace.shape[1]!=19 or len(trace)<3 or not np.isfinite(trace).all() or np.any(np.diff(trace[:,0])<=0):
            return {"status":"failed","task_success":False,"reason":"Invalid every-step joint_trace, expected increasing time plus nine joints per arm."}
        joint_time=trace[:,0]
        joint_values={"left":trace[:,1:10],"right":trace[:,10:19]}
        motion_source="every-physics-step actual joint_trace"
    soft_sampling=audit_soft_metrics(soft_metrics,time,frame_metrics,joint_time,
                                    float(config.get("simulation",{}).get("dt",.005)))
    if soft_sampling.get("minimum_tetrahedron_volume_ratio") is not None:
        tetra_ratio_min=min(tetra_ratio_min,soft_sampling["minimum_tetrahedron_volume_ratio"])
        stretch_max=max(stretch_max,soft_sampling["maximum_bag_edge_stretch_ratio"])
        for name in table_penetration:
            table_penetration[name]=max(table_penetration[name],soft_sampling["maximum_table_penetration_m"][name])
    for side in ("left","right"):
        q=joint_values[side]
        velocity=np.gradient(q[:,:7],joint_time,axis=0,edge_order=2)
        acceleration=np.gradient(velocity,joint_time,axis=0,edge_order=2)
        motion[side]={"maximum_joint_velocity_rad_s":float(np.abs(velocity).max()),
            "maximum_joint_acceleration_rad_s2":float(np.abs(acceleration).max()),
            "per_joint_peak_velocity_rad_s":np.abs(velocity).max(0).tolist(),
            "per_joint_peak_acceleration_rad_s2":np.abs(acceleration).max(0).tolist(),
            "within_configured_limits":bool(np.abs(velocity).max()<=vmax and np.abs(acceleration).max()<=amax)}
    body_faces,rim,topology=mouth_structure(bag0,faces,config)
    final_inside={"all_bear_vertices_inside":False}
    held=False
    hold_checks=[]
    if len(rim):
        final_inside=containment(bags[-1],bears[-1],body_faces,rim,weights,tets)
        hold_indices=np.flatnonzero(time>=time[-1]-hold_duration)
        if time[-1]-time[0]>=hold_duration and len(hold_indices)>=3:
            check_indices=np.unique(np.r_[hold_indices[0],hold_indices[len(hold_indices)//2],hold_indices[-1]])
            for i in check_indices:
                value=containment(bags[i],bears[i],body_faces,rim,weights,tets)
                hold_checks.append({"time":float(time[i]),**value})
            held=all(v["all_bear_vertices_inside"] for v in hold_checks)
    contact=final_contact_audit(bags[-1],bears[-1],faces,tets)
    events=events or []
    deformation=bear_deformation_audit(bear0,tets,bears,time,frames["q_right"],events)
    phase_names=[e.get("phase") for e in events]
    required_phases=("close_both_handles","spread_bag","release_right_handle",
        "teddy_pregrasp","teddy_grasp_approach","close_teddy_gripper","lift_teddy",
        "transport_teddy","insert_teddy","release_teddy","withdraw_right_arm","final_settle")
    if flat_start:
        required_phases=("approach_single_handle","close_single_handle","lift_single_handle","settle_single_handle",
            "teddy_pregrasp","teddy_grasp_approach","close_teddy_gripper","lift_teddy",
            "transport_teddy","insert_teddy","release_teddy","withdraw_right_arm","final_settle")
    ordered=all(p in phase_names for p in required_phases)
    if ordered:
        ordered=bool(np.all(np.diff([phase_names.index(p) for p in required_phases])>0))
        if flat_start:
            ordered=ordered and all(phase_names.count(p)==1 for p in required_phases)
    by_phase={e["phase"]:e for e in events if "phase" in e}
    left_handle=audit_left_handle_hold(bag0,frames,config,events)
    lift_clear=False
    left_displacement=None
    if "lift_teddy" in by_phase:
        e=by_phase["lift_teddy"]
        index=int(np.argmin(np.abs(time-(e["start"]+e["duration"]))))
        lift_clear=bool(bears[index,:,2].min()>table_height+.04)
    if "release_right_handle" in by_phase:
        fixed_window=time>=by_phase["release_right_handle"]["start"]
        tcp_left=np.asarray(frames["tcp_left"])[fixed_window]
        if len(tcp_left):
            left_displacement=float(np.linalg.norm(tcp_left-tcp_left[0],axis=1).max())
    flat_evidence=None
    if flat_start:
        flat_evidence=audit_flat_start(bag0,body_faces,frames,config,events,joint_time,joint_values,
                                       every_step_joint_trace=joint_trace is not None)
        left_displacement=None
        anchor=flat_evidence.get("left_stationary_anchor_time_s")
        if anchor is not None and flat_evidence["checks"]["flat_start_single_left_prefix_recorded"]:
            index=int(np.argmin(np.abs(time-anchor)))
            if abs(time[index]-anchor)<=1e-7:
                tcp_left=np.asarray(frames["tcp_left"])[time>=anchor-1e-9]
                left_displacement=float(np.linalg.norm(tcp_left-tcp_left[0],axis=1).max())
    # A finite-gap plush grasp now holds at roughly 30 mm per finger. That
    # must not be mistaken for the requested release to the 40 mm open pose.
    release_open_threshold=float(config.get("validation",{}).get("released_finger_minimum_m",.038))
    right_open=bool("release_teddy" in by_phase
        and np.min(joint_values["right"][-1,7:9])>=release_open_threshold)
    checks={"finite_saved_states":finite,"per_step_soft_metrics_valid_when_supplied":soft_sampling["valid"],
        "actual_open_bag_topology":topology["actual_open_body_boundary"],
        "two_cloth_panels_with_direct_u_seam_and_no_bottom_or_side_panels":topology["two_panels_directly_sewn_along_u"],
        "no_tetrahedron_inversion":tetra_ratio_min>0.,
        "no_tetrahedron_volume_below_5_percent":tetra_ratio_min>.05,
        "table_penetration_below_2mm":max(table_penetration.values())<=.002,
        "cloth_edge_stretch_below_15_percent":stretch_max<=1.15,
        "joint_motion_within_configured_limits":all(v["within_configured_limits"] for v in motion.values()),
        "motion_sample_interval_at_most_20ms":float(np.diff(joint_time).max())<=.020001,
        "no_final_sampled_bag_bear_penetration_over_1mm":contact["penetration_over_tolerance"]==0,
        "bear_fully_inside_bag_final":final_inside["all_bear_vertices_inside"],
        "bear_remains_inside_during_final_hold_samples":held,
        "requested_phases_recorded_in_order":ordered,
        "bear_lifted_clear_of_table":lift_clear,
        "right_fingers_open_after_release":right_open,
        "left_gripper_holds_pose_within_15mm_after_handoff":left_displacement is not None and left_displacement<=.015,
        "named_left_handle_material_within_18mm_after_handoff":left_handle["material_group_retained"],
        "both_left_fingers_command_clamping_after_handoff":left_handle["both_fingers_command_clamping"],
        "controller_explicitly_reports_task_complete":bool(task_completed)}
    if flat_start:
        for key in ("left_gripper_holds_pose_within_15mm_after_handoff",
                    "named_left_handle_material_within_18mm_after_handoff",
                    "both_left_fingers_command_clamping_after_handoff"):
            checks.pop(key)
        checks.update(flat_evidence["checks"])
        checks.update(left_gripper_holds_pose_within_15mm_after_single_lift=
                      left_displacement is not None and left_displacement<=.015,
                      named_left_handle_material_within_18mm_after_single_close=left_handle["material_group_retained"],
                      both_left_fingers_command_clamping_after_single_close=left_handle["both_fingers_command_clamping"])
    passed=all(checks.values())
    physical_keys=("finite_saved_states","actual_open_bag_topology","no_tetrahedron_inversion",
        "per_step_soft_metrics_valid_when_supplied",
        "no_tetrahedron_volume_below_5_percent",
        "table_penetration_below_2mm","cloth_edge_stretch_below_15_percent",
        "joint_motion_within_configured_limits","no_final_sampled_bag_bear_penetration_over_1mm")
    physical_failure=not all(checks[k] for k in physical_keys) or (left_handle["applicable"] and not left_handle["passed"])
    result={"status":"passed" if passed else "failed" if task_completed or physical_failure else "incomplete",
        "task_success":passed,"checks":checks,"frames":len(time),"duplicate_timestamps_removed":duplicate_count,
        "duration_s":float(time[-1]-time[0]),
        "maximum_saved_interval_s":float(np.diff(time).max()),"mass":mass,"bag_topology":topology,
        "minimum_tetrahedron_volume_ratio":tetra_ratio_min,
        "maximum_inverted_reference_volume_fraction":inverted_volume_fraction,
        "inverted_volume_fraction_sampling_scope":"Full saved frames only; per-step minimum J cannot recover inverted volume.",
        "soft_metric_sampling":soft_sampling,
        "maximum_table_penetration_m":table_penetration,"maximum_bag_edge_stretch_ratio":stretch_max,
        "motion":motion,"motion_source":motion_source,"maximum_joint_trace_interval_s":float(np.diff(joint_time).max()),
        "motion_thresholds":{"velocity_rad_s":vmax,"acceleration_rad_s2":amax,
            "source":"config.validation or conservative audit defaults; not robot manufacturer limits"},
        "final_containment":final_inside,"hold_containment_samples":hold_checks,"final_contact":contact,
        "bear_deformation":deformation,
        "left_handle_hold_evidence":left_handle,
        "sequence_evidence":{"recorded_phases":phase_names,
            "maximum_left_tcp_displacement_after_handoff_m":left_displacement,
            "released_finger_minimum_m":release_open_threshold,
            "final_right_finger_positions_m":joint_values["right"][-1,7:9].tolist()},
        "limitations":["Saved geometry and integration-step summaries do not certify continuous nonpenetration between recorded states.",
            "Two-way IPC ABD versus rendered link transform error is unavailable in these arrays.",
            "Finger opening and event timestamps support sequence checks but do not certify force closure.",
            "Named strap proximity and negative finger PD are independent hold evidence, not measured native contact forces; the IPC soft-contact guard provides complementary reaction evidence.",
            "Material values remain uncalibrated; no real-world success claim follows."]}
    if flat_start:
        result["flat_start_evidence"]=flat_evidence
        result["sequence_evidence"].pop("maximum_left_tcp_displacement_after_handoff_m")
        result["sequence_evidence"]["maximum_left_tcp_displacement_after_single_lift_m"]=left_displacement
        result["sequence_evidence"]["material_hold_anchor"]= "end of close_single_handle"
        result["sequence_evidence"]["stationary_left_anchor"]= "start of settle_single_handle, after completed lift_single_handle"
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rest",type=Path,required=True)
    parser.add_argument("--frames",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=HERE/"config.json")
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--task-completed",action="store_true")
    parser.add_argument("--joint-trace",type=Path)
    parser.add_argument("--soft-metrics",type=Path)
    parser.add_argument("--events",type=Path)
    args=parser.parse_args()
    trace_path=args.joint_trace or args.frames.parent/"joint_trace.npy"
    joint_trace=np.load(trace_path,allow_pickle=False) if trace_path.exists() else None
    metrics_path=args.soft_metrics or args.frames.parent/"soft_metrics.npy"
    soft_metrics=np.load(metrics_path,allow_pickle=False) if metrics_path.exists() else None
    events_path=args.events or args.frames.parent/"events.json"
    events=json.loads(events_path.read_text()) if events_path.exists() else []
    if (args.frames.parent/"report.json").exists():
        final_events=json.loads((args.frames.parent/"report.json").read_text()).get("events",[])
        if len(final_events)>len(events):
            events=final_events
    with np.load(args.rest,allow_pickle=False) as rest,np.load(args.frames,allow_pickle=False) as frames:
        result=audit_arrays(rest,frames,json.loads(args.config.read_text()),task_completed=args.task_completed,
                            joint_trace=joint_trace,events=events,soft_metrics=soft_metrics)
    args.output.write_text(json.dumps(result,indent=2))
    print(json.dumps({"status":result["status"],"task_success":result["task_success"],"output":str(args.output)}))
    return 0 if result["task_success"] else 2


if __name__=="__main__":
    raise SystemExit(main())
