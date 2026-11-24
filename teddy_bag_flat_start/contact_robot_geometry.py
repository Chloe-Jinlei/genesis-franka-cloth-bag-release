"""CPU certificates for a frozen tetrahedral body swept past the measured left arm.

No simulator is created and no state array is modified.  A tetrahedron translated
along a segment sweeps exactly the convex hull of its eight endpoint vertices.
The hull of *all* endpoint vertices encloses their union, providing a fast first
certificate.  Optional per-tetrahedron refinement removes that first hull's
concavity overestimate.  Robot collision meshes are treated as convex solids,
as in MuJoCo's collision model; this can overestimate native nonconvex geometry.

The acceptance certificate is a separating-axis gap, recomputed using the
original double-precision world vertices and reduced by a numerical guard.
MuJoCo's closest segment only proposes an axis: its distance never authorizes
acceptance.  Thus all faces, edges, interiors and every intermediate translation
are covered.  This is not a vertex-only proximity check or a dynamics prediction.
"""
from __future__ import annotations

import itertools
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, QhullError

from motion import Planner


def _array(value, shape, label):
    out = np.asarray(value, dtype=float)
    if out.shape != shape or not np.isfinite(out).all():
        raise ValueError(f"{label} must have finite shape {shape}")
    return out


def _convex(points):
    points = np.unique(np.asarray(points, float), axis=0)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 \
            or not np.isfinite(points).all():
        raise ValueError("Convex solid needs at least four finite 3D points")
    hull = ConvexHull(points)
    if not np.isfinite(hull.volume) or hull.volume <= 1e-18:
        raise ValueError("Convex solid has zero or numerically degenerate volume")
    mapping = np.full(len(points), -1, int)
    mapping[hull.vertices] = np.arange(len(hull.vertices))
    vertices = points[hull.vertices].copy()
    faces = mapping[hull.simplices].copy()
    tri = vertices[faces]
    flip = np.einsum("ij,ij->i", np.cross(tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]),
                     hull.equations[:, :3]) < 0
    faces[flip] = faces[flip][:, [0, 2, 1]]
    return dict(vertices=vertices, faces=faces, equations=hull.equations.copy(),
                volume=float(hull.volume), bounds=np.array([vertices.min(0), vertices.max(0)]))


def _surface_topology(vertices, faces):
    """Count surface edges after welding duplicate positions, for diagnostics only."""
    _, labels = np.unique(np.round(vertices, 10), axis=0, return_inverse=True)
    faces = labels[faces]
    degenerate = np.any(np.diff(np.sort(faces, axis=1), axis=1) == 0, axis=1)
    faces = faces[~degenerate]
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return dict(surface_watertight_after_position_weld=bool(len(counts) and np.all(counts == 2)),
                surface_degenerate_faces=int(degenerate.sum()), surface_boundary_edges=int(np.sum(counts == 1)),
                surface_nonmanifold_edges=int(np.sum(counts > 2)))


def left_collision_solids(planner, q_left):
    """Extract every nonbase left collision geom; preserve each measured finger.

    The caller's Planner qpos is restored before returning.  Unsupported geom
    types fail closed rather than silently disappearing from the certificate.
    """
    q_left = _array(q_left, (9,), "q_left")
    m, d = planner.m, planner.d
    old_qpos = d.qpos.copy()
    solids = []
    try:
        d.qpos[planner.qindices["left"]] = q_left[:7]
        for index in (1, 2):
            d.qpos[m.joint(f"left_finger_joint{index}").qposadr[0]] = q_left[6+index]
        mujoco.mj_forward(m, d)
        for geom in range(m.ngeom):
            body = planner.body_name(geom)
            if not body.startswith("left_") or body.endswith("link0") \
                    or not (m.geom_contype[geom] or m.geom_conaffinity[geom]):
                continue
            kind = int(m.geom_type[geom])
            if kind == int(mujoco.mjtGeom.mjGEOM_MESH):
                mesh = int(m.geom_dataid[geom])
                va, vn = int(m.mesh_vertadr[mesh]), int(m.mesh_vertnum[mesh])
                fa, fn = int(m.mesh_faceadr[mesh]), int(m.mesh_facenum[mesh])
                vertices = m.mesh_vert[va:va+vn].astype(float).copy()
                faces = m.mesh_face[fa:fa+fn].astype(int).copy()
                if not len(faces) or faces.min() < 0 or faces.max() >= len(vertices):
                    raise ValueError(f"Invalid collision mesh for {body} geom {geom}")
                topology = _surface_topology(vertices, faces)
                original_type = "mesh"
            elif kind == int(mujoco.mjtGeom.mjGEOM_BOX):
                size = m.geom_size[geom].copy()
                if not np.isfinite(size).all() or np.any(size <= 0):
                    raise ValueError(f"Invalid collision box for {body} geom {geom}")
                vertices = np.array(list(itertools.product((-1., 1.), repeat=3))) * size
                topology = dict(surface_watertight_after_position_weld=True,
                                surface_degenerate_faces=0, surface_boundary_edges=0,
                                surface_nonmanifold_edges=0)
                original_type = "box"
            else:
                raise ValueError(f"Unsupported left collision geom type {kind} on {body}, geom {geom}")
            vertices = vertices @ d.geom_xmat[geom].reshape(3, 3).T + d.geom_xpos[geom]
            solid = _convex(vertices)
            solid.update(geom_id=geom, geom_name=m.geom(geom).name, body=body,
                         original_geom_type=original_type, **topology)
            solids.append(solid)
    finally:
        d.qpos[:] = old_qpos
        mujoco.mj_forward(m, d)
    if not solids:
        raise ValueError("No nonbase left collision solids were extracted")
    return solids


def _projection_gap(a, b, axes):
    axes = np.asarray(axes, float).reshape(-1, 3)
    lengths = np.linalg.norm(axes, axis=1)
    valid = np.isfinite(axes).all(axis=1) & np.isfinite(lengths) & (lengths > 1e-12)
    axes = axes[valid] / lengths[valid, None]
    if not len(axes):
        return 0., None
    av, bv = a @ axes.T, b @ axes.T
    gaps = np.maximum(bv.min(0)-av.max(0), av.min(0)-bv.max(0))
    best = int(np.argmax(gaps))
    return max(0., float(gaps[best])), axes[best]


def _mujoco_axis(a, b):
    root = ET.Element("mujoco")
    ET.SubElement(root, "compiler", angle="radian")
    asset = ET.SubElement(root, "asset")
    world = ET.SubElement(root, "worldbody")
    for name, shape in (("a", a), ("b", b)):
        ET.SubElement(asset, "mesh", name=name,
            vertex=" ".join(format(float(x), ".17g") for x in shape["vertices"].ravel()),
            face=" ".join(str(int(x)) for x in shape["faces"].ravel()))
        ET.SubElement(world, "geom", name=name, type="mesh", mesh=name)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    if hasattr(model.opt, "ccd_tolerance"):
        model.opt.ccd_tolerance = 1e-10
    if hasattr(model.opt, "ccd_iterations"):
        model.opt.ccd_iterations = 100
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    segment = np.zeros(6)
    distance = float(mujoco.mj_geomDistance(model, data, 0, 1, 10., segment))
    if not np.isfinite(distance) or not np.isfinite(segment).all():
        raise ValueError("Nonfinite temporary convex-distance result")
    return distance, segment[3:]-segment[:3]


def _common_ball(a, b):
    """Positive optimum proves intersecting convex *interiors*, including containment."""
    equations = np.concatenate([a["equations"], b["equations"]])
    matrix = np.column_stack([equations[:, :3], np.ones(len(equations))])
    result = linprog([0., 0., 0., -1.], A_ub=matrix, b_ub=-equations[:, 3],
                     bounds=[(None, None)]*4, method="highs")
    if not result.success or not np.isfinite(result.x).all():
        return None
    # Recompute feasible slack; solver feasibility tolerances cannot invent an
    # interior witness. This is an inscribed common radius, not penetration depth.
    slack = -(equations[:, :3] @ result.x[:3] + equations[:, 3])
    return dict(center_world_m=result.x[:3].tolist(), radius_m=float(slack.min()))


def convex_clearance_certificate(a, b, *, required_gap_m, numerical_guard_m=1e-7):
    """Sufficient solid-separation certificate; uncertainty always rejects."""
    axes = np.vstack([np.eye(3), a["equations"][:, :3], b["equations"][:, :3],
                      b["vertices"].mean(0)-a["vertices"].mean(0)])
    gap, axis = _projection_gap(a["vertices"], b["vertices"], axes)
    signed_distance = None
    if gap-numerical_guard_m < required_gap_m:
        signed_distance, candidate = _mujoco_axis(a, b)
        candidate_gap, candidate_axis = _projection_gap(a["vertices"], b["vertices"], [candidate])
        if candidate_gap > gap:
            gap, axis = candidate_gap, candidate_axis
    lower = max(0., gap-numerical_guard_m)
    result = dict(status="clear" if lower >= required_gap_m else "indeterminate",
        conservative_gap_lower_bound_m=lower, raw_separating_projection_gap_m=gap,
        separating_axis_world=None if axis is None else axis.tolist(), numerical_guard_m=numerical_guard_m,
        mujoco_convex_signed_distance_diagnostic_m=signed_distance,
        solid_envelope_intersection_proven=False,
        original_surface_distance_computed=False)
    if result["status"] != "clear":
        witness = _common_ball(a, b)
        result["common_interior_ball"] = witness
        if witness is not None and witness["radius_m"] > numerical_guard_m:
            result.update(status="blocked", solid_envelope_intersection_proven=True,
                          reason="convex solid interiors intersect")
        else:
            result["reason"] = "required continuous solid gap was not certified"
    return result


def check_translation_sweep(points, tets, q_left, config, *, displacement_m,
                         translation_m=(0., 0., 0.), required_gap_m=.010,
                         numerical_guard_m=1e-7, refine_tetrahedra=True, planner=None):
    """Check all left collision solids against a continuous frozen-shape sweep.

    ``translation_m`` maps this actual observation to the segment start.
    ``displacement_m`` is the three-dimensional straight segment vector.
    ``q_left`` is measured
    joint1..7, finger1, finger2; no finger averaging or clipping occurs.

    The configurable gap is an engineering planning allowance. Default 10 mm
    is conservative; an explicitly reviewed caller may pass 0.005. Neither
    changes contact laws. A clear result requires every collision geom to pass.
    A rejected convex envelope is not evidence of a native force or of actual
    nonconvex surface contact. Reobserve after physical preparation.
    """
    report = dict(kind="frozen_tetra_translation_sweep_vs_left_robot", status="indeterminate",
        physical_success=False, runtime_execution_performed=False, rejections=[], per_geom=[],
        original_surface_distance_computed=False,
        model_scope="continuous frozen tetrahedral body versus MuJoCo convex collision solids",
        limitations=["Does not predict deformation, friction, grasp retention or contact forces.",
            "Robot convex envelopes can overestimate nonconvex native collision surfaces.",
            "Left pose is fixed at measured q; native affine compliance and future tracking are not included.",
            "Only the declared straight translation segment is certified; other segments and obstacles require separate gates.",
            "The certificate is for frozen-shape Cartesian translation, not continuous FK tracking of an interpolated joint trajectory.",
            "A fresh actual-shape check remains necessary after physical preparation."])
    try:
        points = np.asarray(points, float)
        tets = np.asarray(tets)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 4 or not np.isfinite(points).all():
            raise ValueError("points must be a finite nonempty (n,3) array")
        if tets.ndim != 2 or tets.shape[1] != 4 or not len(tets) or tets.dtype.kind not in "iu" \
                or tets.min() < 0 or tets.max() >= len(points) \
                or np.any(np.diff(np.sort(tets, axis=1), axis=1) == 0):
            raise ValueError("tets must contain valid distinct in-range integer vertex indices")
        tetra = points[tets]
        determinants = np.linalg.det(tetra[:, 1:]-tetra[:, :1])
        if not np.isfinite(determinants).all() or np.any(np.abs(determinants) <= 1e-18):
            raise ValueError("Degenerate or nonfinite actual tetrahedron")
        q_left = _array(q_left, (9,), "q_left")
        translation = _array(translation_m, (3,), "translation_m")
        displacement = _array(displacement_m, (3,), "displacement_m")
        required, guard = map(float, (required_gap_m, numerical_guard_m))
        if not np.isfinite([required, guard]).all() or required <= 0. or guard <= 0. or guard >= required:
            raise ValueError("Displacement and positive gap/guard must be finite, with guard < gap")
        report.update(q_left_measured=q_left.tolist(), individual_left_fingers_m=q_left[7:].tolist(),
            translation_m=translation.tolist(), displacement_m=displacement.tolist(),
            required_gap_m=required, numerical_guard_m=guard, refine_tetrahedra=bool(refine_tetrahedra),
            input_vertex_count=len(points), tetrahedron_count=len(tets),
            tetra_vertex_union_used=True, includes_all_intermediate_translations=True)
        planner = Planner(config) if planner is None else planner
        robot = left_collision_solids(planner, q_left)
        used = np.unique(tets)
        outside = points[used]+translation
        sweep = _convex(np.concatenate([outside, outside+displacement]))
        report["whole_sweep_hull"] = dict(vertex_count=len(sweep["vertices"]),
            volume_m3=sweep["volume"], bounds_world_m=sweep["bounds"].tolist(), watertight_convex_solid=True)
        report["left_collision_geom_count"] = len(robot)
        report["left_collision_bodies"] = sorted({r["body"] for r in robot})
        tetra_start = tetra+translation
        endpoints = np.concatenate([tetra_start, tetra_start+displacement], axis=1)
        bounds_lo, bounds_hi = endpoints.min(1), endpoints.max(1)
        prism_cache = {}
        for shape in robot:
            row = {k: shape[k] for k in ("geom_id", "geom_name", "body", "original_geom_type",
                "surface_watertight_after_position_weld", "surface_degenerate_faces",
                "surface_boundary_edges", "surface_nonmanifold_edges")}
            whole = convex_clearance_certificate(sweep, shape, required_gap_m=required, numerical_guard_m=guard)
            row["whole_hull_certificate"] = whole
            result = whole
            if whole["status"] != "clear" and refine_tetrahedra:
                boxgap = np.linalg.norm(np.maximum(np.maximum(shape["bounds"][0]-bounds_hi,
                                        bounds_lo-shape["bounds"][1]), 0.), axis=1)
                clear_by_box = boxgap-guard >= required
                lower = float(np.min(boxgap[clear_by_box]-guard)) if np.any(clear_by_box) else math.inf
                detailed = []
                status = "clear"
                for index in np.flatnonzero(~clear_by_box)[np.argsort(boxgap[~clear_by_box])]:
                    index = int(index)
                    if index not in prism_cache:
                        prism_cache[index] = _convex(endpoints[index])
                    check = convex_clearance_certificate(prism_cache[index], shape,
                        required_gap_m=required, numerical_guard_m=guard)
                    detailed.append(dict(tetrahedron_index=index, **check))
                    lower = min(lower, check["conservative_gap_lower_bound_m"])
                    if check["status"] != "clear":
                        status = check["status"]
                        break
                result = dict(status=status, conservative_gap_lower_bound_m=lower if np.isfinite(lower) else 0.,
                    exact_tetra_prism_union=True, aabb_certified_tetrahedra=int(clear_by_box.sum()),
                    refined_tetrahedra_checked=len(detailed), tetrahedra_total=len(tets),
                    first_uncertified_tetrahedron=detailed[-1] if status != "clear" else None,
                    minimum_refined_certificate=min(detailed, key=lambda v:v["conservative_gap_lower_bound_m"]) if detailed else None,
                    early_reject=status != "clear")
            row["sweep_certificate"] = result
            report["per_geom"].append(row)
            if result["status"] != "clear":
                report.update(status=result["status"], conservative_gap_lower_bound_m=0.,
                              early_reject=True, complete_robot_certificate=False)
                report["rejections"].append(f"{shape['body']} geom {shape['geom_id']}: continuous {required:g} m gap not certified")
                return report
        report.update(status="clear", conservative_gap_lower_bound_m=min(
            row["sweep_certificate"]["conservative_gap_lower_bound_m"] for row in report["per_geom"]),
            early_reject=False, complete_robot_certificate=True)
    except (ValueError, TypeError, IndexError, KeyError, OSError, QhullError) as exc:
        report["rejections"].append(str(exc))
        report.update(status="indeterminate", conservative_gap_lower_bound_m=0., complete_robot_certificate=False)
    return report


def check_vertical_sweep(points, tets, q_left, config, *, vertical_displacement_m,
                         outside_translation_m=(0., 0., 0.), required_gap_m=.010,
                         numerical_guard_m=1e-7, refine_tetrahedra=True, planner=None):
    """Compatibility API for a vertical segment, using the general solid check."""
    try:
        dz=float(vertical_displacement_m)
    except (TypeError, ValueError):
        dz=float("nan")
    result=check_translation_sweep(points,tets,q_left,config,
        translation_m=outside_translation_m,displacement_m=(0.,0.,dz),
        required_gap_m=required_gap_m,numerical_guard_m=numerical_guard_m,
        refine_tetrahedra=refine_tetrahedra,planner=planner)
    result["kind"]="frozen_tetra_vertical_sweep_vs_left_robot"
    if "translation_m" in result:
        result["outside_translation_m"]=result["translation_m"]
        result["vertical_displacement_m"]=float(result["displacement_m"][2])
    return result
