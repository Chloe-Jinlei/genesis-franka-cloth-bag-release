"""Geometry-only contact targeting; never constrains or moves a soft vertex.

All positions are in one common frame and all lengths are metres. Panda's hand
local Y is the closing axis, local Z points from hand toward the fingertips,
and the TCP is the centre of the pads (hand-local [0, 0, .1029]). Rotations map
hand-local vectors into the input/world frame. SciPy quaternions are xyzw.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial.transform import Rotation


def _unit(v, label):
    v = np.asarray(v, dtype=float)
    length = np.linalg.norm(v)
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError(f"Cannot determine {label}")
    return v / length


def _clip(poly, axis, bound, sign):
    """Clip a 3D polygon to sign*(coordinate-bound)<=0."""
    out = []
    if len(poly) == 0:
        return np.empty((0, 3))
    previous = poly[-1]
    dp = sign * (previous[axis] - bound)
    for current in poly:
        dc = sign * (current[axis] - bound)
        if (dp <= 0) != (dc <= 0):
            out.append(previous + dp / (dp - dc) * (current - previous))
        if dc <= 0:
            out.append(current)
        previous, dp = current, dc
    return np.asarray(out).reshape(-1, 3)


@dataclass(frozen=True)
class ClothGrasp:
    position: np.ndarray
    rotation: Rotation
    minimum_half_gap: float
    diagnostics: dict

    def as_dict(self):
        quat = self.rotation.as_quat()
        return {
            "position_m": self.position.tolist(),
            "quaternion_xyzw": quat.tolist(),
            "quaternion_wxyz": quat[[3, 0, 1, 2]].tolist(),
            "rotation_columns_hand_axes_in_world": self.rotation.as_matrix().tolist(),
            "minimum_initial_finger_q_m": self.minimum_half_gap,
            **self.diagnostics,
        }


def cloth_grasp_frame(points, faces, grasp_indices, *,
                      preferred_approach=(0., 0., -1.),
                      preferred_closing=(0., -1., 0.),
                      shell_radius=.0003, clearance=.0002,
                      pad_half_width=.0085, pad_half_height=.0085):
    """Return a surface-aligned, nonintersecting initial Panda grasp target.

    The named grasp nodes can be a single transverse row, so their PCA is not
    used. An area-weighted mean of incident triangle normals determines the
    sheet normal. Its sign follows preferred_closing. The projected approach
    determines the remaining rotation about that normal.

    minimum_half_gap includes sheet curvature inside the actual rectangular
    pad footprint, the one-sided IPC shell radius, and requested clearance.
    This is a local *initial* geometric check, not a friction-grasp guarantee
    or a whole-hand collision check. The caller must preflight whole robots.
    """
    points = np.asarray(points, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    indices = np.unique(np.asarray(grasp_indices, dtype=np.int64))
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("points must be a finite (n,3) array")
    if len(indices) < 1 or np.any(indices < 0) or np.any(indices >= len(points)):
        raise ValueError("grasp_indices must name existing vertices")
    if min(shell_radius, clearance) < 0 or min(pad_half_width, pad_half_height) <= 0:
        raise ValueError("Invalid physical radius, clearance, or pad dimensions")
    incident = np.flatnonzero(np.isin(faces, indices).any(axis=1))
    if len(incident) < 2:
        raise ValueError("The grasp needs a two-dimensional incident surface patch")
    tri = points[faces[incident]]
    area_vectors = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    twice_areas = np.linalg.norm(area_vectors, axis=1)
    if np.any(twice_areas < 1e-14):
        raise ValueError("Degenerate incident cloth triangle")
    normal = _unit(area_vectors.sum(axis=0), "coherent cloth surface normal")
    if normal @ _unit(preferred_closing, "preferred closing direction") < 0:
        normal = -normal
    z = np.asarray(preferred_approach, dtype=float)
    z = _unit(z - np.dot(z, normal) * normal, "tangent approach direction")
    x = _unit(np.cross(normal, z), "hand transverse axis")
    matrix = np.column_stack((x, normal, z))
    position = points[indices].mean(axis=0)
    # Clip actual incident triangles instead of testing sparse vertices. A pad
    # may intersect a triangle even if none of its vertices lie in the pad.
    local_tri = (tri - position) @ matrix
    footprint = []
    for poly in local_tri:
        for axis, bound, sign in ((0, pad_half_width, 1), (0, -pad_half_width, -1),
                                  (2, pad_half_height, 1), (2, -pad_half_height, -1)):
            poly = _clip(poly, axis, bound, sign)
        if len(poly):
            footprint.append(poly)
    if not footprint:
        raise ValueError("No cloth surface intersects the pad footprint")
    footprint = np.concatenate(footprint)
    ymin, ymax = footprint[:, 1].min(), footprint[:, 1].max()
    minimum_half_gap = float(max(abs(ymin), abs(ymax)) + shell_radius + clearance)
    normal_cos = np.abs((area_vectors / twice_areas[:, None]) @ normal)
    return ClothGrasp(position, Rotation.from_matrix(matrix), minimum_half_gap, {
        "incident_triangle_count": int(len(incident)),
        "normal_in_world": normal.tolist(),
        "cloth_normal_range_in_pad_m": [float(ymin), float(ymax)],
        "shell_radius_m": float(shell_radius),
        "clearance_m": float(clearance),
        "max_incident_normal_deviation_deg": float(np.degrees(np.arccos(np.clip(normal_cos, -1, 1))).max()),
        "whole_hand_collision_checked": False,
        "friction_grasp_proven": False,
    })


def track_reference_displacement(rest_points, current_points, reference, *, radius=.032):
    """Track a chosen point using its neighborhood's mean displacement.

    This preserves the requested reference point when motion is zero; returning
    the current neighborhood centroid instead would introduce a geometry bias.
    It follows local translation approximately, not rotation or strain exactly.
    """
    rest = np.asarray(rest_points, dtype=float)
    current = np.asarray(current_points, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if current.shape != rest.shape or not np.isfinite(current).all():
        raise ValueError("Current and rest point arrays must be finite and match")
    indices = np.flatnonzero(np.linalg.norm(rest - reference, axis=1) < radius)
    if len(indices) < 3:
        raise ValueError("Fewer than three material nodes near the reference")
    return reference + (current[indices] - rest[indices]).mean(axis=0)


@dataclass(frozen=True)
class TetraMaterialPoint:
    vertex_indices: np.ndarray
    weights: np.ndarray
    tetra_index: int

    def evaluate(self, points):
        """Observe the same material point under the FEM linear deformation."""
        points = np.asarray(points, dtype=float)
        return self.weights @ points[self.vertex_indices]


def tetra_material_point(rest_points, tetrahedra, reference, *, tolerance=1e-9):
    """Locate a reference inside a rest tetrahedron; fail if outside the solid.

    The returned barycentric observer tracks rigid rotation and linear strain
    exactly inside a tetrahedron. It does not alter forces or pin the target.
    """
    points = np.asarray(rest_points, dtype=float)
    tetrahedra = np.asarray(tetrahedra, dtype=np.int64)
    reference = np.asarray(reference, dtype=float)
    tri = points[tetrahedra]
    matrix = np.moveaxis(tri[:, 1:] - tri[:, :1], 1, 2)
    valid = np.flatnonzero(np.abs(np.linalg.det(matrix)) > 1e-16)
    if not len(valid):
        raise ValueError("No nondegenerate rest tetrahedra")
    three = np.linalg.solve(matrix[valid], (reference - tri[valid, 0])[..., None])[..., 0]
    bary = np.column_stack((1 - three.sum(axis=1), three))
    inside = np.flatnonzero((bary.min(axis=1) >= -tolerance) & (bary.max(axis=1) <= 1+tolerance))
    if not len(inside):
        raise ValueError("Reference grasp point is outside the tetrahedral teddy")
    # Prefer the deepest numerical containment if the point lies on a face.
    local = inside[np.argmax(bary[inside].min(axis=1))]
    k = int(valid[local])
    return TetraMaterialPoint(tetrahedra[k].copy(), bary[local].copy(), k)
