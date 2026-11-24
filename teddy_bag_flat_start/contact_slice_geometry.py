"""CPU geometry for conservative horizontal sections and concave mouth overlap.

No simulation, deformation, contact force or physical insertion is inferred.
All lengths use the caller's common coordinate unit. Polygon results are
counterclockwise; areas are nonnegative. No Shapely dependency is required.

``horizontal_section_hulls`` evaluates every distinct tetrahedral-vertex Z,
not a sparse height sample. Between two consecutive events the active edge
set is fixed, and every edge/plane intersection is affine in Z. Its convex
hull's directional support is a maximum of affine functions, hence convex in
Z. Cauchy's perimeter formula integrates support (equivalently widths), so
the convex-section perimeter is convex on that interval: its maximum is at
an event endpoint. Event sections may contain additional horizontal edges,
which only increase that bound. This proves an event-endpoint bound for
*convex hull perimeter*, not exact occupied area, fit in a concave opening,
or a continuous-time collision/deformation guarantee.

Public polygon functions reject invalid/degenerate inputs. A section at a
legitimate top/bottom event may be a point or segment: area is zero; segment
perimeter is twice its length (the limiting convex-body perimeter).
"""
from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError

_TOL = 1e-12  # Relative tolerance after normalization to the data extent.


def _array(value, columns, name, minimum=1):
    try:
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite numeric array") from exc
    if result.ndim != 2 or result.shape[1] != columns or len(result) < minimum \
            or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a finite (n,{columns}) array, n >= {minimum}")
    return result


def _cross(a, b):
    return a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]


def _signed_area(poly):
    shifted = poly-poly[0]
    return float(_cross(shifted, np.roll(shifted, -1, axis=0)).sum()/2)


def _normalized(points):
    lo, hi = points.min(axis=0), points.max(axis=0)
    scale = float(np.max(hi-lo))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Geometry has zero or nonfinite extent")
    return (points-lo)/scale, lo, scale


def _on_segment(p, a, b):
    return abs(_cross(b-a, p-a)) <= _TOL and \
        bool(np.all(p >= np.minimum(a, b)-_TOL) and np.all(p <= np.maximum(a, b)+_TOL))


def _segments_intersect(a, b, c, d):
    x, y = _cross(b-a, c-a), _cross(b-a, d-a)
    u, v = _cross(d-c, a-c), _cross(d-c, b-c)
    if ((x > _TOL and y < -_TOL) or (x < -_TOL and y > _TOL)) and \
            ((u > _TOL and v < -_TOL) or (u < -_TOL and v > _TOL)):
        return True
    return _on_segment(c, a, b) or _on_segment(d, a, b) or \
        _on_segment(a, c, d) or _on_segment(b, c, d)


def _simple_polygon(poly):
    """Validate a simple ring, remove harmless collinear vertices, return CCW."""
    p = _array(poly, 2, "polygon", 3).copy()
    if len(p) > 3 and np.array_equal(p[0], p[-1]):
        p = p[:-1]  # A conventional repeated closing vertex is accepted.
    q, _, _ = _normalized(p)
    n = len(p)
    for i in range(n):
        if np.any(np.linalg.norm(q[i+1:]-q[i], axis=1) <= _TOL):
            raise ValueError("Polygon has duplicate or indistinguishable vertices")
        a, b, c = q[i-1], q[i], q[(i+1) % n]
        if abs(_cross(b-a, c-b)) <= _TOL and np.dot(b-a, c-b) < 0:
            raise ValueError("Polygon has overlapping adjacent edges")
        for j in range(i+1, n):
            if j == i+1 or (i == 0 and j == n-1):
                continue
            if _segments_intersect(q[i], q[(i+1) % n], q[j], q[(j+1) % n]):
                raise ValueError("Polygon self-intersects or has touching nonadjacent edges")
    indices = list(range(n))
    changed = True
    while changed and len(indices) > 3:
        changed = False
        for k, current in enumerate(indices):
            previous, following = indices[k-1], indices[(k+1) % len(indices)]
            if abs(_cross(q[current]-q[previous], q[following]-q[current])) <= _TOL:
                del indices[k]
                changed = True
                break
    p, q = p[indices], q[indices]
    area = _signed_area(q)
    if abs(area) <= _TOL:
        raise ValueError("Polygon is degenerate or numerically too thin")
    return p if area > 0 else p[::-1].copy()


def _hull_metrics(points, *, allow_degenerate=False):
    points = np.unique(_array(points, 2, "points2D"), axis=0)
    if len(points) == 1:
        if not allow_degenerate:
            raise ValueError("Convex hull needs at least three noncollinear points")
        return {"hull_xy": points.copy(), "area": 0., "perimeter": 0.}
    normalized, _, scale = _normalized(points)
    centered = normalized-normalized.mean(axis=0)
    _, singular, axes = np.linalg.svd(centered, full_matrices=False)
    collinear = len(points) < 3 or len(singular) < 2 or singular[1] <= _TOL*singular[0]
    if collinear:
        if not allow_degenerate:
            raise ValueError("Convex hull points are collinear or numerically too thin")
        projection = centered @ axes[0]
        hull = points[[int(np.argmin(projection)), int(np.argmax(projection))]]
        return {"hull_xy": hull, "area": 0., "perimeter": float(2*np.linalg.norm(hull[1]-hull[0]))}
    try:
        indices = ConvexHull(normalized).vertices
    except QhullError as exc:
        raise ValueError("Could not construct a nondegenerate convex hull") from exc
    hull, norm_hull = points[indices], normalized[indices]
    area = _signed_area(norm_hull)*scale**2
    if area <= _TOL*scale**2 and not allow_degenerate:
        raise ValueError("Convex hull is degenerate or numerically too thin")
    return {"hull_xy": hull.copy(), "area": float(area),
            "perimeter": float(np.linalg.norm(np.roll(hull, -1, axis=0)-hull, axis=1).sum())}


def convex_hull_metrics(points2D):
    """Return ``{hull_xy, area, perimeter}`` for an unordered finite point set.

Duplicate input points are harmless. Collinear/point hulls raise ValueError.
The returned hull contains unique extreme vertices in counterclockwise order.
"""
    return _hull_metrics(points2D)


def horizontal_section_hulls(points, tets):
    """Return ``[{z, hull_xy, area, perimeter}, ...]`` at all unique vertex Zs.

``points`` is (N,3), ``tets`` is integer (M,4). Every vertex must belong to a
tetrahedron; repeated/zero-volume tetrahedra and invalid indices are rejected.
Both tetrahedron windings are accepted. This does not certify a conforming,
nonoverlapping volume mesh; it computes the convex hull of its section union.
The hull area can include holes/gaps in the occupied section. See the module
docstring for the event-endpoint perimeter bound and its limits.
"""
    p = _array(points, 3, "points", 4)
    cells = np.asarray(tets)
    if cells.ndim != 2 or cells.shape[1] != 4 or len(cells) == 0 or \
            cells.dtype.kind not in "iu" or np.any(cells < 0) or np.any(cells >= len(p)):
        raise ValueError("tets must be a nonempty integer (m,4) array of valid vertex indices")
    cells = cells.astype(np.int64, copy=False)
    canonical = np.sort(cells, axis=1)
    if np.any(np.diff(canonical, axis=1) == 0) or len(np.unique(canonical, axis=0)) != len(cells):
        raise ValueError("Tetrahedra must have four distinct vertices and no duplicate cells")
    if len(np.unique(cells)) != len(p):
        raise ValueError("Every point must be referenced by a tetrahedron")
    vertices = p[cells]
    relative = vertices[:, 1:]-vertices[:, :1]
    local_scale = np.linalg.norm(relative, axis=2).max(axis=1)
    if np.any(local_scale <= 0) or np.any(~np.isfinite(local_scale)):
        raise ValueError("Degenerate tetrahedron extent")
    determinants = np.linalg.det(relative/local_scale[:, None, None])
    if np.any(np.abs(determinants) <= _TOL):
        raise ValueError("Zero-volume or numerically degenerate tetrahedron")
    pairs = np.array([[0,1], [0,2], [0,3], [1,2], [1,3], [2,3]])
    edges = np.unique(np.sort(cells[:, pairs].reshape(-1, 2), axis=1), axis=0)
    a, b = p[edges[:, 0]], p[edges[:, 1]]
    dz = b[:, 2]-a[:, 2]
    horizontal = dz == 0.
    low, high = np.minimum(a[:, 2], b[:, 2]), np.maximum(a[:, 2], b[:, 2])
    result = []
    for z in np.unique(p[:, 2]):
        active = (~horizontal) & (low <= z) & (z <= high)
        first, second, delta = a[active], b[active], dz[active]
        fraction = (z-first[:, 2])/delta
        hits = first[:, :2] + fraction[:, None]*(second[:, :2]-first[:, :2])
        # Preserve exact vertex coordinates at endpoints to deduplicate hits.
        at_first, at_second = z == first[:, 2], z == second[:, 2]
        hits[at_first] = first[at_first, :2]
        hits[at_second] = second[at_second, :2]
        flat = horizontal & (a[:, 2] == z)
        hits = np.concatenate((hits, a[flat, :2], b[flat, :2]), axis=0)
        result.append({"z": float(z), **_hull_metrics(hits, allow_degenerate=True)})
    return result


def simple_polygon_triangles(poly):
    """Ear-clip a simple 2D ring into an array (T,3,2) of CCW triangles.

Clockwise input, a repeated final closing vertex, and intermediate collinear
vertices are accepted. Holes, duplicate vertices, backtracking and touching
or crossing nonadjacent edges are rejected. Area is preserved to tolerance.
"""
    p = _simple_polygon(poly)
    q, _, _ = _normalized(p)
    remaining = list(range(len(p)))
    triangles = []
    while len(remaining) > 3:
        for k, current in enumerate(remaining):
            before, after = remaining[k-1], remaining[(k+1) % len(remaining)]
            a, b, c = q[[before, current, after]]
            if _cross(b-a, c-b) <= _TOL:
                continue
            other = [i for i in remaining if i not in (before, current, after)]
            test = q[other]
            inside = (_cross(b-a, test-a) >= -_TOL) & \
                     (_cross(c-b, test-b) >= -_TOL) & (_cross(a-c, test-c) >= -_TOL)
            if np.any(inside):
                continue
            triangles.append(p[[before, current, after]])
            del remaining[k]
            break
        else:
            raise ValueError("Ear clipping failed: invalid or numerically degenerate polygon")
    triangles.append(p[remaining])
    triangles = np.asarray(triangles)
    expected = _signed_area(q)
    actual = sum(_signed_area((tri-p.min(axis=0))/np.ptp(p, axis=0).max()) for tri in triangles)
    if not np.isclose(actual, expected, rtol=1e-10, atol=_TOL):
        raise ValueError("Triangulation did not preserve polygon area")
    return triangles


def _clip_convex(subject, clip):
    """Sutherland-Hodgman clipping, both polygons in one normalized frame."""
    out = np.asarray(subject, dtype=float)
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        if len(out) == 0:
            break
        edge = b-a
        signed = _cross(edge, out-a)
        signed[np.abs(signed) <= _TOL] = 0.
        previous, previous_distance = out[-1], signed[-1]
        new = []
        for current, distance in zip(out, signed):
            if (distance >= 0) != (previous_distance >= 0):
                fraction = previous_distance/(previous_distance-distance)
                new.append(previous + fraction*(current-previous))
            if distance >= 0:
                new.append(current)
            previous, previous_distance = current, distance
        out = np.asarray(new, dtype=float).reshape(-1, 2)
    return out


@lru_cache(maxsize=16)
def _validated_triangles(shape, contents):
    """Validate a triangulated disk once per exact array value, then cache.

Oppositely directed interior edges cancel. A single simple CCW boundary and
positive triangles guarantee unit winding inside that boundary, preventing
overlap/double-counting without pairwise triangle intersection tests.
"""
    triangles = np.frombuffer(contents, dtype=np.float64).reshape(shape).copy()
    flat = triangles.reshape(-1, 2)
    normalized, _, scale = _normalized(flat)
    normalized = normalized.reshape(shape)
    areas = _cross(normalized[:, 1]-normalized[:, 0], normalized[:, 2]-normalized[:, 0])/2
    if np.any(np.abs(areas) <= _TOL/2):
        raise ValueError("Pretriangles contain a degenerate triangle")
    triangles[areas < 0] = triangles[areas < 0][:, ::-1]
    unique, labels = np.unique(triangles.reshape(-1, 2), axis=0, return_inverse=True)
    labels = labels.reshape(-1, 3)
    edge_map = {}
    for row in labels:
        for a, b in zip(row, np.roll(row, -1)):
            edge_map.setdefault(tuple(sorted((int(a), int(b)))), []).append((int(a), int(b)))
    boundary = []
    for directed in edge_map.values():
        if len(directed) == 1:
            boundary.append(directed[0])
        elif len(directed) != 2 or directed[0] != directed[1][::-1]:
            raise ValueError("Pretriangles overlap or have inconsistent shared edges")
    if not boundary or len(unique)-len(edge_map)+len(triangles) != 1:
        raise ValueError("Pretriangles must form one triangulated disk without holes")
    outgoing = {}
    for a, b in boundary:
        if a in outgoing:
            raise ValueError("Pretriangle boundary branches")
        outgoing[a] = b
    if len(set(outgoing.values())) != len(outgoing):
        raise ValueError("Pretriangle boundary branches")
    start, ring = boundary[0][0], []
    current = start
    while current not in ring:
        ring.append(current)
        if current not in outgoing:
            raise ValueError("Pretriangle boundary is open")
        current = outgoing[current]
    if current != start or len(ring) != len(boundary):
        raise ValueError("Pretriangles have multiple boundary loops")
    boundary_poly = _simple_polygon(unique[ring])
    area = sum(_signed_area((tri-flat.min(axis=0))/scale) for tri in triangles)
    if not np.isclose(area, _signed_area((boundary_poly-flat.min(axis=0))/scale),
                      rtol=1e-10, atol=_TOL):
        raise ValueError("Pretriangle area does not equal its simple boundary area")
    triangles.setflags(write=False)
    return triangles


def convex_polygon_overlap_area(convex_hull, simple_polygon=None, *, pretriangles=None):
    """Area of an ordered convex polygon intersected with a simple polygon.

Pass either a 2D ring as the second argument, or an array (T,3,2) there or as
``pretriangles=...``. Pretriangles must form one nonoverlapping triangulated
disk with identical shared vertex coordinates; invalid topology is rejected.
Use ``simple_polygon_triangles`` once and reuse its array for many sections.
Validation is cached by exact array content, so mutation cannot bypass it.
The sum of convex clips is exact up to floating-point geometric tolerance;
there is no rasterization, sampling approximation, or convexification of the
concave input polygon. The convex_hull itself must be a nondegenerate ring.
"""
    hull = _simple_polygon(convex_hull)
    normalized, _, _ = _normalized(hull)
    turns = _cross(np.roll(normalized, -1, axis=0)-normalized,
                   np.roll(normalized, -2, axis=0)-np.roll(normalized, -1, axis=0))
    if np.any(turns <= _TOL):
        raise ValueError("convex_hull must be convex and cyclically ordered")
    if (simple_polygon is None) == (pretriangles is None):
        raise ValueError("Provide exactly one simple_polygon or pretriangles")
    if simple_polygon is not None:
        candidate = np.asarray(simple_polygon, dtype=float)
        raw = simple_polygon_triangles(candidate) if candidate.ndim == 2 else candidate
    else:
        raw = np.asarray(pretriangles, dtype=float)
    if raw.ndim != 3 or raw.shape[1:] != (3, 2) or len(raw) == 0 or not np.isfinite(raw).all():
        raise ValueError("pretriangles must be a finite nonempty (t,3,2) array")
    raw = np.ascontiguousarray(raw, dtype=np.float64)
    triangles = _validated_triangles(raw.shape, raw.tobytes())
    together = np.concatenate((hull, triangles.reshape(-1, 2)), axis=0)
    _, origin, scale = _normalized(together)
    subject = (hull-origin)/scale
    area = 0.
    lo, hi = subject.min(axis=0), subject.max(axis=0)
    for tri in (triangles-origin)/scale:
        if np.any(tri.max(axis=0) < lo-_TOL) or np.any(tri.min(axis=0) > hi+_TOL):
            continue
        clipped = _clip_convex(subject, tri)
        if len(clipped) >= 3:
            area += abs(_signed_area(clipped))
    result = area*scale**2
    maximum = min(_signed_area(hull), sum(_signed_area(t) for t in triangles))
    if result < -_TOL*scale**2 or result > maximum+max(_TOL*scale**2, 1e-10*maximum):
        raise ValueError("Overlap area exceeded an input polygon area")
    return float(min(max(result, 0.), maximum))


def _selftest(output):
    checks = []

    def close(name, actual, expected):
        passed = bool(np.allclose(actual, expected, rtol=1e-10, atol=1e-11))
        checks.append({"name": name, "passed": passed,
                       "actual": np.asarray(actual).tolist(), "expected": np.asarray(expected).tolist()})

    def rejects(name, function):
        try:
            function()
        except ValueError as exc:
            checks.append({"name": name, "passed": True, "rejected": str(exc)})
        else:
            checks.append({"name": name, "passed": False, "rejected": False})

    rectangle = np.array([[0.,0.], [2.,0.], [2.,1.], [0.,1.]])
    metrics = convex_hull_metrics(np.r_[rectangle, [[1.,.5], [0.,0.]]])
    close("rectangle_hull_area_perimeter", [metrics["area"], metrics["perimeter"]], [2., 6.])
    shifted = rectangle+[1., .5]
    close("rectangle_overlap", convex_polygon_overlap_area(rectangle, shifted), .5)
    close("touching_rectangle_zero_area", convex_polygon_overlap_area(rectangle, rectangle+[2.,0.]), 0.)
    close("disjoint_rectangle_zero_area", convex_polygon_overlap_area(rectangle, rectangle+[3.,0.]), 0.)
    ell = np.array([[0.,0.], [2.,0.], [2.,1.], [1.,1.], [1.,2.], [0.,2.]])
    triangles = simple_polygon_triangles(ell)
    close("concave_L_triangulation_area", sum(_signed_area(t) for t in triangles), 3.)
    square = np.array([[.5,.5], [1.5,.5], [1.5,1.5], [.5,1.5]])
    close("concave_L_overlap_exact", convex_polygon_overlap_area(square, ell), .75)
    close("pretriangles_overlap", convex_polygon_overlap_area(square, pretriangles=triangles), .75)
    close("positional_pretriangles_overlap", convex_polygon_overlap_area(square, triangles), .75)
    close("clockwise_L_overlap", convex_polygon_overlap_area(square[::-1], ell[::-1]), .75)
    closed = np.r_[rectangle[:1], [[1.,0.]], rectangle[1:], rectangle[:1]]
    close("closed_collinear_ring_area", sum(_signed_area(t) for t in simple_polygon_triangles(closed)), 2.)
    tetra = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.]])
    sections = horizontal_section_hulls(tetra, np.array([[0,1,2,3]]))
    close("tetra_all_event_heights", [s["z"] for s in sections], [0.,1.])
    close("tetra_base_section", [sections[0]["area"],sections[0]["perimeter"]], [.5, 2+np.sqrt(2)])
    close("tetra_top_point", [sections[-1]["area"],sections[-1]["perimeter"]], [0.,0.])
    split = np.r_[tetra, [[0.,0.,.5]]]
    split_sections = horizontal_section_hulls(split, np.array([[0,1,2,4],[4,1,2,3]]))
    close("tetra_mid_event_area_perimeter", [split_sections[1]["area"],split_sections[1]["perimeter"]],
          [.125, (2+np.sqrt(2))/2])
    close("split_tetra_all_events", [s["z"] for s in split_sections], [0.,.5,1.])
    edge_tetra = np.array([[0.,0.,0.], [1.,0.,0.], [0.,0.,1.], [0.,1.,1.]])
    edge_sections = horizontal_section_hulls(edge_tetra, np.array([[0,1,2,3]]))
    close("tetra_extreme_segment_limit_perimeters", [s["perimeter"] for s in edge_sections], [2.,2.])
    close("translated_scaled_overlap", convex_polygon_overlap_area(square*1e-3+[123.,-55.], ell*1e-3+[123.,-55.]), .75e-6)
    rejects("reject_bowtie", lambda: simple_polygon_triangles([[0,0],[1,1],[0,1],[1,0]]))
    rejects("reject_duplicate_ring_vertex", lambda: simple_polygon_triangles([[0,0],[1,0],[1,1],[1,0],[0,1]]))
    rejects("reject_collinear_hull", lambda: convex_hull_metrics([[0,0],[1,0],[2,0]]))
    rejects("reject_concave_convex_argument", lambda: convex_polygon_overlap_area(ell, rectangle))
    rejects("reject_duplicate_pretriangles", lambda: convex_polygon_overlap_area(square, pretriangles=np.repeat(triangles[:1],2,axis=0)))
    rejects("reject_flat_tetra", lambda: horizontal_section_hulls(np.c_[rectangle, np.zeros(4)], np.array([[0,1,2,3]])))
    rejects("reject_float_tetra_indices", lambda: horizontal_section_hulls(tetra, [[0.,1.,2.,3.]]))
    rejects("reject_unreferenced_point", lambda: horizontal_section_hulls(split, [[0,1,2,3]]))
    rejects("reject_nonfinite_polygon", lambda: simple_polygon_triangles([[0,0],[1,0],[0,np.nan]]))
    report = {"kind": "CPU geometry unit checks only", "physical_success": False,
              "passed": all(c["passed"] for c in checks), "checks": checks,
              "section_perimeter_scope": "All exact vertex-Z events; convex hull perimeter bound, not concave fit or physical success."}
    Path(output).write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]:
        raise AssertionError(f"Geometry checks failed; see {output}")
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output", default=str(Path(__file__).with_name("contact_slice_geometry_selftest.json")))
    args = parser.parse_args()
    if not args.self_test:
        parser.error("Use --self-test to run the isolated CPU checks")
    result = _selftest(args.output)
    print(json.dumps({"passed": result["passed"], "check_count": len(result["checks"]), "report": args.output}))
