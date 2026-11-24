"""Original deformable teddy and open cotton tote assets, in metres.

The bag is a single sewn triangle shell with no lid: its two ribbons share
vertices with the rim. The teddy is one connected watertight soft volume; eyes,
mouth stitches and fur are texture only, never rigid collision inserts.
No scene, installed package or previously created asset is modified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import fast_simplification
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes
import tetgen
import trimesh

HERE = Path(__file__).resolve().parent
ASSETS = HERE / "assets"


def write_obj(path, points, faces, uv=None, texture=None):
    """Keep physical vertex indices intact; UV seam gets separate vt indices."""
    uv_faces = np.asarray(faces).copy()
    if uv is not None:
        uv = np.asarray(uv)
        extra = {}
        for fi in np.flatnonzero(np.ptp(uv[faces, 0], axis=1) > .5):
            for ci in np.flatnonzero(uv[faces[fi], 0] < .5):
                vi = int(faces[fi, ci])
                if vi not in extra:
                    extra[vi] = len(uv) + len(extra)
                uv_faces[fi, ci] = extra[vi]
        texture_uv = np.concatenate([uv, uv[list(extra)] + [1, 0]]) if extra else uv
    with Path(path).open("w") as f:
        f.write("# Original deformable teddy-bag asset. SI units.\n")
        if texture:
            f.write(f"mtllib {Path(path).stem}.mtl\nusemtl fabric\n")
        for p in points:
            f.write("v " + " ".join(f"{v:.10g}" for v in p) + "\n")
        if uv is not None:
            for p in texture_uv:
                f.write("vt " + " ".join(f"{v:.10g}" for v in p) + "\n")
        for fi, face in enumerate(faces):
            entries = [f"{v+1}/{uv_faces[fi,ci]+1}" if uv is not None else str(v+1)
                       for ci, v in enumerate(face)]
            f.write("f " + " ".join(entries) + "\n")
    if texture:
        Path(path).with_suffix(".mtl").write_text(
            f"newmtl fabric\nKd 1 1 1\nKa .2 .2 .2\nKs .01 .01 .01\nNs 3\nmap_Kd {texture}\n")


def topology(points, faces):
    mesh = trimesh.Trimesh(points, faces, process=False)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    edge_lengths = np.linalg.norm(points[unique[:, 0]] - points[unique[:, 1]], axis=1)
    # Count connected boundary loops independently from physical components.
    adjacency = {}
    for a, b in boundary:
        adjacency.setdefault(int(a), []).append(int(b))
        adjacency.setdefault(int(b), []).append(int(a))
    seen, loops = set(), 0
    for start in adjacency:
        if start in seen:
            continue
        loops += 1
        stack = [start]
        while stack:
            v = stack.pop()
            if v not in seen:
                seen.add(v)
                stack.extend(adjacency[v])
    return {
        "vertices": len(points), "triangles": len(faces),
        "connected_components": len(mesh.split(only_watertight=False)),
        "watertight": bool(mesh.is_watertight),
        "consistent_winding": bool(mesh.is_winding_consistent),
        "nonmanifold_edges": int(np.sum(counts > 2)),
        "boundary_edges": len(boundary), "boundary_loops": loops,
        "boundary_degree_two": all(len(v) == 2 for v in adjacency.values()),
        "degenerate_triangles": int(np.sum(mesh.area_faces < 1e-12)),
        "triangle_area_m2": float(mesh.area),
        "bounds_m": [np.min(points, axis=0).tolist(), np.max(points, axis=0).tolist()],
        "edge_length_m": {"min": float(edge_lengths.min()),
                          "median": float(np.median(edge_lengths)), "max": float(edge_lengths.max())},
    }


def make_bag(output_dir):
    width, depth, height, strap_width = .38, .22, .32, .025
    across = np.unique(np.round(np.r_[np.linspace(-strap_width / 2, strap_width / 2, 6),
                                      [-.01, 0., .01]], 10))
    # These nonuniform rim samples make the sewn ends share exactly the same
    # indices as the shell, without duplicate vertices or nonmanifold seams.
    xs = np.unique(np.round(np.r_[np.linspace(-width / 2, width / 2, 39),
                                  -.12 + across, .12 + across], 10))
    ys = np.linspace(-depth / 2, depth / 2, 23)
    zs = np.linspace(0, height, 33)
    points, faces, index = [], [], {}
    def vertex(p):
        key = tuple(np.round(p, 9))
        if key not in index:
            index[key] = len(points)
            points.append(p)
        return index[key]
    def quad(a, b, c, d, flip=False):
        # a,b at previous row; c,d at next row.
        tris = [[a, b, d], [a, d, c]]
        faces.extend([t[::-1] for t in tris] if flip else tris)
    perimeter = ([(x, -depth / 2) for x in xs[:-1]]
                 + [(width / 2, y) for y in ys[:-1]]
                 + [(x, depth / 2) for x in xs[::-1][:-1]]
                 + [(-width / 2, y) for y in ys[::-1][:-1]])
    rings = np.array([[vertex([x, y, z]) for x, y in perimeter] for z in zs])
    for k in range(len(zs) - 1):
        for j in range(len(perimeter)):
            j2 = (j + 1) % len(perimeter)
            quad(rings[k, j], rings[k, j2], rings[k + 1, j], rings[k + 1, j2])
    bottom = np.array([[vertex([x, y, 0.]) for x in xs] for y in ys])
    for j in range(len(ys) - 1):
        for i in range(len(xs) - 1):
            quad(bottom[j, i], bottom[j, i + 1], bottom[j + 1, i], bottom[j + 1, i + 1], flip=True)
    body_indices = list(range(len(points)))
    groups = {"rim": rings[-1].tolist(), "bottom": np.unique(bottom).tolist(),
              "body": body_indices}
    arc = np.linspace(np.pi, 0, 39)
    path = np.r_[np.column_stack([np.full(4, -.12), np.linspace(.32, .36, 5)[:-1]]),
                 np.column_stack([.12 * np.cos(arc), .36 + .12 * np.sin(arc)]),
                 np.column_stack([np.full(4, .12), np.linspace(.36, .32, 5)[1:]])]
    tangent = np.gradient(path, axis=0)
    tangent[0], tangent[-1] = [0, 1], [0, -1]
    tangent /= np.linalg.norm(tangent, axis=1)[:, None]
    normals = np.column_stack([tangent[:, 1], -tangent[:, 0]])
    for name, y in (("front", -depth / 2), ("back", depth / 2)):
        strip = np.array([[vertex([p[0] + q * n[0], y, p[1] + q * n[1]])
                           for q in across] for p, n in zip(path, normals)])
        for j in range(len(path) - 1):
            for i in range(len(across) - 1):
                quad(strip[j, i], strip[j, i + 1], strip[j + 1, i], strip[j + 1, i + 1], flip=name == "back")
        groups[f"handle_{name}"] = np.unique(strip).tolist()
        groups[f"handle_{name}_root_left"] = strip[0].tolist()
        groups[f"handle_{name}_root_right"] = strip[-1].tolist()
        # ~30 mm longitudinal patch at top of strap; actual contact must be
        # resolved by the gripper and cloth solver, these are selection labels.
        ids = strip[np.abs(path[:, 0]) <= .02]
        groups[f"grasp_{name}"] = np.unique(ids).tolist()
    points, faces = np.asarray(points, float), np.asarray(faces, np.int32)
    mesh = trimesh.Trimesh(points, faces, process=False)
    trimesh.repair.fix_normals(mesh)
    faces = mesh.faces.copy()
    stats = topology(points, faces)
    assert stats["connected_components"] == 1 and stats["nonmanifold_edges"] == 0
    assert stats["consistent_winding"] and stats["boundary_degree_two"]
    assert stats["boundary_loops"] == 3 and not stats["watertight"]
    assert stats["degenerate_triangles"] == 0
    uv = np.column_stack([(points[:, 0] + width / 2) / width,
                          (points[:, 2] + points[:, 1] * .15) / .5])
    initial_points = points.copy()
    initial_points[:, 1] *= .5
    write_obj(output_dir / "bag.obj", points, faces, uv, "cotton_canvas.png")
    write_obj(output_dir / "bag_collision.obj", points, faces)
    write_obj(output_dir / "bag_initial_half_depth.obj", initial_points, faces, uv, "cotton_canvas.png")
    np.savez_compressed(output_dir / "bag.npz", points=points, faces=faces, uv=uv,
                        initial_points=initial_points,
                        **{k: np.asarray(v, np.int32) for k, v in groups.items()})
    centers = {k: points[v].mean(axis=0).tolist() for k, v in groups.items() if k.startswith("grasp")}
    return {"file": "bag.obj", "collision_file": "bag_collision.obj",
            "half_depth_preview_file": "bag_initial_half_depth.obj",
            "npz_file": "bag.npz", "width_m": width, "depth_m": depth, "height_m": height,
            "strap_width_m": strap_width, "strap_peak_z_m": float(points[:, 2].max()),
            "mesh": stats, "vertex_groups": groups, "grasp_centers_local_m": centers,
            "initial_half_depth_factor": .5,
            "initial_state_note": "Use bag.obj as rest mesh and bag.npz initial_points as initial state; importing the compressed preview alone changes the rest shape.",
            "seams": "Bag bottom and walls and both handle ends share geometric vertices. No lid, hidden supports, fixed nodes or disconnected visual handles.",
            "boundary_note": "Three boundary loops: the open mouth plus one hole under each sewn handle.",
            "suggested_material": {"name": "light cotton canvas", "thickness_m": .00065,
                                   "surface_density_kg_m2": .24, "poisson_ratio": .3,
                                   "young_modulus_Pa": 1.5e7, "contact_friction": .55,
                                   "note": "Starting estimates only; effective woven fabric stiffness and damping require experimental calibration."},
            "estimated_mass_kg": stats["triangle_area_m2"] * .24}


def bear_field(p):
    shapes = [((0, -.025, .045), (.065, .075, .040)),
              ((0, .080, .051), (.058, .055, .047)),
              ((-.047, .126, .057), (.023, .023, .022)),
              ((.047, .126, .057), (.023, .023, .022)),
              ((-.076, -.010, .040), (.040, .035, .028)),
              ((.076, -.010, .040), (.040, .035, .028)),
              ((-.040, -.102, .034), (.036, .041, .030)),
              ((.040, -.102, .034), (.036, .041, .030)),
              ((0, .068, .091), (.031, .024, .017)),
              ((0, .083, .106), (.012, .008, .006))]
    fields = [(np.linalg.norm((p - c) / r, axis=-1) - 1) * min(r) for c, r in shapes]
    result = fields[0]
    for other in fields[1:]:
        k = .007
        h = np.maximum(k - np.abs(result - other), 0) / k
        result = np.minimum(result, other) - .25 * k * h * h
    return result


def tetra_boundary(points, tetra):
    faces = np.concatenate([tetra[:, [1, 2, 3]], tetra[:, [0, 3, 2]],
                            tetra[:, [0, 1, 3]], tetra[:, [0, 2, 1]]])
    opposite = np.concatenate([tetra[:, 0], tetra[:, 1], tetra[:, 2], tetra[:, 3]])
    _, inv, counts = np.unique(np.sort(faces, axis=1), axis=0, return_inverse=True, return_counts=True)
    assert counts.max() == 2
    faces, opposite = faces[counts[inv] == 1], opposite[counts[inv] == 1]
    tri = points[faces]
    inward = np.einsum("ij,ij->i", np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]),
                       points[opposite] - tri[:, 0]) > 0
    faces[inward] = faces[inward][:, [0, 2, 1]]
    return faces


def bear_uv(points):
    return np.column_stack([.5 + np.arctan2(points[:, 0], points[:, 2] - .043) / (2 * np.pi),
                            (points[:, 1] + .16) / .32])


def make_bear(output_dir):
    spacing = .0025
    # Offset the sampling grid from exact ellipsoid extrema; otherwise zero
    # field samples at symmetric poles can produce zero-area marching faces.
    low, high = np.array([-.13537, -.16029, -.01031]), np.array([.135, .17, .125])
    axes = [np.arange(a, b + spacing / 2, spacing) for a, b in zip(low, high)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)
    vertices, faces, _, _ = marching_cubes(bear_field(grid), 0, spacing=(spacing,) * 3)
    vertices += low
    bounds = np.ptp(vertices, axis=0)
    vertices -= (vertices.min(axis=0) + vertices.max(axis=0)) / 2
    vertices *= np.array([.24, .32, .10]) / bounds
    vertices[:, 2] -= vertices[:, 2].min()
    mesh = trimesh.Trimesh(vertices, faces, process=True)
    trimesh.repair.fix_normals(mesh)
    assert mesh.is_watertight and len(mesh.split()) == 1
    # Redistribute tiny marching-cubes pole edges before simplification. This
    # removes micrometre-scale tetrahedra while preserving the plush silhouette.
    trimesh.smoothing.filter_taubin(mesh, lamb=.5, nu=.53, iterations=8)
    vv, vf = fast_simplification.simplify(mesh.vertices, mesh.faces, target_count=6500, agg=5.)
    visual = trimesh.Trimesh(vv, vf, process=True)
    trimesh.repair.fix_normals(visual)
    vv, vf = visual.vertices.copy(), visual.faces.copy()
    pv, pf = fast_simplification.simplify(mesh.vertices, mesh.faces, target_count=1700, agg=5.)
    pv[:, 2] -= pv[:, 2].min()
    vv[:, 2] -= vv[:, 2].min()
    physical = trimesh.Trimesh(pv, pf, process=True)
    trimesh.repair.fix_normals(physical)
    assert physical.is_watertight and len(physical.split()) == 1
    tet = tetgen.TetGen(physical.vertices, physical.faces.astype(np.int32))
    points, tetra, _, _ = tet.tetrahedralize(quality=True, minratio=1.5, mindihedral=5,
                                           nobisect=True, maxvolume=1.2e-6, fixedvolume=True,
                                           steinerleft=1500, quiet=True)
    volumes = np.linalg.det(points[tetra[:, 1:]] - points[tetra[:, :1]]) / 6
    tetra[volumes < 0] = tetra[volumes < 0][:, [0, 2, 1, 3]]
    assert np.min(np.abs(volumes)) > 1e-14
    boundary = tetra_boundary(points, tetra)
    used, inv = np.unique(boundary, return_inverse=True)
    surface_points, surface_faces = points[used], inv.reshape(-1, 3)
    stats = topology(surface_points, surface_faces)
    assert stats["watertight"] and stats["connected_components"] == 1
    assert stats["nonmanifold_edges"] == 0 and stats["consistent_winding"]
    arm_left = np.flatnonzero((points[:, 0] < -.080) & (np.abs(points[:, 1] + .01) < .025))
    arm_right = np.flatnonzero((points[:, 0] > .080) & (np.abs(points[:, 1] + .01) < .025))
    write_obj(output_dir / "bear.obj", vv, vf, bear_uv(vv), "teddy_fur.png")
    write_obj(output_dir / "bear_physics.obj", surface_points, surface_faces,
              bear_uv(surface_points), "teddy_fur.png")
    write_obj(output_dir / "bear_collision.obj", surface_points, surface_faces)
    # Native tetrahedral data avoids repeated remeshing in engines that accept it.
    np.savez_compressed(output_dir / "bear.npz", points=points, tetra=tetra,
                        surface_points=surface_points, surface_faces=surface_faces,
                        surface_point_indices=used, boundary_faces_global=boundary,
                        visual_points=vv, visual_faces=vf, uv=bear_uv(vv),
                        grasp_arm_left=arm_left, grasp_arm_right=arm_right,
                        fixed_indices=np.empty(0, np.int32))
    # Gmsh v2 ASCII volume mesh; useful for FEM import without TetGen at runtime.
    with (output_dir / "bear.msh").open("w") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n$Nodes\n" + str(len(points)) + "\n")
        for i, p in enumerate(points):
            f.write(f"{i+1} " + " ".join(f"{v:.12g}" for v in p) + "\n")
        f.write("$EndNodes\n$Elements\n" + str(len(tetra)) + "\n")
        for i, t in enumerate(tetra):
            f.write(f"{i+1} 4 2 0 0 " + " ".join(str(v+1) for v in t) + "\n")
        f.write("$EndElements\n")
    volume = float(np.abs(volumes).sum())
    return {"file": "bear.obj", "physical_surface_file": "bear_physics.obj",
            "collision_file": "bear_collision.obj",
            "tetrahedral_file": "bear.msh", "npz_file": "bear.npz",
            "orientation": "Head +Y, face +Z, lying on back, lowest point Z=0.",
            "nominal_dimensions_xyz_m": [.24, .32, .10], "surface": stats,
            "tetrahedral_vertices": len(points), "tetrahedra": len(tetra),
            "visual_triangles": len(vf), "volume_m3": volume,
            "minimum_tetra_volume_m3": float(np.min(np.abs(volumes))),
            "grasp_arm_left_indices": arm_left.tolist(), "grasp_arm_right_indices": arm_right.tolist(),
            "grasp_arm_left_local_m": points[arm_left].mean(axis=0).tolist(),
            "grasp_arm_right_local_m": points[arm_right].mean(axis=0).tolist(),
            "upper_arm_thickness_m": .055,
            "visual_only_features": "Fur, embroidered eyes, smile, belly and paw patches are UV texture on the deforming mesh. Ears, four limbs, muzzle and nose are part of the connected physical volume.",
            "fixed_particles": 0, "rigid_skeleton": False,
            "suggested_material": {"name": "effective stuffed plush continuum", "density_kg_m3": .18 / volume,
                                   "young_modulus_Pa": 15000, "poisson_ratio": .28,
                                   "contact_friction": .65,
                                   "note": "Homogeneous calibrated approximation to stuffing plus textile skin, not individually resolved fibres."},
            "target_mass_kg": .18}


def make_textures(output_dir):
    n = 1024
    rng = np.random.default_rng(1126)
    yy, xx = np.mgrid[:n, :n]
    noise = rng.normal(0, 1, (n, n))
    weave = 1.2 * (np.cos(xx * np.pi) + np.cos(yy * np.pi)) + noise * 1.1
    canvas = np.clip(np.array([230, 220, 194]) + weave[..., None], 0, 255).astype(np.uint8)
    Image.fromarray(canvas).save(output_dir / "cotton_canvas.png")
    grain = gaussian_filter(noise, .7) * 5 + gaussian_filter(noise, 8) * 16
    fur = np.clip(np.array([163, 113, 73]) + grain[..., None], 0, 255).astype(np.uint8)
    im = Image.fromarray(fur)
    draw = ImageDraw.Draw(im)
    def ellipse(u, y, du, dy, color):
        v = (y + .16) / .32
        draw.ellipse(((u-du)*n, (1-v-dy/.32)*n, (u+du)*n, (1-v+dy/.32)*n), fill=color)
    def line(points, color, width=3):
        draw.line([(u*n, (1-(y+.16)/.32)*n) for u, y in points], fill=color, width=width)
    cream, dark = (223, 190, 143), (46, 30, 23)
    ellipse(.5, -.035, .19, .057, cream)
    ellipse(.5, .069, .085, .020, (235, 206, 160))
    for u in (.43, .57):
        ellipse(u, .111, .015, .008, dark)
        ellipse(u-.004, .113, .004, .0024, (250, 239, 217))
    ellipse(.5, .084, .024, .006, dark)
    line([(.5, .080), (.5, .067)], dark, 4)
    for sign in (-1, 1):
        line([(.5+sign*.037*t, .067-.004*np.sin(t*np.pi)) for t in np.linspace(0, 1, 20)], dark, 3)
    for u in (.22, .78):
        ellipse(u, .141, .036, .013, (205, 156, 110))
    for u in (.31, .69):
        ellipse(u, -.122, .047, .020, (218, 181, 132))
    im.save(output_dir / "teddy_fur.png")


def build_assets(output_dir=ASSETS):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    make_textures(output_dir)
    metadata = {"units": "metres", "generator": "assets_builder.py", "seed": 1126,
                "bag": make_bag(output_dir), "bear": make_bear(output_dir)}
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ASSETS)
    args = parser.parse_args()
    result = build_assets(args.output_dir)
    print(json.dumps({"bag": result["bag"]["mesh"],
                      "bear": result["bear"]["surface"],
                      "bear_tetrahedra": result["bear"]["tetrahedra"]}, indent=2))
