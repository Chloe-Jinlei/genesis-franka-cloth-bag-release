"""CPU-only 0.46 x 0.40 m two-panel U-seam candidate, with original handles.

Writes only new wide_tall coarse assets/metadata, the roomy 4 N candidate
configuration, and its geometric audit. No existing asset or preview is used
as an output, and no Genesis simulation is initialized.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import numpy as np

import two_panel_builder as builder
from contact_grasp import cloth_grasp_frame
from motion import Planner, SIDES

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "config_upper_arm_roomy_4n.json"
REPORT = HERE / "wide_tall_bag_cpu_validation.json"
STEM = "two_panel_bag_wide_tall_coarse"
METADATA = "metadata_two_panel_wide_tall_coarse.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protected = [p for p in builder.ASSETS.iterdir()
                 if p.is_file() and "wide_tall" not in p.name]
    protected += [HERE / name for name in (
        "config.json", "config_two_panel_centered.json", "config_head_grasp_tall.json",
        "two_panel_builder.py", "build_tall_bag.py", "run.py", "scene.py", "motion.py")]
    before = {str(p): sha256(p) for p in protected}
    base = json.loads((HERE / "config_two_panel_centered.json").read_text())
    builder.W, builder.H = .46, .40
    point, geometry = builder.opened_panel_geometry()
    bag = builder.build_bag(.02, STEM, point, geometry)
    bear = json.loads((builder.ASSETS / base["bag"]["metadata"]).read_text())["bear"]
    (builder.ASSETS / METADATA).write_text(json.dumps({
        "units": "metres", "generator": "build_wide_tall_bag.py",
        "construction_source": "two_panel_builder.py",
        "candidate_status": "CPU geometry only; no dynamic validation",
        "bag": bag, "bear": bear}, indent=2) + "\n")
    config = copy.deepcopy(base)
    config["bag"].update(mesh=bag["file"], npz_file=bag["npz_file"],
                         metadata=METADATA, height_m=.40)
    config["task"].update(bear_finger_force_limit=4., bear_finger_target=0.)
    for target in config["task"]["open_targets"].values():
        target[2] = 1.26
    CONFIG.write_text(json.dumps(config, indent=2) + "\n")

    planner = Planner(config)
    source = np.load(builder.ASSETS / config["bag"]["npz_file"])
    front, back = source["panel_front_grid"], source["panel_back_grid"]
    panel_faces = np.r_[source["panel_front_face_indices"], source["panel_back_face_indices"]]
    topology_checks = {
        "left_bottom_right_seams_share_vertices": all(np.array_equal(front[sl], back[sl])
            for sl in ((0, slice(None)), (slice(None), 0), (slice(None), -1))),
        "body_faces_are_exactly_the_two_panels": np.array_equal(
            np.sort(panel_faces), np.sort(source["body_face_indices"])),
        "zero_bottom_seam_transverse_depth": np.ptp(source["points"][source["bottom_seam"], 1]) == 0.,
        "one_connected_manifold_cloth": bag["mesh"]["connected_components"] == 1
            and bag["mesh"]["nonmanifold_edges"] == 0,
        "three_full_boundary_loops_one_body_opening": bag["mesh"]["boundary_loops"] == 3
            and bag["body_mesh"]["boundary_loops"] == 1,
        "no_degenerate_faces": bag["mesh"]["degenerate_triangles"] == 0}
    topology_checks = {key: bool(value) for key, value in topology_checks.items()}
    qs, fingers, rotations, initial = {}, {}, {}, {}
    for side in SIDES:
        spec = config["task"]["handle_grasps"][side]
        frame = cloth_grasp_frame(source["points"], source["faces"], source[spec["group"]],
            shell_radius=config["bag"]["thickness"], preferred_approach=spec["preferred_approach"])
        target = np.asarray(config["bag"]["position"]) + frame.position
        q, position_error, angle_error = planner.solve(side, target, frame.rotation)
        qs[side], fingers[side], rotations[side] = q, max(.001, frame.minimum_half_gap), frame.rotation
        initial[side] = {"target_tcp_m": target.tolist(), "q_arm_rad": q.tolist(),
            "finger_q_m": fingers[side], "position_error_m": position_error,
            "rotation_error_rad": angle_error, "grasp_frame": frame.as_dict(),
            "joint_limit_margin_rad": float(np.minimum(
                q-planner.limits[side][:, 0], planner.limits[side][:, 1]-q).min()),
            "reachable": position_error <= .0005 and angle_error <= .002}
    initial_clearance = planner.clearance(qs, fingers)
    report = {"status": "candidate_geometry_only", "dynamic_success": False,
        "config": str(CONFIG), "source_config": str(HERE / "config_two_panel_centered.json"),
        "dimensions_m": {"width": .46, "height": .40, "bottom_depth": 0.},
        "coarse": bag, "topology_checks": topology_checks, "initial": initial,
        "initial_clearance": initial_clearance,
        "initial_min_rigid_clearance_m": min(v["distance"] for v in initial_clearance.values()),
        "initial_hand_tcp_separation_m": float(np.linalg.norm(
            np.asarray(initial["left"]["target_tcp_m"])-initial["right"]["target_tcp_m"])),
        "material_parameters_unchanged": all(config["bag"][k] == base["bag"][k] for k in (
            "E", "nu", "rho", "thickness", "bending_stiffness", "friction_mu", "shear_stiffness")),
        "limitations": [
            "No Genesis/GPU simulation; no frictional hold, natural hanging pose or successful insertion is claimed.",
            "Rigid clearance uses CPU MuJoCo convex geometry, not measured native contact force.",
            "Soft bag/bear collisions and usable single-handle mouth/depth are untested.",
            "Planner self-clearance excludes opposing fingers and adjacent/welded links; zero finger target means force-limited closure, not proven separation.",
            "Opening checks ideal joint targets from undeformed initial handles; actual measured joints depend on cloth dynamics.",
            "Only coarse 20 mm body sampling is generated. The material parameters and handle construction rules are unchanged."]}
    closed = dict.fromkeys(SIDES, 0.)
    try:
        speed = float(config["task"].get("motion_speed_scale", 1.))
        plan = planner.trajectory(qs, config["task"]["open_targets"], rotations=rotations,
            samples=45, duration=2.5/speed, finger_start=closed, finger_end=closed,
            max_joint_velocity=.55*speed, max_joint_acceleration=speed**2,
            clearance_margin=.008, validate_samples=101)
        best = {}
        for time in np.linspace(0, plan.duration, 101):
            sample = plan.sample(float(time))
            for key, value in planner.clearance(sample["q"], sample["finger"]).items():
                if key not in best or value["distance"] < best[key]["distance"]:
                    best[key] = dict(value, time_s=float(time))
        end = plan.sample(plan.duration)
        report["open_route"] = {"passed_cpu_geometry": True, "duration_s": plan.duration,
            "ik_samples": 45, "clearance_samples": 101, "commanded_finger_q_m": closed,
            "clearance_minima": best, "minimum_rigid_clearance_m": min(v["distance"] for v in best.values()),
            "end_q_arm_rad": {s: end["q"][s].tolist() for s in SIDES},
            "endpoint_tcp_error_m": {s: float(np.linalg.norm(planner.fk(s, end["q"][s])[0]
                -config["task"]["open_targets"][s])) for s in SIDES}}
    except ValueError as exc:
        report["open_route"] = {"passed_cpu_geometry": False, "error": str(exc)}
    after = {str(p): sha256(p) for p in protected}
    report["protected_existing_files_unchanged"] = before == after
    report["protected_file_count"] = len(before)
    report["protected_hashes_before"] = before
    report["cpu_checks_passed"] = all(topology_checks.values()) and \
        all(v["reachable"] for v in initial.values()) and \
        report["initial_min_rigid_clearance_m"] >= .008 and \
        report["open_route"]["passed_cpu_geometry"] and before == after
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "cpu_checks_passed", "topology_checks",
        "initial_min_rigid_clearance_m", "initial_hand_tcp_separation_m", "open_route",
        "protected_existing_files_unchanged")}, indent=2))
    print(f"REPORT {REPORT}")


if __name__ == "__main__":
    main()
