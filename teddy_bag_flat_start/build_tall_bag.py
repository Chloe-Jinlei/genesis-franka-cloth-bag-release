"""Build an independent 0.42 x 0.40 m two-panel bag candidate, using CPU only.

Reuses the existing U-seam/handle construction. It writes only tall-named
assets, metadata, candidate configuration and a geometric check report. It
does not render or replace the current bag preview and does not run Genesis.
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
CONFIG = HERE / "config_head_grasp_tall.json"
REPORT = HERE / "tall_bag_cpu_validation.json"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    protected = [p for p in builder.ASSETS.iterdir()
                 if p.is_file() and "tall" not in p.name]
    protected += [HERE / name for name in (
        "config.json", "config_head_grasp_following_3n.json", "two_panel_builder.py",
        "run.py", "scene.py", "motion.py")]
    before = {str(p): sha256(p) for p in protected}
    base = json.loads((HERE / "config_head_grasp_following_3n.json").read_text())
    builder.W, builder.H = .42, .40
    point, geometry = builder.opened_panel_geometry()
    fine = builder.build_bag(.01, "two_panel_bag_tall", point, geometry)
    coarse = builder.build_bag(.02, "two_panel_bag_tall_coarse", point, geometry)
    scale = fine["mesh"]["triangle_area_m2"] / coarse["mesh"]["triangle_area_m2"]
    coarse["estimated_mass_kg"] = fine["estimated_mass_kg"]
    coarse["suggested_material"]["surface_density_kg_m2"] *= scale
    coarse["production_rho_multiplier_for_equal_mass"] = scale
    bear = json.loads((builder.ASSETS / base["bag"]["metadata"]).read_text())["bear"]
    common = {"units": "metres", "generator": "build_tall_bag.py",
              "construction_source": "two_panel_builder.py", "bear": bear,
              "candidate_status": "CPU geometry only; no dynamic validation"}
    for detail, other_key, other_file in (
        (fine, "coarse_metadata_file", "metadata_two_panel_tall_coarse.json"),
        (coarse, "fine_metadata_file", "metadata_two_panel_tall.json")):
        name = "metadata_two_panel_tall" + ("_coarse" if detail is coarse else "") + ".json"
        (builder.ASSETS / name).write_text(json.dumps(
            {**common, "bag": detail, other_key: other_file}, indent=2) + "\n")
    config = copy.deepcopy(base)
    config["bag"].update(mesh=coarse["file"], npz_file=coarse["npz_file"],
                         metadata="metadata_two_panel_tall_coarse.json", height_m=.40)
    for target in config["task"]["open_targets"].values():
        target[2] += .06
    CONFIG.write_text(json.dumps(config, indent=2) + "\n")

    planner = Planner(config)
    source = np.load(builder.ASSETS / config["bag"]["npz_file"])
    qs, fingers, rotations, initial = {}, {}, {}, {}
    for side in SIDES:
        spec = config["task"]["handle_grasps"][side]
        frame = cloth_grasp_frame(source["points"], source["faces"], source[spec["group"]],
            shell_radius=config["bag"]["thickness"],
            preferred_approach=spec["preferred_approach"])
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
        "config": str(CONFIG), "dimensions_m": {"width": .42, "height": .40, "bottom_depth": 0.},
        "fine": fine, "coarse": coarse, "initial": initial,
        "initial_clearance": initial_clearance,
        "initial_min_rigid_clearance_m": min(v["distance"] for v in initial_clearance.values()),
        "initial_hand_tcp_separation_m": float(np.linalg.norm(
            np.asarray(initial["left"]["target_tcp_m"])-initial["right"]["target_tcp_m"])),
        "material_parameters_unchanged": all(config["bag"][k] == base["bag"][k] for k in (
            "E", "nu", "rho", "thickness", "bending_stiffness", "friction_mu", "shear_stiffness")),
        "limitations": [
            "No Genesis or GPU simulation was run; grasp holding and usable bag depth remain unproven.",
            "Rigid clearance uses the CPU MuJoCo model; soft bag/bear collision and native contact forces are not checked.",
            "Standard planner self-clearance excludes opposing fingers and adjacent/welded links; cloth pad fit is local only.",
            "The open motion is a straight-path geometric candidate before cloth settling, not a demonstrated opening.",
            "Nominal cloth height is 0.40 m; the initially opened geometry and later unsupported drape have smaller usable height."]}
    try:
        plan = planner.trajectory(qs, config["task"]["open_targets"], rotations=rotations,
            samples=35, duration=3., finger_start=fingers, finger_end=fingers,
            clearance_margin=.005, validate_samples=61)
        best = {k: dict(v, time_s=0.) for k, v in initial_clearance.items()}
        for time in np.linspace(0, plan.duration, 61):
            sample = plan.sample(float(time))
            for key, value in planner.clearance(sample["q"], sample["finger"]).items():
                if value["distance"] < best[key]["distance"]:
                    best[key] = dict(value, time_s=float(time))
        report["open_route"] = {"passed_cpu_geometry": True, "duration_s": plan.duration,
            "ik_samples": 35, "clearance_samples": 61, "clearance_minima": best,
            "minimum_rigid_clearance_m": min(v["distance"] for v in best.values()),
            "end_q_arm_rad": {s: plan.sample(plan.duration)["q"][s].tolist() for s in SIDES}}
    except ValueError as exc:
        report["open_route"] = {"passed_cpu_geometry": False, "error": str(exc)}
    after = {str(p): sha256(p) for p in protected}
    report["protected_existing_files_unchanged"] = before == after
    report["protected_file_count"] = len(before)
    report["protected_hashes_before"] = before
    report["cpu_checks_passed"] = all(v["reachable"] for v in initial.values()) and \
        report["initial_min_rigid_clearance_m"] >= .005 and \
        report["open_route"]["passed_cpu_geometry"] and before == after
    REPORT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: report[k] for k in ("status", "cpu_checks_passed",
        "initial_min_rigid_clearance_m", "initial_hand_tcp_separation_m", "open_route",
        "protected_existing_files_unchanged")}, indent=2))
    print(f"REPORT {REPORT}")


if __name__ == "__main__":
    main()
