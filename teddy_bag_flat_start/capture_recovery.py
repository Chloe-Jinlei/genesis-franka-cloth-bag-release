"""Bounded native-contact-trial planning after an actual capture-only rejection.

All soft translations here are forecasts. The controller executes only the saved
robot curve, keeping real IPC contact and the original holding/physical guards.
No motion packet authorizes release; the original actual release checks do so.
"""
from __future__ import annotations

import hashlib
import json
import numpy as np
from scipy.spatial.transform import Rotation

from contact_insertion import _policy, _state_context, _serial
from contact_robot_geometry import check_translation_sweep
from partial_gravity_geometry import MODE as PARTIAL_MODE, partial_policy, validate_commands, check_table_segment, curve_record
from postlift_yaw import _finite_tree

MODE = "bounded_capture_recovery_v1"
KIND = "bounded_capture_recovery_step_v1"
CAPTURE_CHECKS = {"actual_mass_COM_inside", "actual_all_node_fraction_at_least_declared_minimum"}
RELEASE_CHECKS = CAPTURE_CHECKS | {
    "actual_state_has_no_original_preflight_rejection", "actual_time_after_candidate",
    "actual_endpoint_within_2mm", "actual_full_orientation_retained",
    "actual_joint_trace_matches_endpoint_time", "actual_joint_trace_matches_saved_endpoint_joints",
    "actual_right_joint_velocity_below_0_05_rad_s", "contact_row_is_current_and_still_holding",
    "both_right_opening_reactions_above_original_0_1N", "actual_analysis_body_valid"}


def recovery_policy(config):
    value = config.get("task", {}).get("contact_trial", {}).get("capture_recovery")
    if value is None:
        return None
    if value != {"mode": MODE} or partial_policy(config) is None:
        raise ValueError("Recovery requires its exact opt-in mode and unchanged partial policy")
    task = config["task"]
    if task.get("bear_finger_target") != .012 or task.get("bear_finger_force_limit") != 1.5:
        raise ValueError("Recovery must retain the reviewed finite 1.5 N / 12 mm finger command")
    return dict(mode=MODE, maximum_steps=3, maximum_step_xy_m=.02,
                maximum_step_down_m=.02, maximum_cumulative_down_m=.06,
                maximum_global_actual_xy_m=.05, tracking_reserve_m=.001,
                maximum_robot_candidates=3)


def _digest(value):
    return hashlib.sha256(json.dumps(_serial(value), sort_keys=True, separators=(",", ":"),
                                    allow_nan=False).encode()).hexdigest()


def _vec(value, n, label):
    result = np.asarray(value, float)
    if result.shape != (n,) or not np.isfinite(result).all():
        raise ValueError("Invalid " + label)
    return result


def _number(value, label):
    array = np.asarray(value)
    if array.shape != () or array.dtype.kind not in "fiu" or not np.isfinite(array):
        raise ValueError("Invalid " + label)
    return float(array)


def _eligible(review):
    checks = review.get("checks", {})
    failed = {k for k, v in checks.items() if v is False}
    if review.get("status") != "rejected" or set(checks) != RELEASE_CHECKS \
            or any(type(v) is not bool for v in checks.values()) \
            or not failed or not failed <= CAPTURE_CHECKS:
        raise ValueError("Recovery is restricted to actual capture/COM-only release rejection")


def initialize_recovery_budget(candidate, state, actual_cumulative_xy_m):
    """Use measured previous correction travel; never reset it to a planned sum."""
    if candidate.get("kind") != "explicit_partial_gravity_release_v1" \
            or candidate.get("planning_mode") != PARTIAL_MODE \
            or candidate.get("status") != "candidate" or not _finite_tree(candidate):
        raise ValueError("Recovery must start from the original partial release candidate")
    original = _vec(candidate["insert_stage"]["right_tcp_m"], 3, "original target")
    actual = _vec(state["tcp_right"], 3, "actual recovery start")
    used = _number(actual_cumulative_xy_m, "measured preceding correction travel")
    now = _number(state["time"], "actual recovery start time")
    if not 0 <= used <= .05 or np.linalg.norm(original-actual) > .002 \
            or now <= candidate["observed_time_s"]:
        raise ValueError("Invalid original recovery anchor or cumulative actual XY budget")
    anchor = dict(initial_nominal_tcp_m=original.tolist(), initial_actual_tcp_m=actual.tolist(),
                  original_rotation_xyzw=candidate["insert_stage"]["rotation_xyzw"],
                  outside_tcp_m=candidate["outside_tcp_m"],
                  original_candidate_sha256=_digest(candidate),
                  original_candidate_observed_time_s=candidate["observed_time_s"],
                  initial_actual_time_s=now, initial_cumulative_actual_xy_m=used)
    return dict(mode=MODE, steps_completed=0, cumulative_down_m=0.,
                cumulative_actual_xy_m=used, reference_anchor=anchor,
                observations=[], last_actual_time_s=now, last_actual_tcp_m=actual.tolist(),
                last_candidate_sha256=_digest(candidate))


def _budget(budget, config):
    policy = recovery_policy(config)
    if policy is None or not isinstance(budget, dict) or budget.get("mode") != MODE \
            or not _finite_tree(budget) or type(budget.get("steps_completed")) is not int:
        raise ValueError("Invalid recovery budget")
    anchor = budget["reference_anchor"]
    nominal = _vec(anchor["initial_nominal_tcp_m"], 3, "nominal anchor")
    if abs(nominal[2]-partial_policy(config)["release_tcp_z_m"]) > 1e-12:
        raise ValueError("Recovery changed the original nominal 1.04 m release anchor")
    initial = _vec(anchor["initial_actual_tcp_m"], 3, "actual anchor")
    if np.linalg.norm(initial-nominal) > .002:
        raise ValueError("Initial actual recovery point is not the original endpoint")
    rotation = _vec(anchor["original_rotation_xyzw"], 4, "anchor rotation")
    if abs(np.linalg.norm(rotation)-1) > 1e-8:
        raise ValueError("Invalid anchor orientation")
    _vec(anchor["outside_tcp_m"], 3, "anchor withdrawal")
    if not 0 <= budget["steps_completed"] <= policy["maximum_steps"] \
            or len(budget["observations"]) != budget["steps_completed"]:
        raise ValueError("Recovery observation history does not match step budget")
    xy = _number(anchor["initial_cumulative_actual_xy_m"], "initial measured XY")
    down = 0.; last = initial; when = _number(anchor["initial_actual_time_s"], "anchor time")
    digest = anchor["original_candidate_sha256"]
    if not 0 <= xy <= policy["maximum_global_actual_xy_m"]:
        raise ValueError("Invalid initial measured XY budget")
    for index, row in enumerate(budget["observations"], 1):
        end = _vec(row["actual_tcp_m"], 3, "observed endpoint")
        t = _number(row["actual_time_s"], "observed time")
        distance = float(np.linalg.norm(end[:2]-last[:2])); dz = float(last[2]-end[2])
        if row["step_index"] != index or t <= when \
                or distance > policy["maximum_step_xy_m"] \
                or dz < -policy["tracking_reserve_m"] \
                or dz > policy["maximum_step_down_m"] \
                or abs(row["actual_xy_m"]-distance)>1e-12 \
                or abs(row["actual_down_m"]-max(0.,dz))>1e-12:
            raise ValueError("Invalid observed recovery step or reset history")
        xy += distance; down += max(0., dz); last = end; when = t; digest = row["candidate_sha256"]
    if xy > policy["maximum_global_actual_xy_m"] or down > policy["maximum_cumulative_down_m"] \
            or abs(xy-budget["cumulative_actual_xy_m"])>1e-12 \
            or abs(down-budget["cumulative_down_m"])>1e-12 \
            or when != budget["last_actual_time_s"] \
            or not np.array_equal(last, budget["last_actual_tcp_m"]) \
            or digest != budget["last_candidate_sha256"]:
        raise ValueError("Recovery cumulative measured budget or provenance changed")
    return policy


def require_recovery_candidate(candidate, config):
    """Schema validation only; never replaces the actual release checks."""
    policy = recovery_policy(config)
    if policy is None or candidate.get("kind") != KIND or candidate.get("schema_version") != 1 \
            or candidate.get("recovery_mode") != MODE or candidate.get("planning_mode") != PARTIAL_MODE \
            or candidate.get("status") != "candidate" or candidate.get("release_authorized") is not False \
            or candidate.get("physical_success") is not False \
            or candidate.get("full_depth_preflight_performed") is not False \
            or candidate.get("recovery_policy") != policy or candidate.get("contact_policy") != _policy(config) \
            or not _finite_tree(candidate):
        raise ValueError("Recovery release candidate mode/schema/policy changed")
    before = candidate["budget_before"]
    _budget(before, config)
    anchor = before["reference_anchor"]
    start = _vec(candidate["source_tcp_m"], 3, "recovery source")
    target = _vec(candidate["insert_stage"]["right_tcp_m"], 3, "recovery target")
    delta = target-start; xy = float(np.linalg.norm(delta[:2])); down = float(-delta[2])
    if candidate["reference_anchor"] != anchor \
            or not np.array_equal(start, before["last_actual_tcp_m"]) \
            or candidate["observed_time_s"] != before["last_actual_time_s"] \
            or candidate["step_index"] != before["steps_completed"]+1 \
            or candidate["step_index"] > policy["maximum_steps"] \
            or not 0 <= xy <= policy["maximum_step_xy_m"]-policy["tracking_reserve_m"]+1e-12 \
            or not 0 < down <= policy["maximum_step_down_m"]-policy["tracking_reserve_m"]+1e-12 \
            or before["cumulative_actual_xy_m"]+xy+policy["tracking_reserve_m"] > policy["maximum_global_actual_xy_m"] \
            or before["cumulative_down_m"]+down+policy["tracking_reserve_m"] > policy["maximum_cumulative_down_m"] \
            or candidate["outside_tcp_m"] != anchor["outside_tcp_m"] \
            or abs(candidate["initial_nominal_tcp_z_m"]-partial_policy(config)["release_tcp_z_m"])>1e-12:
        raise ValueError("Recovery exceeded or changed its anchored movement budget")
    rotation = Rotation.from_quat(candidate["insert_stage"]["rotation_xyzw"])
    if (rotation*Rotation.from_quat(anchor["original_rotation_xyzw"]).inv()).magnitude() > .015:
        raise ValueError("Recovery no longer retains original orientation")
    if candidate["insert_stage"]["command_q_end"] != candidate["command_q_end"] \
            or candidate["insert_stage"]["exact_reference_curve"] != candidate["exact_reference_curve"]:
        raise ValueError("Recovery execution packet differs from its accepted exact curve")
    direction = candidate["selected_geometry"]
    prediction = direction["prediction"]
    path = direction["contact_prediction"]["COM_path"]
    if not np.allclose(direction["delta_m"], delta, rtol=0., atol=1e-12) \
            or direction["observed_time_s"] != candidate["observed_time_s"] \
            or path.get("admissible") is not True \
            or path.get("physical_cloth_crossing") is not False \
            or path.get("physical_cloth_intersection_indeterminate") is not False \
            or path.get("analysis_cap_intersection_indeterminate") is not False \
            or prediction["containment"]["bear_center_of_mass_inside"] is not True \
            or prediction["containment"]["bear_vertex_inside_fraction"] < candidate["policy"]["minimum_inside_vertex_fraction"] \
            or prediction["capture_fraction_improvement"] <= 0 \
            or prediction["COM_inside_boundary_distance_m"] < .002:
        raise ValueError("Recovery does not preserve its cap-only COM path and bounded contact prediction")


def observe_recovery_endpoint(candidate, state, budget, config):
    """Count actual endpoint XY/down movement before any next trial or release."""
    require_recovery_candidate(candidate, config)
    if budget != candidate["budget_before"]:
        raise ValueError("Recovery observation did not continue its exact prior budget")
    end = _vec(state["tcp_right"], 3, "native recovery endpoint")
    now = _number(state["time"], "native recovery endpoint time")
    if now <= candidate["observed_time_s"] \
            or np.linalg.norm(end-np.asarray(candidate["insert_stage"]["right_tcp_m"])) > .002:
        raise ValueError("Recovery lacks a newer actual endpoint within 2 mm")
    start = np.asarray(candidate["source_tcp_m"])
    xy = float(np.linalg.norm(end[:2]-start[:2])); down = max(0., float(start[2]-end[2]))
    updated = json.loads(json.dumps(budget))
    updated["observations"].append(dict(step_index=candidate["step_index"], actual_time_s=now,
        actual_tcp_m=end.tolist(), actual_xy_m=xy, actual_down_m=down, candidate_sha256=_digest(candidate)))
    updated.update(steps_completed=candidate["step_index"], cumulative_actual_xy_m=budget["cumulative_actual_xy_m"]+xy,
                   cumulative_down_m=budget["cumulative_down_m"]+down, last_actual_time_s=now,
                   last_actual_tcp_m=end.tolist(), last_candidate_sha256=_digest(candidate))
    _budget(updated, config)
    return updated


def plan_capture_recovery(rest, state, config, previous_candidate, previous_actual_review,
                          contact_row, joint_trace, command_qs, *, budget, return_plan=False):
    from gravity_release import review_actual_release, release_policy, _checked_motion, _opening_and_withdrawal
    from capture_recovery_geometry import rank_recovery_directions
    limits = _budget(budget, config)
    if budget["steps_completed"] >= limits["maximum_steps"] \
            or budget["last_candidate_sha256"] != _digest(previous_candidate) \
            or float(state["time"]) != budget["last_actual_time_s"] \
            or not np.array_equal(state["tcp_right"], budget["last_actual_tcp_m"]):
        raise ValueError("Recovery exhausted or did not continue its last actual observation")
    _eligible(previous_actual_review)
    if previous_actual_review["actual_time_s"] != float(state["time"]):
        raise ValueError("Stale actual capture failure")
    actual = review_actual_release(rest, state, config, previous_candidate, contact_row, joint_trace, command_qs)
    _eligible(actual)
    if actual["checks"] != previous_actual_review["checks"] \
            or actual["actual_containment"] != previous_actual_review["actual_containment"]:
        raise ValueError("Supplied capture failure differs from current actual reobservation")
    context = _state_context(rest, state, config, _policy(config))
    if context["reasons"]:
        raise ValueError("Actual state has an original physical or robot rejection")
    commanded = validate_commands(context, command_qs)
    if any(not np.array_equal(commanded[s], previous_candidate["command_q_end"][s]) for s in ("left", "right")):
        raise ValueError("Recovery changed the previous actual controller command endpoint")
    ranking = rank_recovery_directions(rest, state, config,
        maximum_xy_m=min(.019, .05-budget["cumulative_actual_xy_m"]-.001),
        maximum_down_m=min(.019, .06-budget["cumulative_down_m"]-.001))
    failures = []; chosen = None
    for direction in ranking["candidates"][:limits["maximum_robot_candidates"]]:
        try:
            start = np.asarray(state["tcp_right"], float); delta = _vec(direction["delta_m"], 3, "ranked recovery delta")
            target = start+delta
            if np.linalg.norm(delta[:2])>.019+1e-12 or not -.019-1e-12 <= delta[2] < 0:
                raise ValueError("Geometry returned an out-of-budget displacement")
            table = check_table_segment(state["bear"], start, start, target, config, _policy(config))
            sweeps = {}
            for label, left in (("actual", context["qs"]["left"]), ("commanded", commanded["left"])):
                sweep = check_translation_sweep(np.asarray(state["bear"]), np.asarray(rest["bear_tets"], int),
                    np.r_[left, state["q_left"][7:]], config, translation_m=np.zeros(3), displacement_m=delta,
                    required_gap_m=_policy(config)["left_robot_soft_sweep_clearance_m"]+.001, planner=context["planner"])
                if not _finite_tree(sweep) or sweep["status"] != "clear":
                    raise ValueError("Recovery fails frozen teddy/left robot sweep plus 1 mm allowance")
                sweeps[label] = sweep
            rotation = Rotation.from_quat(budget["reference_anchor"]["original_rotation_xyzw"])
            plan, proof = _checked_motion(context, commanded, target, rotation, 2., context["fingers"])
            opening, _ = _opening_and_withdrawal(context, plan.end_q(), budget["reference_anchor"]["outside_tcp_m"], rotation)
            exact = curve_record(plan); ends = {s:q.tolist() for s,q in plan.end_q().items()}
            stage = dict(name="capture_recovery", right_tcp_m=target.tolist(), rotation_xyzw=rotation.as_quat().tolist(),
                         duration_s=plan.duration, exact_reference_curve=exact, command_q_end=ends)
            chosen = dict(kind=KIND, schema_version=1, status="candidate", recovery_mode=MODE,
                planning_mode=PARTIAL_MODE, full_depth_preflight_performed=False,
                recovery_policy=limits, policy=release_policy(config), contact_policy=_policy(config),
                observed_time_s=float(state["time"]), source_tcp_m=start.tolist(),
                reference_anchor=budget["reference_anchor"], initial_nominal_tcp_z_m=partial_policy(config)["release_tcp_z_m"],
                budget_before=budget, step_index=budget["steps_completed"]+1,
                insert_stage=stage, outside_tcp_m=budget["reference_anchor"]["outside_tcp_m"],
                command_q_start={s:q.tolist() for s,q in commanded.items()}, command_q_end=ends,
                continuous_reference_zero_endpoint_velocity_acceleration=all(np.max(np.abs(plan.sample(t)[key][s]))<1e-10
                    for t in (0.,plan.duration) for key in ("qd","qdd") for s in ("left","right")),
                exact_reference_curve=exact, insert_curve_review=proof, joint_extrema=plan.joint_extrema(),
                table_path_review=table, left_robot_soft_sweeps=sweeps,
                release_opening_and_withdrawal_preflight=opening, selected_geometry=direction,
                candidate_search=ranking, robot_candidate_rejections=failures,
                actual_source_release_review=actual, release_authorized=False, physical_success=False,
                requires_actual_partial_capture_before_release=True,
                requires_unchanged_final_full_containment_and_retention=True,
                scope="Bounded native contact trial only. Frozen node paths may meet real cloth; only COM avoids the physical wall and enters through the analysis mouth cap. No whole-body cloth-safe sweep or actual capture is claimed. Execute only the robot reference, retain native IPC and every original holding/physical guard, then observe actual capture again.")
            require_recovery_candidate(chosen, config)
            if not chosen["continuous_reference_zero_endpoint_velocity_acceleration"]:
                raise ValueError("Recovery curve does not stop smoothly")
            break
        except (ValueError, RuntimeError) as exc:
            failures.append(dict(delta_m=direction["delta_m"], reason=str(exc)))
            chosen = None
    if chosen is None:
        raise ValueError("No bounded recovery direction passed original robot/table/left-soft/opening/withdrawal checks: "
                         + json.dumps(failures, allow_nan=False))
    return (chosen, plan) if return_plan else chosen
