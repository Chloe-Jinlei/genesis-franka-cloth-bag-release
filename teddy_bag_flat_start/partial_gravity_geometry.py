"""CPU geometry predictions for an explicit partial-gravity candidate schema.

No simulator or soft-state setters. Frozen predictions never certify native
contact, stable support, or final containment. No native state is modified.
"""
from __future__ import annotations

import numpy as np
from validate import containment, mouth_structure, signed_volumes

MODE = "partial_gravity_v1"
KIND = "guarded_partial_gravity_trial_v1"


def planning_mode(config):
    mode = config.get("task", {}).get("contact_trial", {}).get("planning_mode", "full_depth")
    if type(mode) is not str or mode not in ("full_depth", MODE):
        raise ValueError("Unknown contact planning_mode")
    return mode


def partial_policy(config):
    if planning_mode(config) != MODE:
        return None
    trial = config["task"]["contact_trial"]
    spec = trial.get("gravity_release")
    if not isinstance(spec, dict) or set(spec) != {
            "mode", "tcp_height_above_table_m", "minimum_inside_vertex_fraction", "settle_seconds"} \
            or spec["mode"] != "gravity_seating":
        raise ValueError("Partial mode requires the unchanged explicit gravity-seating policy")
    for key, lo, hi in (("tcp_height_above_table_m", .30, .30),
                        ("minimum_inside_vertex_fraction", .5, 1.), ("settle_seconds", 2., 10.)):
        value = spec[key]
        if isinstance(value, (bool, str)) or not np.isscalar(value) or not np.isfinite(value) \
                or not lo <= value <= hi:
            raise ValueError("Invalid partial-gravity option: " + key)
    if trial.get("insertion_strategy", "vertical") != "vertical" \
            or type(trial.get("recenter_enabled")) is not bool:
        raise ValueError("Partial mode requires fixed-XY descent and explicit recenter policy")
    return dict(planning_mode=MODE, schema_version=1,
                release_tcp_z_m=float(config["table"]["height"])+.30,
                minimum_inside_vertex_fraction=float(spec["minimum_inside_vertex_fraction"]),
                frozen_path_allowance_m=.001,
                maximum_mouth_relative_com_offset_m=.035,
                candidate_ranking="maximum_minimum_normalized_capture_and_section_margin",
                full_depth_preflight_performed=False)


def require_partial_candidate(candidate, policy):
    if not isinstance(candidate,dict) or candidate.get("kind")!=KIND \
            or candidate.get("schema_version")!=1 or candidate.get("planning_mode")!=MODE \
            or candidate.get("full_depth_preflight_performed") is not False \
            or candidate.get("status")!="candidate" or candidate.get("physical_success") is not False \
            or candidate.get("report",{}).get("policy")!=policy \
            or candidate.get("report",{}).get("rejections")!=[]:
        raise ValueError("Invalid or switched partial-candidate schema/policy")
    route=candidate.get("route")
    if not isinstance(route,dict) or route.get("kind")!=KIND or route.get("schema_version")!=1 \
            or route.get("planning_mode")!=MODE or route.get("full_depth_preflight_performed") is not False:
        raise ValueError("Invalid partial-route schema")


def validate_commands(context, command_qs):
    if not isinstance(command_qs, dict) or set(command_qs) != {"left", "right"}:
        raise ValueError("Partial mode requires explicit actual controller arm command endpoints")
    result = {}
    for side in ("left", "right"):
        q = np.asarray(command_qs[side], float)
        if q.shape != (7,) or not np.isfinite(q).all():
            raise ValueError("Invalid explicit command endpoint")
        pc, rc = context["planner"].fk(side, q)
        pa, ra = context["planner"].fk(side, context["qs"][side])
        if np.linalg.norm(pc-pa) > .0005 or (rc*ra.inv()).magnitude() > .002:
            raise ValueError("Actual and command poses disagree beyond 0.5 mm / 2 mrad")
        result[side] = q.copy()
    return result


def predict_endpoint(rest, state, config, target):
    spec = partial_policy(config)
    target = np.asarray(target, float)
    if spec is None or target.shape != (3,) or not np.isfinite(target).all() \
            or abs(target[2]-spec["release_tcp_z_m"]) > 1e-12:
        raise ValueError("Invalid direct partial release endpoint")
    # Fresh arrays only; no caller array or native state is modified.
    predicted = np.asarray(state["bear"]) + target-np.asarray(state["tcp_right"])
    body, rim, topology = mouth_structure(rest["bag"], rest["bag_faces"], config)
    if not len(rim) or not topology["two_panels_directly_sewn_along_u"]:
        raise ValueError("Invalid actual body topology for frozen capture forecast")
    weights = np.abs(signed_volumes(rest["bear"], rest["bear_tets"]))
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Invalid native rest mass weights")
    captured = containment(state["bag"], predicted, body, rim, weights, rest["bear_tets"])
    checks = dict(valid_analysis_body=bool(captured["virtual_closed_body_valid"]),
                  mass_COM_inside=bool(captured["bear_center_of_mass_inside"]),
                  node_fraction_at_least_declared_minimum=bool(
                      captured["bear_vertex_inside_fraction"] >= spec["minimum_inside_vertex_fraction"]))
    return dict(kind="frozen_partial_capture_forecast_v1", checks=checks,
                admissible=all(checks.values()), target_tcp_m=target.tolist(),
                predicted_bear_bbox_min_m=predicted.min(0).tolist(),
                predicted_bear_bbox_max_m=predicted.max(0).tolist(), containment=captured,
                physical_success=False, actual_release_review_still_required=True,
                scope="Frozen observed cloth and translated observed teddy only; no native state change, contact/support or final-containment proof.")


def check_table_segment(bear, source_tcp, start, end, config, policy):
    # A linearly translated frozen shape has its lowest Z at one endpoint.
    lower = float(np.asarray(bear)[:, 2].min() + min(
        float(np.asarray(start)[2]-np.asarray(source_tcp)[2]),
        float(np.asarray(end)[2]-np.asarray(source_tcp)[2])))
    allowance = partial_policy(config)["frozen_path_allowance_m"]
    required = float(config["table"]["height"])+policy["bottom_clearance_m"]
    bound = lower-allowance
    if not np.isfinite([lower, bound, required]).all() or bound < required:
        raise ValueError("Executed partial path lacks original table clearance plus 1 mm allowance")
    return dict(ideal_frozen_path_min_z_m=lower, allowance_m=allowance,
                conservative_min_z_m=bound, required_min_z_m=required,
                margin_m=bound-required, continuous_ideal_translation_bound=True,
                scope="Exact frozen straight-translation Z bound; allowance is not a proof of native shape or continuous FK error.")


def curve_record(plan):
    return dict(duration_s=float(plan.duration),
                spline_independent_variable="path fraction in [0,1]",
                time_scaling="original MotionPlan minimum_jerk(t/duration), not a seconds-domain spline",
                curves={s: dict(x=plan.curves[s].x.tolist(), c=plan.curves[s].c.tolist())
                        for s in ("left", "right")},
                geometry_finger_start=dict(plan.finger_start),
                geometry_finger_end=dict(plan.finger_end),
                execution_must_keep_existing_finite_motor_targets=True)
