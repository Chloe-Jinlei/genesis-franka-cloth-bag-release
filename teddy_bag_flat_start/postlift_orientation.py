"""PROPOSED ONLY: guarded post-lift world-yaw plus world-axis tilt preflight.

No native execution and no active-runner integration. Return the exact checked
MotionPlan for separate review; a future runner would still need finite-force
controls, original physical guards and a fresh actual post-motion observation.
Frozen rigid-body geometry is a forecast, never observed soft-body motion.
"""
from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from motion import MotionPlan, CubicSpline, minimum_jerk
from contact_insertion import _policy, _state_context
from contact_robot_geometry import _convex, convex_clearance_certificate, left_collision_solids

SIDES = ("left", "right")
MAX_POSTLIFT_YAW_DEGREES = 90
MAX_POSTLIFT_TILT_DEGREES = 40
RIGID_CATEGORIES = {"left_self", "left_table", "left_mounts", "right_self", "right_table", "right_mounts", "arms"}


def _finite_tree(value):
    if isinstance(value, Mapping): return all(_finite_tree(v) for v in value.values())
    if isinstance(value, (list, tuple)): return all(_finite_tree(v) for v in value)
    if isinstance(value, np.ndarray): return bool(np.isfinite(value).all()) if value.dtype.kind in "biufc" else False
    if isinstance(value, (Real, np.number)): return bool(np.isfinite(value))
    return value is None or isinstance(value, str)


def _serial(value):
    if isinstance(value, Mapping): return {str(k): _serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [_serial(v) for v in value]
    if isinstance(value, np.ndarray): return _serial(value.tolist())
    if isinstance(value, np.generic): return _serial(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def _number(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(name+" must be a finite numeric scalar")
    return float(value)


def _clearance_rows(checks, label):
    if not isinstance(checks, Mapping) or set(checks) != RIGID_CATEGORIES:
        raise ValueError(label+": requires all seven original raw rigid clearance categories")
    for name, row in checks.items():
        if not isinstance(row, Mapping): raise ValueError(label+": malformed "+str(name))
        distance = _number(row.get("distance"), label+" "+str(name)+" distance")
        if not _finite_tree(row): raise ValueError(label+": nonfinite raw rigid evidence")
        if distance < .008:
            raise ValueError(f"{label}: {name} rigid clearance {distance:.9f} m below 0.008 m; bodies={row.get('bodies')}")
    return min(checks.items(), key=lambda item: float(item[1]["distance"]))


def continuous_rate_bounds(plan):
    """Global analytic derivative bounds for cubic q(s) composed with min-jerk.

    q_s is quadratic (check endpoints and its vertex); q_ss is linear.
    max s_u=15/8 and max |s_uu|=10/sqrt(3), so the chain rule bounds
    every time, independently of MotionPlan's continuous-extrema retiming.
    """
    duration = _number(plan.duration, "plan duration")
    if duration <= 0: raise ValueError("plan duration must be positive")
    if not isinstance(plan.curves, Mapping) or set(plan.curves) != set(SIDES):
        raise ValueError("Continuous rate bounds require both original arm curves")
    result = {}
    for side in SIDES:
        curve = plan.curves[side]
        if not hasattr(curve,"x") or not hasattr(curve,"c"):
            raise ValueError("Missing cubic knot/coefficient arrays")
        if curve.x.ndim != 1 or len(curve.x)<2 or curve.c.shape != (4,len(curve.x)-1,7) \
                or not np.isfinite(curve.x).all() or not np.isfinite(curve.c).all() \
                or curve.x[0] != 0. or curve.x[-1] != 1. or np.any(np.diff(curve.x)<=0):
            raise ValueError("Finite seven-joint cubic must cover [0,1] exactly with increasing knots")
        first = np.zeros(7); second = np.zeros(7)
        for k, width in enumerate(np.diff(curve.x)):
            if width <= 0: raise ValueError("Cubic knot intervals must be positive")
            a, b, c = curve.c[:3, k]
            for j in range(7):
                xs = [0., float(width)]
                if a[j] != 0:
                    vertex = -b[j]/(3*a[j])
                    if 0. < vertex < width: xs.append(float(vertex))
                first[j] = max(first[j], *[abs(3*a[j]*x*x+2*b[j]*x+c[j]) for x in xs])
                second[j] = max(second[j], abs(2*b[j]), abs(6*a[j]*width+2*b[j]))
        velocity = first*(15/8)/duration
        acceleration = (second*(15/8)**2+first*(10/np.sqrt(3)))/duration**2
        if not np.isfinite([first, second, velocity, acceleration]).all():
            raise ValueError("Nonfinite continuous derivative bound")
        # Explicit floating evaluation guard; no directed-rounding claim.
        result[side] = dict(path_first_derivative_bound=first, path_second_derivative_bound=second,
            velocity_upper_bound_rad_s=velocity+1e-10,
            acceleration_upper_bound_rad_s2=acceleration+1e-10)
    return result


def _orientation_option(option):
    """Validate the exact proposed option and construct a WORLD-frame delta.

    Composition is tilt-after-yaw: R_tilt(axis_xy) * R_z(yaw). A negligible
    norm-roundoff is normalized only after the axis passes the unit-vector test.
    Identity orientation is supported by the declared inclusive angle ranges.
    """
    expected = {"world_z_degrees", "tilt_axis_xy", "tilt_degrees", "duration_s"}
    if not isinstance(option, Mapping) or set(option) != expected:
        raise ValueError("postlift_orientation requires exactly "+", ".join(sorted(expected)))
    degrees = _number(option["world_z_degrees"], "world_z_degrees")
    tilt_degrees = _number(option["tilt_degrees"], "tilt_degrees")
    duration = _number(option["duration_s"], "duration_s")
    axis_option = option["tilt_axis_xy"]
    if not isinstance(axis_option, (list, tuple, np.ndarray)):
        raise ValueError("tilt_axis_xy must be a finite numeric unit 2-vector")
    raw_axis = np.asarray(axis_option)
    if raw_axis.shape != (2,) or raw_axis.dtype.kind not in "fiu" or any(
            isinstance(value, (bool, np.bool_)) for value in axis_option):
        raise ValueError("tilt_axis_xy must be a finite numeric unit 2-vector")
    axis_xy = raw_axis.astype(float)
    norm = float(np.linalg.norm(axis_xy))
    if not np.isfinite(axis_xy).all() or not np.isfinite(norm) or abs(norm-1.) > 1e-10:
        raise ValueError("tilt_axis_xy must have unit norm within 1e-10")
    if abs(degrees) > MAX_POSTLIFT_YAW_DEGREES or abs(tilt_degrees) > MAX_POSTLIFT_TILT_DEGREES or duration < 6:
        raise ValueError("postlift orientation requires |yaw|<=90, |tilt|<=40 and duration>=6s")
    axis_world = np.r_[axis_xy/norm, 0.]
    delta = (Rotation.from_rotvec(axis_world*np.deg2rad(tilt_degrees))
             * Rotation.from_rotvec([0., 0., np.deg2rad(degrees)]))
    parameters = dict(world_z_degrees=degrees, tilt_axis_xy=axis_xy.tolist(),
                      tilt_degrees=tilt_degrees, duration_s=duration)
    return parameters, delta


def target_rotation(option, current_rotation):
    """Pure validated R_tilt(world axis) * R_z(world yaw) * current_R."""
    _, delta = _orientation_option(option)
    if not isinstance(current_rotation, Rotation) or current_rotation.as_quat().shape != (4,):
        raise ValueError("current_rotation must be one scipy Rotation")
    return delta*current_rotation


def _rotation_arc_geometry(relative_points, rotation_delta):
    """Enclose every segment's rigid rotation arc by its endpoint hull + ball.

    For total axis a and angle theta, a point's radius is |relative x a|.
    The segment sagitta is r_max * (1-cos(theta/(2*n))). The equivalent
    2*sin(theta/(4*n))**2 expression avoids cancellation for small angles.
    The same convex envelope contains every triangle/tetrahedron interior.
    """
    relative = np.asarray(relative_points, float)
    rotvec = np.asarray(rotation_delta.as_rotvec(), float)
    if relative.ndim != 2 or relative.shape[1] != 3 or not len(relative) or not np.isfinite(relative).all():
        raise ValueError("Rotation sweep needs finite nonempty relative points of shape (N,3)")
    if rotvec.shape != (3,) or not np.isfinite(rotvec).all():
        raise ValueError("Rotation sweep needs one finite total world rotation")
    angle = float(np.linalg.norm(rotvec))
    axis = rotvec/angle if angle > 0. else np.array([1., 0., 0.])
    radius = float(np.linalg.norm(np.cross(relative, axis), axis=1).max()) if angle > 0. else 0.
    count = max(1, int(np.ceil(angle/np.deg2rad(5.))))
    sagitta = float(2.*radius*np.sin(angle/(4.*count))**2)
    return dict(rotation_vector_world_rad=rotvec, rotation_axis_world=axis,
                rotation_axis_is_arbitrary_for_identity=bool(angle == 0.),
                total_rotation_angle_rad=angle, segment_count=count,
                maximum_axis_perpendicular_vertex_radius_m=radius,
                sagitta_outer_bound_m=sagitta)


def _rotation_table_sweep(state, rotation_delta, table_height):
    """Conservative complete frozen rotation clearance above the table plane.

    Each endpoint convex hull has minimum Z equal to the smallest endpoint
    vertex Z. Expanding that hull by its 3D sagitta decreases this bound by
    at most the sagitta. A further 1e-7 m numerical guard matches the original
    separation-certificate padding. Strictly more than 40 mm is required.
    This is a frozen-shape geometric certificate, not dynamic soft-body safety.
    """
    table = _number(table_height, "table height")
    pivot = np.asarray(state["tcp_right"], float)
    if pivot.shape != (3,) or not np.isfinite(pivot).all():
        raise ValueError("Table rotation sweep needs one finite source TCP")
    rel = np.asarray(state["bear"], float)-pivot
    geometry = _rotation_arc_geometry(rel, rotation_delta)
    count = geometry["segment_count"]; rotvec = geometry["rotation_vector_world_rad"]
    sagitta = geometry["sagitta_outer_bound_m"]; guard = 1e-7
    required = .04; threshold = table+required; segments = []
    for index in range(count):
        fractions = (index/count, (index+1)/count)
        endpoint_minimum_z = [float((Rotation.from_rotvec(rotvec*f).apply(rel)+pivot)[:,2].min())
                              for f in fractions]
        lower = min(endpoint_minimum_z)-sagitta-guard
        segments.append(dict(index=index, start_fraction=fractions[0], end_fraction=fractions[1],
            endpoint_minimum_z_m=endpoint_minimum_z,
            conservative_minimum_z_m=lower,
            conservative_clearance_above_table_m=lower-table))
    lower = min(row["conservative_minimum_z_m"] for row in segments)
    report = dict(status="clear" if lower > threshold else "rejected", physical_success=False,
        **geometry, source_tcp_world_m=pivot,
        rotation_delta_world_xyzw=rotation_delta.as_quat(),
        table_height_m=table, required_clearance_m=required,
        strict_minimum_z_threshold_m=threshold, numerical_guard_m=guard,
        segments=segments, conservative_minimum_z_m=lower,
        conservative_clearance_above_table_m=lower-table,
        scope="Continuous ideal rigid frozen-teddy total-axis rotation about the measured fixed TCP. Every vertex and convex triangle/tetrahedron interior remains above the infinite table-height plane by strictly more than 40mm when clear. No native deformation or tracking claim.")
    if not _finite_tree(report):
        raise ValueError("Nonfinite frozen bear/table rotation evidence")
    return report


def _rotation_sweep(planner, state, commanded_left, rotation_delta):
    pivot = np.asarray(state["tcp_right"], float)
    rel = np.asarray(state["bear"], float)-pivot
    geometry = _rotation_arc_geometry(rel, rotation_delta)
    count = geometry["segment_count"]; sagitta = geometry["sagitta_outer_bound_m"]
    rotvec = geometry["rotation_vector_world_rad"]
    report = dict(status="not_certified", physical_success=False, required_gap_m=.005,
        **geometry, left_poses_checked=["actual", "commanded"], per_pose=[],
        rotation_delta_world_xyzw=rotation_delta.as_quat(),
        scope="Continuous ideal rigid frozen-teddy arc along the shortest total WORLD-frame axis-angle/Slerp rotation about measured fixed TCP, against both static left poses. No dynamic or continuous joint-FK tracking certificate.")
    for label, qleft in (("actual", np.asarray(state["q_left"], float)),
                        ("commanded", np.r_[commanded_left, np.asarray(state["q_left"], float)[7:]])):
        solids = left_collision_solids(planner, qleft)
        if not solids: raise ValueError("Missing left collision solids")
        pose = dict(pose=label, q_left=qleft.tolist(), segments=[])
        report["per_pose"].append(pose)
        for k in range(count):
            ends = [Rotation.from_rotvec(rotvec*f).apply(rel)+pivot
                    for f in (k/count,(k+1)/count)]
            hull = _convex(np.vstack(ends)); rows = []
            for solid in solids:
                cert = convex_clearance_certificate(hull, solid, required_gap_m=.005+sagitta)
                if not isinstance(cert, Mapping) or not _finite_tree(cert):
                    raise ValueError("Nonfinite/malformed raw frozen rotation certificate")
                lower = _number(cert.get("conservative_gap_lower_bound_m"), "rotation certificate gap")-sagitta
                rows.append(dict(left_body=solid["body"], geom_id=solid["geom_id"], certificate=cert,
                                 continuous_arc_lower_bound_m=lower))
                if cert.get("status") != "clear" or lower < .005:
                    raise ValueError(f"Frozen orientation lacks 5mm left clearance for {label} {solid['body']} geom {solid['geom_id']}: {lower:.9f}m")
            pose["segments"].append(dict(index=k, per_geom=rows,
                minimum_continuous_gap_lower_bound_m=min(v["continuous_arc_lower_bound_m"] for v in rows)))
    report.update(status="clear", minimum_continuous_gap_lower_bound_m=min(
        segment["minimum_continuous_gap_lower_bound_m"] for pose in report["per_pose"] for segment in pose["segments"]))
    return report

def plan_postlift_orientation(rest, state, config, command_qs):
    """Return a JSON-safe report and the SAME verified plan, or rejected/None.

    A missing option is rejected if called; the caller should skip this
    helper entirely when task.contact_trial.postlift_orientation is absent.
    """
    report = dict(kind="guarded_postlift_orientation", status="rejected", physical_success=False,
        runtime_execution_performed=False, requires_fresh_actual_preparation_after_execution=True,
        modifies_soft_points=False, checks={}, rejections=[], limitations=[
            "Frozen rigid-grasp rotation does not predict soft deformation, slip, force, or eventual insertion.",
            "The complete 40mm table-height sweep is certified only for the ideal frozen rigid bear and fixed TCP; native deformation and tracking require the original runtime guards.",
            "Joint position and rate bounds cover the continuous reference spline; rigid robot clearances and FK errors are sampled.",
            "The left arm is checked at actual and commanded static poses; motion between these poses or native compliance is not certified.",
            "Finger geometry uses measured individual fingers for frozen left solids and paired means for robot clearance. The runner retains finite-force grasp motor targets.",
            "Analytic bounds are evaluated in double precision with explicit padding, not directed-rounding interval arithmetic.",
            "Native force/torque/grip/deformation/table gates remain mandatory; no task-success claim follows."])
    try:
        if not all(isinstance(x, Mapping) for x in (rest,state,config,command_qs)):
            raise ValueError("rest, state, config and command_qs must be mappings")
        option = config["task"]["contact_trial"]["postlift_orientation"]
        parameters, rotation_delta = _orientation_option(option)
        duration = parameters["duration_s"]
        report["parameters"] = parameters
        report["rotation_composition"] = "R_tilt(world horizontal axis) * R_z(world yaw) * actual_R"
        report["rotation_path"] = "Single shortest total-axis Slerp; not sequential yaw then tilt motions"
        if set(command_qs) != set(SIDES): raise ValueError("command_qs needs exactly left and right")
        commanded = {s:np.asarray(command_qs[s],float).copy() for s in SIDES}
        if any(q.shape!=(7,) or not np.isfinite(q).all() for q in commanded.values()):
            raise ValueError("Each command_qs must contain seven finite joints")
        context = _state_context(rest,state,config,_policy(config))
        report.update(observed_time_s=context["time"], actual_physical_state=context["physical"])
        if not _finite_tree(context["physical"]): raise ValueError("Nonfinite actual physical evidence")
        _clearance_rows(context["physical"]["actual_rigid_clearances"], "actual state")
        if context["reasons"]: raise ValueError("Actual physical state rejected: "+str(context["reasons"]))
        table = _number(config["table"]["height"], "table height")
        minimum_z = float(np.asarray(state["bear"])[:,2].min())
        if minimum_z <= table+.04: raise ValueError("Actual lifted bear must be strictly above table+40mm")
        table_sweep = _rotation_table_sweep(state, rotation_delta, table)
        report["frozen_bear_table_rotation_sweep"] = table_sweep
        if table_sweep["status"] != "clear":
            raise ValueError("Frozen orientation lacks strictly more than 40mm table clearance: "+
                             str(table_sweep["conservative_clearance_above_table_m"])+" m")
        planner = context["planner"]; actual_poses = {}; command_poses = {}
        for side in SIDES:
            bounds = planner.limits[side]
            if np.any(commanded[side]<bounds[:,0]) or np.any(commanded[side]>bounds[:,1]):
                raise ValueError(side+" command joint outside physical limit")
            pa,ra = planner.fk(side,context["qs"][side]); pc,rc = planner.fk(side,commanded[side])
            positional = float(np.linalg.norm(pc-pa)); angular = float((rc*ra.inv()).magnitude())
            if not np.isfinite([*pa,*pc,*ra.as_quat(),*rc.as_quat(),positional,angular]).all():
                raise ValueError("Nonfinite actual/command FK")
            if positional>.002 or angular>.015:
                raise ValueError(side+" command FK differs from actual by over 2mm or 15mrad")
            actual_poses[side] = dict(tcp_m=pa.tolist(),rotation_xyzw=ra.as_quat().tolist(),q=context["qs"][side].tolist())
            command_poses[side] = dict(tcp_m=pc.tolist(),rotation_xyzw=rc.as_quat().tolist(),q=commanded[side].tolist(),
                actual_position_error_m=positional,actual_rotation_error_rad=angular)
        tcp = np.asarray(state["tcp_right"],float).copy(); actual = context["rotation"]
        target = target_rotation(option, actual)
        report.update(actual_source_poses=actual_poses,command_source_poses=command_poses,
            actual_lifted_bear_minimum_z_m=minimum_z,target_tcp_m=tcp.tolist(),target_rotation_xyzw=target.as_quat().tolist())
        paths = planner.plan_cartesian(commanded,{"right":tcp},rotations={"right":target},samples=45)
        fractions = np.array([v["fraction"] for v in paths]); curves = {}
        if fractions.ndim!=1 or len(fractions)<2 or not np.isfinite(fractions).all() \
                or fractions[0]!=0. or fractions[-1]!=1. or np.any(np.diff(fractions)<=0):
            raise ValueError("IK fractions must cover [0,1] exactly and increase strictly")
        for side in SIDES:
            values = np.array([v["q"].get(side,commanded[side]) for v in paths],float)
            values[0] = commanded[side]  # exact existing motor-command continuity
            if not np.isfinite(values).all(): raise ValueError("Nonfinite IK path")
            curves[side] = CubicSpline(fractions,values,axis=0)
        plan = MotionPlan(curves,duration,context["fingers"],context["fingers"])
        extrema = plan.validate_joint_limits(planner.limits)
        if not _finite_tree(extrema): raise ValueError("Nonfinite analytic joint extrema")
        plan.limit_rate(.35,.5)
        rates = continuous_rate_bounds(plan)
        peak_v = max(float(np.max(v["velocity_upper_bound_rad_s"])) for v in rates.values())
        peak_a = max(float(np.max(v["acceleration_upper_bound_rad_s2"])) for v in rates.values())
        plan.duration *= max(1.,peak_v/.35,np.sqrt(peak_a/.5))*1.002
        rates = continuous_rate_bounds(plan)
        if any(np.any(v["velocity_upper_bound_rad_s"]>.35) or np.any(v["acceleration_upper_bound_rad_s2"]>.5) for v in rates.values()):
            raise ValueError("Continuous rate bound exceeds 0.35rad/s or 0.5rad/s2")
        expected = Slerp([0.,1.],Rotation.from_quat([actual.as_quat(),target.as_quat()]))
        minimum = None; worst = None; pmax = 0.; rmax = 0.; vmax = 0.; amax = 0.
        for time in np.linspace(0.,plan.duration,401):
            sample = plan.sample(float(time))
            if not _finite_tree(sample): raise ValueError("Nonfinite raw trajectory sample")
            if not np.array_equal(sample["q"]["left"],commanded["left"]): raise ValueError("Left command changed during orientation")
            for label,leftq in (("commanded",commanded["left"]),("actual",context["qs"]["left"])):
                qs = dict(left=leftq,right=sample["q"]["right"])
                category,row = _clearance_rows(planner.clearance(qs,sample["finger"]),f"orientation t={time:.6f} left={label}")
                if minimum is None or row["distance"]<minimum:
                    minimum = float(row["distance"]); worst = dict(time_s=float(time),left_pose=label,category=category,**row)
            position,rotation = planner.fk("right",sample["q"]["right"])
            fraction = float(minimum_jerk(time/plan.duration)[0])
            pe = float(np.linalg.norm(position-tcp)); re = float((rotation*expected(fraction).inv()).magnitude())
            if not np.isfinite([pe,re]).all() or pe>.0005 or re>.002:
                raise ValueError(f"Sampled actual-reference TCP/Slerp error exceeds 0.5mm/2mrad: {pe},{re}")
            pmax=max(pmax,pe);rmax=max(rmax,re)
            vmax=max(vmax,*[float(np.abs(sample["qd"][s]).max()) for s in SIDES])
            amax=max(amax,*[float(np.abs(sample["qdd"][s]).max()) for s in SIDES])
        sweep = _rotation_sweep(planner,state,commanded["left"],rotation_delta)
        end = plan.end_q()
        report.update(status="candidate",duration_s=float(plan.duration),right_q_end=end["right"].tolist(),left_q_end=end["left"].tolist(),
            robot_preflight=dict(rigid_sample_count=401,left_poses_checked=["actual","commanded"],
                minimum_sampled_rigid_clearance_m=minimum,worst_rigid_sample=worst,
                maximum_sampled_tcp_error_m=pmax,maximum_sampled_slerp_error_rad=rmax,
                sampled_peak_velocity_rad_s=vmax,sampled_peak_acceleration_rad_s2=amax,
                analytic_joint_extrema=extrema,continuous_rate_bounds=rates,
                continuous_rigid_clearance_certified=False,command_start_exactly_preserved=True,left_command_constant=True),
            frozen_bear_left_rotation_sweep=sweep)
        report["checks"] = dict(actual_original_physical_gates=True,lift_above_table_40mm=True,
            command_actual_pose_consistent=True,analytic_joint_limits=True,continuous_reference_rate_bounds=True,
            sampled_all_raw_rigid_checks_finite_and_above_8mm=True,sampled_actual_reference_fk_tolerances=True,
            actual_and_commanded_left_frozen_rotation_above_5mm=True,returns_exact_checked_plan=True,
            frozen_rotation_bear_table_above_40mm=True)
        if not _finite_tree(report): raise ValueError("Nonfinite completed preflight evidence")
        result = _serial(report); json.dumps(result,allow_nan=False)
        return result,plan
    except Exception as error:
        report.update(status="rejected",error_type=type(error).__name__)
        report["rejections"].append(str(error))
        result = _serial(report); json.dumps(result,allow_nan=False)
        return result,None
