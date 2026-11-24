"""Offline dual-Panda FK/IK, rigid clearance checks and continuous PD trajectories.

The model is compiled by MuJoCo only; this module does not initialize Genesis.
TCP is the centre of the original Panda fingertip pads, hand-local (0,0,.1029).
Collision distances use panda_non_overlap.xml convex meshes, so final validation
must also inspect the running Genesis contacts and measured TCP tracking.
"""
from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import copy
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from scipy.optimize import brentq, least_squares
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation, Slerp
from numpy.polynomial import polynomial as polynomial

HERE = Path(__file__).resolve().parent
ASSET = HERE.parent / "genesis-world/genesis/assets/xml/franka_emika_panda"
TCP = np.array([0.0, 0.0, 0.1029])
DOWN = Rotation.from_quat([1.0, 0.0, 0.0, 0.0])
PAD_BELOW_TCP = 0.0085
# The original finger collision mesh projects farther than its box-shaped pad.
FINGERTIP_BELOW_TCP = 0.009358
SIDES = ("left", "right")
# Keep IK knots slightly inside the physical limits.  The complete interpolant
# is checked analytically below; this margin never clips an executed command.
IK_JOINT_MARGIN = 0.001


def default_config():
    """Geometrically validated tabletop layout; SI units throughout."""
    return {
        "table": {"size": [1.7, 1.2, .06], "height": .74},
        "robots": {
            "mjcf_path": str(HERE / "assets/panda_tactile.xml"),
            "left_base": [-.60, -.28, .76],
            "right_base": [.38, .32, .76],
            "left_yaw": .35,
            "right_yaw": -2.4,
            "home": [0., -.45, 0., -2.15, 0., 1.7, .785398],
        },
    }


def make_model(config):
    model_path = Path(config["robots"].get("mjcf_path", HERE / "assets/panda_tactile.xml"))
    root = ET.parse(model_path).getroot()
    meshdir = Path(root.find("compiler").get("meshdir", "assets"))
    if not meshdir.is_absolute():
        meshdir = model_path.parent / meshdir
    root.find("compiler").set("meshdir", str(meshdir))
    for name in ("actuator", "keyframe", "tendon", "equality"):
        node = root.find(name)
        if node is not None:
            root.remove(node)
    world = root.find("worldbody")
    template = world.find("body")
    world.remove(template)
    for side in SIDES:
        body = copy.deepcopy(template)
        for element in body.iter():
            if "name" in element.attrib:
                element.set("name", side + "_" + element.get("name"))
        yaw = config["robots"][side + "_yaw"]
        base = config["robots"][side + "_base"]
        body.set("pos", " ".join(map(str, base)))
        body.set("quat", f"{math.cos(yaw/2)} 0 0 {math.sin(yaw/2)}")
        world.append(body)
        th = config["table"]["height"]
        ET.SubElement(world, "geom", name=side + "_mount", type="box",
            pos=f"{base[0]} {base[1]} {(th+base[2])/2}",
            size=f".095 .095 {(base[2]-th)/2}")
    sx, sy, sz = config["table"]["size"]
    th = config["table"]["height"]
    ET.SubElement(world, "geom", name="table", type="box",
        pos=f"0 0 {th-sz/2}", size=f"{sx/2} {sy/2} {sz/2}")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    return model, mujoco.MjData(model)


class Planner:
    def __init__(self, config=None):
        self.config = config if config is not None else default_config()
        self.home = np.asarray(self.config["robots"]["home"], dtype=float)
        self.m, self.d = make_model(self.config)
        self.qindices = {s: np.array([self.m.joint(f"{s}_joint{i}").qposadr[0]
                                     for i in range(1, 8)]) for s in SIDES}
        self.bodies = {s: self.m.body(s + "_hand").id for s in SIDES}
        self.limits = {s: self.m.jnt_range[[self.m.joint(f"{s}_joint{i}").id
                                          for i in range(1, 8)]].copy() for s in SIDES}
        self.geoms = {s: [i for i in range(self.m.ngeom) if self.m.geom_contype[i]
                         and self.body_name(i).startswith(s + "_")
                         and not self.body_name(i).endswith("link0")] for s in SIDES}
        self.obstacles = [self.m.geom(n).id for n in ("table", "left_mount", "right_mount")]
        self.self_pairs = {}
        for side, geoms in self.geoms.items():
            pairs = []
            for k, first in enumerate(geoms):
                for second in geoms[k+1:]:
                    a, b = [int(self.m.body_weldid[int(self.m.geom_bodyid[g])])
                            for g in (first, second)]
                    if all(self.body_name(g).endswith("_finger") for g in (first, second)):
                        continue
                    if a == b or int(self.m.body_weldid[int(self.m.body_parentid[a])]) == b \
                            or int(self.m.body_weldid[int(self.m.body_parentid[b])]) == a:
                        continue
                    pairs.append((first, second))
            self.self_pairs[side] = pairs
            self.d.qpos[self.qindices[side]] = self.home
        self.set_fingers(0.04)
        mujoco.mj_forward(self.m, self.d)

    def body_name(self, geom):
        return self.m.body(int(self.m.geom_bodyid[geom])).name

    def set_fingers(self, opening):
        """Set each finger joint displacement (half the nominal jaw opening)."""
        for side in SIDES:
            value = opening[side] if isinstance(opening, dict) else opening
            for finger in (1, 2):
                self.d.qpos[self.m.joint(f"{side}_finger_joint{finger}").qposadr[0]] = value

    def fk(self, side, q):
        self.d.qpos[self.qindices[side]] = np.asarray(q)
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        body = self.bodies[side]
        matrix = self.d.xmat[body].reshape(3, 3)
        return self.d.xpos[body].copy() + matrix @ TCP, Rotation.from_matrix(matrix)

    def solve(self, side, p, r=DOWN, seed=None, *, max_nfev=180):
        """Return (seven joint angles, TCP error in m, rotation error in rad)."""
        p = np.asarray(p, dtype=float)
        seed = self.home.copy() if seed is None else np.asarray(seed, dtype=float)
        if not isinstance(r, Rotation):
            raise TypeError("r must be scipy.spatial.transform.Rotation")

        def residual(q):
            fp, fr = self.fk(side, q)
            return np.r_[5.0 * (fp-p), (r*fr.inv()).as_rotvec(), 0.0002*(q-seed)]

        limits = self.limits[side]
        solution = least_squares(residual,
            np.clip(seed, limits[:, 0]+IK_JOINT_MARGIN, limits[:, 1]-IK_JOINT_MARGIN),
            bounds=(limits[:, 0]+IK_JOINT_MARGIN, limits[:, 1]-IK_JOINT_MARGIN),
            ftol=1e-10, gtol=1e-9, xtol=1e-10, max_nfev=max_nfev)
        fp, fr = self.fk(side, solution.x)
        return solution.x.copy(), float(np.linalg.norm(fp-p)), float((r*fr.inv()).magnitude())

    def clearance(self, qs, finger_opening=None):
        """Return minimum signed distances; negative means mesh penetration."""
        for side, q in qs.items():
            self.d.qpos[self.qindices[side]] = q
        if finger_opening is not None:
            self.set_fingers(finger_opening)
        mujoco.mj_forward(self.m, self.d)

        def minimum(pairs):
            distance, first, second = min((float(mujoco.mj_geomDistance(
                self.m, self.d, first, second, 0.20, None)), first, second)
                for first, second in pairs)
            return {"distance": distance, "bodies": [self.body_name(first), self.body_name(second)],
                    "geoms": [self.m.geom(first).name, self.m.geom(second).name]}

        out = {}
        for side in qs:
            out[side + "_self"] = minimum(self.self_pairs[side])
            out[side + "_table"] = minimum((g, self.obstacles[0]) for g in self.geoms[side])
            out[side + "_mounts"] = minimum((g, obstacle) for g in self.geoms[side]
                                               for obstacle in self.obstacles[1:])
        out["arms"] = minimum((a, b) for a in self.geoms["left"] for b in self.geoms["right"])
        return out

    def plan_cartesian(self, start_qs, targets, *, rotations=None, samples=30,
                       position_tolerance=0.0005, angle_tolerance=0.002):
        """Sequential warm-start IK for straight Cartesian paths with Slerp.

        Return samples including endpoints: {fraction, q, tcp, position_error,
        angle_error}. Does not silently accept unreachable targets. Collision
        checks are separate because the caller knows allowed grasp contacts.
        """
        if samples < 2:
            raise ValueError("samples must be at least two")
        seeds = {s: np.asarray(q, dtype=float).copy() for s, q in start_qs.items()}
        start = {s: self.fk(s, seeds[s]) for s in targets}
        rotations = {s: DOWN for s in targets} if rotations is None else rotations
        interpolators = {s: Slerp([0, 1], Rotation.from_quat(
            [start[s][1].as_quat(), rotations[s].as_quat()])) for s in targets}
        result = []
        for fraction in np.linspace(0, 1, samples):
            entry = {"fraction": float(fraction), "q": {}, "tcp": {},
                     "position_error": {}, "angle_error": {}}
            for side, target in targets.items():
                pos = (1-fraction)*start[side][0] + fraction*np.asarray(target)
                q, error, angle = self.solve(side, pos, interpolators[side](fraction), seeds[side])
                if error > position_tolerance or angle > angle_tolerance:
                    raise ValueError(f"Unreachable {side} Cartesian sample {fraction:.3f}: "
                                     f"position={error:.6f}m, rotation={angle:.6f}rad")
                seeds[side] = q
                entry["q"][side] = q.copy()
                entry["tcp"][side] = pos.copy()
                entry["position_error"][side] = error
                entry["angle_error"][side] = angle
            result.append(entry)
        return result

    def trajectory(self, start_qs, targets, *, rotations=None, samples=45,
                   duration=2.0, finger_start=None, finger_end=None,
                   max_joint_velocity=.55, max_joint_acceleration=1.0,
                   clearance_margin=.008, validate_samples=101):
        """Create and validate a C2 Cartesian-derived joint trajectory.

        The idle arm stays at its supplied start pose. Dynamic object contacts
        are handled by the simulation, not this rigid geometric preflight.
        """
        paths = self.plan_cartesian(start_qs, targets, rotations=rotations,
                                    samples=samples)
        fractions = np.asarray([entry["fraction"] for entry in paths])
        curves = {}
        for side in SIDES:
            values = np.asarray([entry["q"].get(side, start_qs[side]) for entry in paths])
            curves[side] = CubicSpline(fractions, values, axis=0)
        plan = MotionPlan(curves, float(duration), finger_start, finger_end)
        # Sampling can miss small spline overshoots between valid IK knots.
        # Minimum-jerk time scaling traverses every path fraction monotonically,
        # so checking each cubic's stationary points covers the full motion.
        plan.validate_joint_limits(self.limits)
        plan.limit_rate(max_joint_velocity, max_joint_acceleration)
        for time in np.linspace(0, plan.duration, validate_samples):
            point = plan.sample(float(time))
            for side in SIDES:
                q = point["q"][side]
                limits = self.limits[side]
                if np.any(q < limits[:, 0]) or np.any(q > limits[:, 1]):
                    raise ValueError(f"Joint limit violation in {side} trajectory at {time:.3f}s")
            checks = self.clearance(point["q"], point["finger"])
            failed = {name: value for name, value in checks.items()
                      if value["distance"] < clearance_margin}
            if failed:
                raise ValueError(f"Rigid clearance violation at {time:.3f}s: {failed}")
        return plan


def minimum_jerk(fraction):
    """Return progress and first/second derivatives with respect to fraction."""
    u = np.clip(np.asarray(fraction, dtype=float), 0., 1.)
    return (10*u**3 - 15*u**4 + 6*u**5,
            30*u**2 - 60*u**3 + 30*u**4,
            60*u - 180*u**2 + 120*u**3)


def _positive_finite_scalar(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) \
            or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite numeric scalar")
    return float(value)


def _real_polynomial_roots_unit(coefficients):
    """Isolate real roots on [0,1] using the derivative's monotone intervals.

    Recursion avoids discarding repeated real roots as nearly complex companion
    roots. Near-zero stationary values are retained as extra candidates; they
    cannot reduce an extremum. This is floating-point root isolation, not
    directed-rounding interval arithmetic.
    """
    raw = np.asarray(coefficients)
    if raw.ndim != 1 or np.iscomplexobj(raw):
        raise ValueError("Rate root coefficients must be a real vector")
    c = np.trim_zeros(np.asarray(raw, float), "b")
    if not np.isfinite(c).all():
        raise ValueError("Nonfinite rate polynomial")
    if len(c) <= 1:
        return []
    c = c / np.max(np.abs(c))
    if len(c) == 2:
        root = -c[0] / c[1]
        return [float(root)] if np.isfinite(root) and 0. <= root <= 1. else []
    knots = sorted(set([0., 1., *_real_polynomial_roots_unit(polynomial.polyder(c))]))
    values = np.asarray(polynomial.polyval(knots, c))
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite rate root evaluation")
    zero_tolerance = 64*np.finfo(float).eps*np.sum(np.abs(c))
    roots = [x for x, y in zip(knots, values) if abs(y) <= zero_tolerance]
    for lo, hi, flo, fhi in zip(knots[:-1], knots[1:], values[:-1], values[1:]):
        if flo != 0. and fhi != 0. and np.signbit(flo) != np.signbit(fhi):
            roots.append(float(brentq(lambda x: polynomial.polyval(x, c), lo, hi,
                                      xtol=5e-15, rtol=4*np.finfo(float).eps)))
    return sorted(set(roots))


@dataclass
class MotionPlan:
    """A C2 path with zero velocity/acceleration at both ends.

    Sampled position + velocity are targets for physical joint PD controllers;
    no kinematic teleports or software attachments are performed here.
    Finger values are the displacement of EACH finger, in [0, .04] metres.
    """
    curves: dict
    duration: float
    finger_start: dict | None = None
    finger_end: dict | None = None

    def __post_init__(self):
        self.duration = _positive_finite_scalar(self.duration, "Duration")
        if self.finger_start is None:
            self.finger_start = dict.fromkeys(SIDES, .04)
        if self.finger_end is None:
            self.finger_end = self.finger_start.copy()
        for values in (self.finger_start, self.finger_end):
            if set(values) != set(SIDES) or any(not 0 <= values[s] <= .04 for s in SIDES):
                raise ValueError("Finger values must provide left/right in [0, .04] m")

    def sample(self, time):
        u = float(np.clip(time / self.duration, 0., 1.))
        s, su, suu = minimum_jerk(u)
        sd, sdd = su/self.duration, suu/self.duration**2
        out = {key: {} for key in ("q", "qd", "qdd", "finger", "finger_velocity")}
        for side, curve in self.curves.items():
            out["q"][side] = curve(s)
            out["qd"][side] = curve(s, 1)*sd
            out["qdd"][side] = curve(s, 2)*sd**2 + curve(s, 1)*sdd
            start, finish = self.finger_start[side], self.finger_end[side]
            out["finger"][side] = float(start + (finish-start)*s)
            out["finger_velocity"][side] = float((finish-start)*sd)
        return out

    def joint_extrema(self):
        """Exact extrema of every cubic joint path, including its endpoints.

        Values are with respect to spline path fraction, before time scaling.
        No samples, clipping, or limit tolerances are used to accept a path.
        """
        result = {}
        for side, curve in self.curves.items():
            count = curve.c.shape[-1]
            minima, maxima, minimum_at, maximum_at = [], [], [], []
            for joint in range(count):
                locations = list(map(float, curve.x))
                for segment, start in enumerate(curve.x[:-1]):
                    width = float(curve.x[segment+1]-start)
                    a, b, c = (float(v) for v in curve.c[:3, segment, joint])
                    # np.roots also handles an exactly linear/constant
                    # derivative by removing leading zero coefficients.
                    for root in np.roots([3*a, 2*b, c]):
                        if abs(float(root.imag)) <= 1e-12:
                            local = float(root.real)
                            if 0.0 < local < width:
                                locations.append(float(start)+local)
                locations = np.asarray(locations)
                if not np.all(np.isfinite(locations)):
                    raise ValueError(f"Non-finite {side} joint{joint+1} spline extrema")
                values = np.asarray(curve(locations))[:, joint]
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"Non-finite {side} joint{joint+1} spline values")
                low, high = int(np.argmin(values)), int(np.argmax(values))
                minima.append(float(values[low])); maxima.append(float(values[high]))
                minimum_at.append(float(locations[low])); maximum_at.append(float(locations[high]))
            result[side] = {"minimum": np.asarray(minima), "maximum": np.asarray(maxima),
                            "minimum_at_fraction": np.asarray(minimum_at),
                            "maximum_at_fraction": np.asarray(maximum_at)}
        return result

    def validate_joint_limits(self, limits):
        """Fail closed if any continuous spline extremum exceeds a limit."""
        extrema = self.joint_extrema()
        for side, values in extrema.items():
            bounds = np.asarray(limits[side])
            for key, sign, column in (("minimum", -1, 0), ("maximum", 1, 1)):
                violations = sign*(values[key]-bounds[:, column]) > 0
                for joint in np.flatnonzero(violations):
                    fraction = values[key+"_at_fraction"][joint]
                    raise ValueError(
                        f"Continuous joint limit violation in {side} joint{joint+1} "
                        f"at path fraction {fraction:.9f}: {key}={values[key][joint]:.12f}, "
                        f"limit={bounds[joint, column]:.12f}")
        return extrema

    def continuous_rate_extrema(self):
        """Per-joint extrema of q(minimum_jerk(t/T)) over every cubic segment.

        Each clock interval is normalized to [0,1] before composition. Its
        degree <=15 position polynomial is differentiated, and velocity and
        acceleration are evaluated at both endpoints and all real stationary
        roots. Both sides of every spline knot are included. No time grid is
        used to accept a peak; the numerical margin is applied by limit_rate.
        """
        duration = _positive_finite_scalar(self.duration, "Duration")
        if not isinstance(self.curves, dict) or set(self.curves) != set(SIDES):
            raise ValueError("Rate limits require both arm cubic paths")

        def stable_progress(u):
            v = u if u <= .5 else 1.-u
            value = v*v*v*(10.-15.*v+6.*v*v)
            return value if u <= .5 else 1.-value

        def clock_at(s):
            if s == 0. or s == 1.:
                return s
            target = s if s <= .5 else 1.-s
            u = brentq(lambda x: stable_progress(x)-target, 0., .5,
                       xtol=5e-15, rtol=4*np.finfo(float).eps)
            return u if s <= .5 else 1.-u

        result = {}
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                for side in SIDES:
                    curve = self.curves[side]
                    if np.iscomplexobj(getattr(curve, "x", [])) or np.iscomplexobj(getattr(curve, "c", [])):
                        raise ValueError("Joint cubic knots and coefficients must be real")
                    x = np.asarray(getattr(curve, "x", []), float)
                    c = np.asarray(getattr(curve, "c", []), float)
                    if x.ndim != 1 or len(x) < 2 or c.shape != (4, len(x)-1, 7) \
                            or not np.isfinite(x).all() or not np.isfinite(c).all() \
                            or x[0] != 0. or x[-1] != 1. or np.any(np.diff(x) <= 0.):
                        raise ValueError("Finite seven-joint cubics must cover [0,1] with increasing knots")
                    # Discontinuous q/q_s/q_ss cannot be repaired by retiming.
                    for k, width in enumerate(np.diff(x)[:-1]):
                        for order in range(3):
                            local = c[:, k, :].copy()
                            for _ in range(order):
                                local = local[:-1]*np.arange(len(local)-1, 0, -1)[:, None]
                            end = np.zeros(7)
                            for coefficient in local:
                                end = end*width+coefficient
                            start = c[3-order, k+1]*math.factorial(order)
                            # Short IK intervals amplify roundoff in q_ss;
                            # both one-sided acceleration extrema are still
                            # included below. Position/velocity stay tighter.
                            atol = 1e-8 if order == 2 else 1e-10
                            if not np.allclose(end, start, rtol=1e-10, atol=atol):
                                raise ValueError(f"{side} rate path is not C2 at knot {k+1}")
                    clocks = np.asarray([clock_at(float(s)) for s in x])
                    if not np.isfinite(clocks).all() or np.any(np.diff(clocks) <= 0.):
                        raise ValueError("Spline clock intervals are below numerical resolution")
                    peaks = np.zeros((2, 7)); times = np.zeros((2, 7))
                    for k, (ua, ub) in enumerate(zip(clocks[:-1], clocks[1:])):
                        h = ub-ua
                        # Taylor expansion of s(ua+h*r)-x[k], using factored
                        # low-order coefficients instead of cancellation near 1.
                        progress = np.array([stable_progress(ua)-x[k],
                            30*ua*ua*(1-ua)**2*h,
                            30*ua*(1-ua)*(1-2*ua)*h*h,
                            (10-60*ua+60*ua*ua)*h**3,
                            (-15+30*ua)*h**4, 6*h**5])
                        q = np.zeros((16, 7))
                        for degree in range(4):
                            basis = polynomial.polypow(progress, degree)
                            q[:len(basis)] += basis[:, None]*c[3-degree, k]
                        qd = polynomial.polyder(q, axis=0)/(h*duration)
                        qdd = polynomial.polyder(qd, axis=0)/(h*duration)
                        qddd = polynomial.polyder(qdd, axis=0)/(h*duration)
                        if not all(np.isfinite(a).all() for a in (q, qd, qdd, qddd)):
                            raise ValueError("Nonfinite composed rate polynomial")
                        for order, (values, stationary) in enumerate(((qd, qdd), (qdd, qddd))):
                            for joint in range(7):
                                candidates = np.asarray([0., 1., *_real_polynomial_roots_unit(stationary[:, joint])])
                                rates = np.abs(polynomial.polyval(candidates, values[:, joint]))
                                if not np.isfinite(rates).all():
                                    raise ValueError("Nonfinite continuous rate extrema")
                                index = int(np.argmax(rates))
                                if rates[index] > peaks[order, joint]:
                                    peaks[order, joint] = rates[index]
                                    times[order, joint] = (ua+h*candidates[index])*duration
                    result[side] = dict(maximum_velocity_rad_s=peaks[0],
                        maximum_acceleration_rad_s2=peaks[1],
                        velocity_peak_time_s=times[0], acceleration_peak_time_s=times[1])
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError("Continuous rate calculation exceeded finite numerical range") from exc
        return result

    def limit_rate(self, max_joint_velocity=.55, max_joint_acceleration=1.0):
        max_joint_velocity = _positive_finite_scalar(max_joint_velocity, "Velocity limit")
        max_joint_acceleration = _positive_finite_scalar(max_joint_acceleration, "Acceleration limit")
        extrema = self.continuous_rate_extrema()
        peak_v = max(float(np.max(v["maximum_velocity_rad_s"])) for v in extrema.values())
        peak_a = max(float(np.max(v["maximum_acceleration_rad_s2"])) for v in extrema.values())
        # Keep the original 0.2% duration margin. Velocity scales as 1/T and
        # acceleration as 1/T². This margin covers floating evaluation/root
        # errors; it is not a directed-rounding interval proof.
        factor = max(1., peak_v/max_joint_velocity, math.sqrt(peak_a/max_joint_acceleration))*1.002
        new_duration = _positive_finite_scalar(self.duration*factor, "Retimed duration")
        self.duration = new_duration
        return self.duration

    def end_q(self):
        return self.sample(self.duration)["q"]


def hold_trajectory(qs, duration, finger_start=None, finger_end=None):
    """Keep both arms still while smoothly moving the real grippers."""
    return MotionPlan({s: CubicSpline([0., 1.], [qs[s], qs[s]], axis=0)
                       for s in SIDES}, duration, finger_start, finger_end)


def command_pd(robots, dof_indices, sample):
    """Issue nine-DOF position/velocity targets to two Genesis rigid robots.

    Caller configures finite torque/force limits and steps the physics solver.
    Initialization may set q once before the first step; motion must use this
    controller. dof_indices[side] is ordered joint1..joint7,finger1,finger2.
    """
    for side in SIDES:
        q = np.r_[sample["q"][side], [sample["finger"][side]]*2]
        qd = np.r_[sample["qd"][side], [sample["finger_velocity"][side]]*2]
        robots[side].control_dofs_position_velocity(q, qd, dof_indices[side])


def nominal_waypoints(config=None, metadata=None):
    """Reference geometry only: replace grasp points with measured soft meshes.

    The left hand moves outward after opening to leave a clear insertion lane.
    The teddy is held by its narrow upper arm; it is expected to hang under
    gravity, not remain horizontal. The runtime must measure this before entry.
    """
    if config is None:
        config = json.loads((HERE / "config.json").read_text()) if (HERE / "config.json").exists() else {}
    bag_config = config.get("bag", {})
    bag_origin = np.asarray(bag_config.get("position", [-.13, 0., .742]), dtype=float)
    bear_origin = np.asarray(config.get("bear", {}).get("position", [.27, -.25, .742]), dtype=float)
    metadata_path = HERE / "assets" / bag_config.get("metadata", "metadata.json")
    if metadata is None and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    metadata = metadata or {}
    handle_centers = metadata.get("bag", {}).get("grasp_centers_local_m", {
        "grasp_front": [0., -.11, .4791813982276508],
        "grasp_back": [0., .11, .4791813982276508]})
    # The task may select a more proximal grasp than the asset's generic arm
    # group centroid. Match the runtime's reference material point exactly.
    teddy_center = config.get("task", {}).get("bear_grasp_local",
        metadata.get("bear", {}).get("grasp_arm_right_local_m",
            [.10238069672434515, -.010474149813803483, .033970580749295064]))
    teddy_center = np.asarray(teddy_center, dtype=float)
    if teddy_center.shape != (3,) or not np.isfinite(teddy_center).all():
        raise ValueError("task.bear_grasp_local must be one finite local XYZ point")
    teddy_grasp = bear_origin + teddy_center
    return {
        "initial": {"left": (bag_origin + handle_centers["grasp_front"]).tolist(),
                    "right": (bag_origin + handle_centers["grasp_back"]).tolist()},
        "open_bag": config.get("task", {}).get("open_targets",
            config.get("robots", {}).get("open_bag_tcp_targets", {
                "left": [-.25, -.13, 1.25], "right": [-.10, .13, 1.25]})),
        "right_clear_handle": {"right": [.12, .20, 1.32]},
        "teddy_pregrasp": {"right": (teddy_grasp + [0., 0., .15]).tolist()},
        "teddy_grasp": {"right": teddy_grasp.tolist()},
        "teddy_lift": {"right": [*teddy_grasp[:2], 1.37]},
        "bag_above": {"right": [-.10, .04, 1.37]},
        "bag_insert": {"right": [-.10, .04, 1.14]},
        "release": {"right": [-.10, .04, .96]},
        "right_retreat": {"right": [-.10, .04, 1.38]},
    }
