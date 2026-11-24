"""Read-only, per-step IPC soft-contact wrench guard for the local Panda model.

This consumes feedback already computed by the scene-local exact STC adapter.
It does not initialize Genesis, fetch GPU tensors, alter forces, or move state.
These are engineering trial limits, not experimentally calibrated safety limits.
Genesis rigid-contact forces are NOT part of the reported IPC wrench.
"""
from __future__ import annotations

from importlib.metadata import version
import copy
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np


def wrench_at_point(forces, torques_at_origins, origins, point):
    """Sum world wrenches, shifting each torque from its origin to point."""
    forces, torques, origins = map(np.asarray, (forces, torques_at_origins, origins))
    point = np.asarray(point)
    if forces.ndim != 2 or forces.shape[1] != 3 or torques.shape != forces.shape or origins.shape != forces.shape or point.shape != (3,):
        raise ValueError("Wrench arrays must have matching (n,3) shape and a (3,) reference")
    if not all(np.isfinite(v).all() for v in (forces, torques, origins, point)):
        raise ValueError("Nonfinite cached IPC wrench")
    return forces.sum(axis=0), (torques + np.cross(origins-point, forces)).sum(axis=0)


def _quat_matrix(wxyz):
    q = np.asarray(wxyz, float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) < 1e-12:
        raise ValueError("Invalid MJCF quaternion")
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def _model_mapping(path):
    """Reject changed fixed-chain/joint-axis assumptions; derive TCP from XML."""
    path = Path(path).resolve()
    root = ET.parse(path).getroot()
    if root.get("model") != "panda_tactile":
        raise ValueError("Contact guard is restricted to the reviewed panda_tactile model")
    link7 = root.find(".//body[@name='link7']")
    hand = None if link7 is None else link7.find("./body[@name='hand']")
    if hand is None or hand.find("joint") is not None or hand.find("freejoint") is not None:
        raise ValueError("Expected a fixed hand directly below link7")
    if any(k in hand.attrib for k in ("euler", "axisangle", "xyaxes", "zaxis")):
        raise ValueError("Unsupported hand orientation encoding")
    hand_pos = np.fromstring(hand.get("pos", "0 0 0"), sep=" ")
    hand_rot = _quat_matrix(np.fromstring(hand.get("quat", "1 0 0 0"), sep=" "))
    if hand_pos.shape != (3,):
        raise ValueError("Invalid fixed hand translation")
    finger_default = root.find(".//default[@class='finger']/joint")
    if finger_default is None or finger_default.get("type") != "slide" or not np.allclose(np.fromstring(finger_default.get("axis", ""), sep=" "), [0,1,0]):
        raise ValueError("Expected +local-Y prismatic finger joints")
    expected = {}
    for i, name in enumerate(("left_finger", "right_finger"), 1):
        body = hand.find(f"./body[@name='{name}']")
        joint = None if body is None else body.find("joint")
        if joint is None or joint.get("name") != f"finger_joint{i}" or joint.get("class") != "finger":
            raise ValueError("Unexpected finger joint mapping")
        if any(k in joint.attrib for k in ("axis", "type")):
            raise ValueError("Unexpected override of reviewed finger joint axis/type")
        pos = np.fromstring(body.get("pos", "0 0 0"), sep=" ")
        rot = _quat_matrix(np.fromstring(body.get("quat", "1 0 0 0"), sep=" "))
        expected[name] = (hand_pos+hand_rot@pos, hand_rot@rot)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "tcp_in_link7_m": (hand_pos+hand_rot@np.array([0,0,.1029])).tolist(),
            "hand_position_in_link7_m": hand_pos.tolist()}, expected


class ContactGuard:
    """Construct after scene build; set baseline at insertion-trial entry.

    set_baseline(time) returns the exact stored baseline record.
    sample(time) returns (JSON-serializable row, reason_or_None). The caller
    saves the row before stopping on a nonempty reason. Mapping/version or
    nonfinite-data errors raise RuntimeError/ValueError, never silently pass.
    Repeated sampling of the same step/time is allowed; a new time with a stale
    feedback step is rejected. No baseline subtraction can exempt raw limits.
    """
    def __init__(self, scene, *, max_net_force_n=8., max_tcp_torque_nm=.15,
                 max_finger_force_n=8., require_left_soft_contact=True,
                 minimum_left_opening_reaction_n=.1):
        found = {name: version(name) for name in ("genesis-world", "pyuipc")}
        if found != {"genesis-world":"1.4.3", "pyuipc":"0.0.29"}:
            raise RuntimeError(f"Unsupported IPC guard versions: {found}")
        self.scene = scene
        self.coupler = scene.sim.coupler
        if type(self.coupler).__name__ != "IPCCoupler" or scene.sim.substeps != 1:
            raise RuntimeError("Guard requires IPCCoupler and exactly one physics substep")
        if not scene.teddy_stc_feedback_report.get("enabled") or getattr(self.coupler._apply_abd_coupling_forces, "__name__", "") != "exact_feedback":
            raise RuntimeError("Reviewed exact STC feedback adapter is not active")
        if self.coupler._ipc_animator.substep() != 1:
            raise RuntimeError("Guard requires native animator substep=1")
        self.data = self.coupler._coupling_data
        if self.data is None:
            raise RuntimeError("Build the scene before constructing ContactGuard")
        self.indices = {}
        for entity, names in (("right_franka", ("link7","left_finger","right_finger")),
                              ("left_franka", ("left_finger","right_finger"))):
            for name in names:
                matches = [i for i, link in enumerate(self.data.links) if link.entity.name == entity and link.name == name]
                if len(matches) != 1:
                    raise RuntimeError(f"Missing or ambiguous coupling mapping: {entity}/{name}")
                self.indices[(entity,name)] = matches[0]
        right_link = self.data.links[self.indices[("right_franka","link7")]]
        if any(link.entity is right_link.entity and link.name == "hand" for link in self.data.links):
            raise RuntimeError("Separate hand ABD detected; reviewed link7 mapping is no longer valid")
        if right_link.entity.material.coup_type != "two_way_soft_constraint":
            raise RuntimeError("Right arm does not use the reviewed two-way STC coupling")
        self.model, self.expected_fingers = _model_mapping(right_link.entity.morph.file)
        left_entity = self.data.links[self.indices[("left_franka","left_finger")]].entity
        if left_entity.material.coup_type != "two_way_soft_constraint" or hashlib.sha256(Path(left_entity.morph.file).resolve().read_bytes()).hexdigest() != self.model["sha256"]:
            raise RuntimeError("Left finger coupling/model differs from the reviewed right-arm model")
        self.tcp_local = np.asarray(self.model["tcp_in_link7_m"])
        self.right = np.array([self.indices[("right_franka", n)] for n in ("link7","left_finger","right_finger")])
        self.left = np.array([self.indices[("left_franka", n)] for n in ("left_finger","right_finger")])
        # Keep all left-arm cached reactions for attribution after a trial.
        # These fields add observations only; guard limits remain unchanged.
        self.left_all = np.array([i for i, link in enumerate(self.data.links)
                                  if link.entity is left_entity], dtype=int)
        self.left_all_names = [self.data.links[i].name for i in self.left_all]
        if len(set(self.left_all_names)) != len(self.left_all_names) or "link7" not in self.left_all_names:
            raise RuntimeError("Ambiguous or incomplete left-arm diagnostic mapping")
        self.left_palm_index = int(self.left_all[self.left_all_names.index("link7")])
        self.limits = {"max_net_force_n": float(max_net_force_n), "max_tcp_torque_nm": float(max_tcp_torque_nm),
                       "max_finger_force_n": float(max_finger_force_n),
                       "minimum_left_opening_reaction_n": float(minimum_left_opening_reaction_n)}
        if any(not np.isfinite(x) or x <= 0 for x in self.limits.values()):
            raise ValueError("Contact guard thresholds must be finite and positive")
        self.require_left_soft_contact = bool(require_left_soft_contact)
        self.baseline = None
        self._last_step = None
        self._last_time = None
        self._validate_pose_mapping()

    def _validate_pose_mapping(self):
        aim = self.data.aim_transforms[0]
        root = aim[self.right[0]]
        for name, index in zip(("left_finger","right_finger"), self.right[1:]):
            relative_r = root[:3,:3].T @ aim[index,:3,:3]
            relative_p = root[:3,:3].T @ (aim[index,:3,3]-root[:3,3])
            fixed_p, expected_r = self.expected_fingers[name]
            delta = relative_p-fixed_p
            axis = expected_r[:,1]
            q = float(delta@axis)
            if not np.allclose(relative_r,expected_r,atol=1e-6,rtol=0) or np.linalg.norm(delta-q*axis)>1e-6 or not -.005 <= q <= .060:
                raise RuntimeError(f"Actual {name} aim transform disagrees with reviewed XML mapping")

    def _read(self, time):
        time = float(time)
        step = int(self.scene.sim.cur_step_global)
        if not np.isfinite(time) or time < 0 or step <= 0:
            raise RuntimeError("Guard needs a finite time after at least one completed physics step")
        if self._last_step is not None and (step < self._last_step or time < self._last_time-1e-10 or (step == self._last_step and abs(time-self._last_time)>1e-10)):
            raise RuntimeError("Stale or reversed IPC feedback sampling")
        if self.scene.sim.coupler._coupling_data is not self.data:
            raise RuntimeError("Coupling buffers changed after guard construction")
        self._validate_pose_mapping()
        aim = self.data.aim_transforms[0]
        f = self.data.out_forces[0,self.right].copy()
        torque = self.data.out_torques[0,self.right].copy()
        root = aim[self.right[0]]
        tcp = root[:3,3] + root[:3,:3] @ self.tcp_local
        force, moment = wrench_at_point(f, torque, aim[self.right,:3,3], tcp)
        right_moments_at_tcp = torque + np.cross(aim[self.right,:3,3]-tcp, f)
        left_f = self.data.out_forces[0,self.left].copy()
        left_all_f = self.data.out_forces[0,self.left_all].copy()
        left_all_torque = self.data.out_torques[0,self.left_all].copy()
        left_root = aim[self.left_palm_index]
        left_tcp = left_root[:3,3] + left_root[:3,:3] @ self.tcp_local
        left_force, left_moment = wrench_at_point(left_all_f, left_all_torque,
                                                aim[self.left_all,:3,3], left_tcp)
        left_moments_at_tcp = left_all_torque + np.cross(aim[self.left_all,:3,3]-left_tcp, left_all_f)
        right_open = np.einsum("ij,ij->i", f[1:], aim[self.right[1:],:3,1])
        left_open = np.einsum("ij,ij->i", left_f, aim[self.left,:3,1])
        if not np.isfinite(np.r_[left_f.ravel(),right_open,left_open]).all():
            raise RuntimeError("Nonfinite finger IPC reaction")
        self._last_step, self._last_time = step, time
        return {"time_s":time, "physics_step":step, "force_world_n":force.tolist(),
                "torque_at_aim_tcp_world_nm":moment.tolist(), "aim_tcp_world_m":tcp.tolist(),
                "net_force_norm_n":float(np.linalg.norm(force)), "tcp_torque_norm_nm":float(np.linalg.norm(moment)),
                "right_link_names":["link7","left_finger","right_finger"],
                "right_link_force_world_n":f.tolist(), "right_link_torque_at_aim_origin_world_nm":torque.tolist(),
                "right_link_aim_origin_world_m":aim[self.right,:3,3].tolist(),
                "right_link_torque_at_aim_tcp_world_nm":right_moments_at_tcp.tolist(),
                "right_finger_aim_opening_axis_world":aim[self.right[1:],:3,1].tolist(),
                "left_link_names":self.left_all_names.copy(),
                "left_link_force_world_n":left_all_f.tolist(),
                "left_link_torque_at_aim_origin_world_nm":left_all_torque.tolist(),
                "left_link_aim_origin_world_m":aim[self.left_all,:3,3].tolist(),
                "left_link_torque_at_left_aim_tcp_world_nm":left_moments_at_tcp.tolist(),
                "left_aim_tcp_world_m":left_tcp.tolist(),
                "left_all_link_net_force_world_n":left_force.tolist(),
                "left_all_link_torque_at_aim_tcp_world_nm":left_moment.tolist(),
                "left_finger_aim_opening_axis_world":aim[self.left,:3,1].tolist(),
                "right_finger_opening_projection_n":right_open.tolist(),
                "right_finger_force_norm_n":np.linalg.norm(f[1:],axis=1).tolist(),
                "left_finger_opening_projection_n":left_open.tolist(),
                "left_finger_force_norm_n":np.linalg.norm(left_f,axis=1).tolist()}

    def set_baseline(self, time):
        self.baseline = self._read(time)
        return copy.deepcopy(self.baseline)

    def sample(self, time):
        if self.baseline is None:
            raise RuntimeError("Set and save the contact-trial baseline before sampling")
        row = self._read(time)
        row["baseline_time_s"] = self.baseline["time_s"]
        for key in ("force_world_n", "torque_at_aim_tcp_world_nm", "right_finger_opening_projection_n", "right_finger_force_norm_n", "left_finger_opening_projection_n", "left_finger_force_norm_n"):
            row["delta_"+key] = (np.asarray(row[key])-np.asarray(self.baseline[key])).tolist()
        reasons = []
        if row["net_force_norm_n"] > self.limits["max_net_force_n"]:
            reasons.append("IPC distal-hand net force exceeded engineering trial limit")
        if row["tcp_torque_norm_nm"] > self.limits["max_tcp_torque_nm"]:
            reasons.append("IPC distal-hand torque at TCP exceeded engineering trial limit")
        if max(row["right_finger_force_norm_n"]) > self.limits["max_finger_force_n"]:
            reasons.append("Individual right finger IPC force exceeded engineering trial limit")
        if self.require_left_soft_contact and min(row["left_finger_opening_projection_n"]) <= self.limits["minimum_left_opening_reaction_n"]:
            reasons.append("Left fingers lack two-sided IPC soft-contact opening reaction")
        reason = "; ".join(reasons) or None
        row["stop_reason"] = reason
        return row, reason

    def report(self):
        return {"scope":"Exact STC feedback from native IPC soft contacts only; excludes Genesis rigid-contact forces",
                "force_links":"right link7 (includes welded palm) plus both fingers",
                "torque_reference":"aim TCP in world axes; link-origin moments shifted by cross(origin-TCP,F)",
                "model":self.model, "limits":self.limits, "experimentally_calibrated":False,
                "left_soft_contact_required":self.require_left_soft_contact, "baseline":copy.deepcopy(self.baseline),
                "diagnostics":"Per-link right aim origins and shifted moments; all left-arm cached raw wrenches/origins and moments about the left aim TCP. No added force/torque thresholds.",
                "limitations":["Opening reactions do not identify the contacted soft material; the named handle material-group proximity guard is also required.",
                               "Net force can cancel opposing loads; per-finger norms, torque, soft strain, grip, and rigid-collision guards remain necessary.",
                               "A baseline is recorded for diagnosis; thresholds always apply to raw values."]}
