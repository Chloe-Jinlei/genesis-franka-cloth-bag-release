"""Native final-stage experiment from an explicitly approximate saved-state restart.

The controller waits between commands without advancing time. A failed motion
pauses the live process; it does not throw away the native solver state.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import time
import traceback

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
import numpy as np
from PIL import Image
import imageio.v2 as imageio
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import scene as scene_module
from motion import Planner, TCP, command_pd, hold_trajectory
from physical_monitor import PhysicalMonitor
from validate import mouth_structure, containment, signed_volumes, final_contact_audit
from contact_guard import ContactGuard
from isolated_stage_restart import load_saved_frame, native_restart_context, sync_genesis_soft_views


def serial(x):
    if isinstance(x, dict): return {k: serial(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [serial(v) for v in x]
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    return x


def write_json(path, obj):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(serial(obj), indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


class LiveStage:
    def __init__(self, args):
        self.out = Path(args.output).resolve()
        self.out.mkdir(exist_ok=False)
        self.commands = self.out / "commands"
        self.commands.mkdir()
        self.config = json.loads((Path(args.source) / "config.json").read_text())
        if args.native_checkpoint_source:
            from native_checkpoint_restart import load_native_checkpoint, native_checkpoint_context, sync_current_native_views
            self.payload = load_native_checkpoint(Path(args.native_checkpoint_source))
            restart_context = native_checkpoint_context
        else:
            self.payload = load_saved_frame(Path(args.source), args.time)
            restart_context = native_restart_context
        self.rest = self.payload["rest"]
        self.qinitial = {s: self.payload["q_" + s] for s in ("left", "right")}
        with restart_context(self.payload) as restart:
            self.scene, self.robots, self.bag, self.bear, self.cameras = scene_module.build(
                self.config, initial_q=self.qinitial, width=960, height=720)
        verification = sync_current_native_views(self.scene,self.payload) if args.native_checkpoint_source else sync_genesis_soft_views(self.scene,self.payload)
        restart["native_state_verification"] = verification
        self.restart = restart
        self.planner = Planner(self.config)
        self.qcmd = {s: np.asarray(self.payload.get("command_"+s,self.qinitial[s][:7])).copy() for s in self.robots}
        self.fingers = {"left": float(self.payload.get("finger_left",.001)), "right": float(self.payload.get("finger_right",self.config["task"]["bear_finger_target"]))}
        self.dofs = {s: np.arange(9) for s in self.robots}
        for side, robot in self.robots.items():
            robot.set_dofs_velocity(np.asarray(self.payload["dq_"+side]))
            limit = np.array([87,87,87,87,12,12,12,12,12], float)
            if side == "right": limit[-2:] = self.config["task"]["bear_finger_force_limit"]
            robot.set_dofs_force_range(-limit, limit)
        self.dt = self.config["simulation"]["dt"]
        self.time = 0.
        self.wall = time.monotonic()
        self.released = False
        self.release_completed_time = None
        self.release_tcp = None
        self.withdraw_completed_time = None
        self.paused = False
        self.finished = False
        self.frames, self.metrics, self.reactions, self.events = [], [], [], []
        if not .15 <= args.torque_limit <= .30:
            raise ValueError("Explicit engineering torque threshold must be between 0.15 and 0.30 Nm")
        self.guard = ContactGuard(self.scene,max_tcp_torque_nm=args.torque_limit)
        self.monitor = PhysicalMonitor(self.rest, self.config)
        self.body_faces, self.rim, topology = mouth_structure(self.rest["bag"], self.rest["bag_faces"], self.config)
        if not topology["actual_open_body_boundary"]: raise RuntimeError("Invalid original bag topology")
        self.weights = np.abs(signed_volumes(self.rest["bear"], self.rest["bear_tets"]))
        source = np.load(HERE/"assets"/self.config["bag"]["npz_file"])
        group = self.config["task"]["handle_grasps"]["left"]["group"]
        mapping_error, self.held_ids = cKDTree(self.rest["bag"]).query(source["points"][source[group]] + self.config["bag"]["position"])
        if np.max(mapping_error) > 1e-6: raise RuntimeError("Held-strap material mapping changed")
        self.held_ids = np.unique(self.held_ids)
        self.writer = imageio.get_writer(str(self.out/"final_stage.mp4"), fps=20, codec="libx264", quality=7)
        self.report = {"status":"ready", "task_success":False,
            "scope":"Separate native final-stage simulation; approximate restart, not seamless continuation",
            "source_run":str(Path(args.source).resolve()), "restart":restart,
            "engineering_contact_limits":self.guard.limits.copy(),
            "engineering_limit_change_reason":"Allow physical contact unloading after old 0.15 Nm stop; robot actuator/material/contact laws unchanged" if args.torque_limit>.15 else None,
            "left_command_fixed":True, "native_soft_contact":True,
            "no_soft_pins_or_scripted_vertex_motion":True, "events":self.events}
        write_json(self.out/"config.json", self.config)
        np.savez_compressed(self.out/"rest.npz", **self.rest)
        self.capture("restart", True)
        self.save("ready")
        print("READY " + str(self.out), flush=True)

    def current(self):
        result = {"time":self.time,"bag":scene_module.positions(self.bag),"bear":scene_module.positions(self.bear)}
        for s,r in self.robots.items():
            result["q_"+s] = r.get_qpos().cpu().numpy().reshape(-1)
            result["dq_"+s] = r.get_dofs_velocity().cpu().numpy().reshape(-1)
            hand = r.get_link("hand")
            quat = hand.get_quat().cpu().numpy().reshape(4)
            result["tcp_"+s] = hand.get_pos().cpu().numpy().reshape(3) + Rotation.from_quat(quat[[1,2,3,0]]).apply(TCP)
        return result

    def inside(self, row=None):
        row = self.current() if row is None else row
        return containment(row["bag"],row["bear"],self.body_faces,self.rim,self.weights,self.rest["bear_tets"])

    def capture(self, name, still=False):
        row = self.current()
        self.frames.append(row)
        rgb = np.asarray(self.cameras["overview"].render()[0])
        self.writer.append_data(rgb)
        if still:
            Image.fromarray(rgb).save(self.out/(name+".png"))
            Image.fromarray(np.asarray(self.cameras["detail"].render()[0])).save(self.out/(name+"-detail.png"))
            np.savez_compressed(self.out/(name+".npz"), **row)
        return row

    def save(self, status, error=None):
        self.report.update(status=status,error=error,time_s=self.time,wall_seconds=time.monotonic()-self.wall,
            actual_containment=self.inside(),released=self.released)
        write_json(self.out/"status.json", self.report)
        write_json(self.out/"reactions.json", self.reactions)
        np.save(self.out/"physical_metrics.npy", np.asarray(self.metrics))
        np.savez_compressed(self.out/"states.npz", **{k:np.asarray([r[k] for r in self.frames]) for k in self.frames[0]})
        # Save actual native solver buffers for this live run, including velocity
        # and contact state. The rigid/controller state is stored alongside them.
        try:
            self.scene.sim.coupler._ipc_world.dump()
            self.report["native_dump"] = {"status":"saved", "time_s":self.time,
                "native_frame":int(self.scene.sim.coupler._ipc_world.frame()),
                "workspace":str(self.scene.sim.coupler._ipc_engine.workspace()),
                "scope":"Native soft solver dump plus separate rigid/controller arrays; cross-process recovery is not certified."}
        except Exception as exc:
            # A failed disk checkpoint must not discard the live native world.
            self.report["native_dump"] = {"status":"failed", "error":str(exc)}
        np.savez_compressed(self.out/"live_control_state.npz", **self.current(),
            command_left=self.qcmd["left"],command_right=self.qcmd["right"],
            finger_left=self.fingers["left"],finger_right=self.fingers["right"])
        write_json(self.out/"status.json", self.report)

    def observe(self):
        row = self.current()
        metrics = self.monitor.measure(self.time,row["bag"],row["bear"])
        self.metrics.append(metrics)
        if self.guard.baseline is None:
            self.report["contact_guard_baseline"] = self.guard.set_baseline(self.time)
        reaction, error = self.guard.sample(self.time)
        self.reactions.append(reaction)
        if not metrics[5] or metrics[1] < .05: error = "Invalid or inverted soft body"
        if max(metrics[2:4]) > .002: error = "Soft body penetrated table"
        if metrics[4] > 1.15: error = "Cloth stretch exceeded 15 percent"
        if np.linalg.norm(row["bag"][self.held_ids]-row["tcp_left"],axis=1).min() > .018:
            error = "Held left strap slipped"
        if not self.released and (np.linalg.norm(row["bear"]-row["tcp_right"],axis=1).min() > .045
                or min(reaction["right_finger_opening_projection_n"]) <= .1):
            error = "Right hand lost teddy contact before release"
        return error

    def execute(self, packet):
        name = packet["name"]
        state = self.current()
        # Keep commanded positions continuous between phases; tracking error is
        # observed, never removed by snapping the next command to measured q.
        qs = {s:self.qcmd[s].copy() for s in self.robots}
        start = dict(self.fingers)
        is_release = packet["type"] in ("release", "release_direct")
        if is_release:
            if packet["type"] == "release":
                inside = self.inside(state)
                if (not inside["virtual_closed_body_valid"] or not inside["bear_center_of_mass_inside"]
                        or inside["bear_vertex_inside_fraction"] < .5):
                    raise RuntimeError("Release rejected: actual teddy capture below 50% or COM outside")
            if np.max(np.abs(state["dq_right"][:7])) > .05:
                raise RuntimeError("Release rejected: actual right arm has not stopped")
            end = start | {"right":.04}
            plan = hold_trajectory(qs, packet.get("duration",.8), start, end)
        elif packet["type"] == "hold":
            plan = hold_trajectory(qs,packet.get("duration",.2),start,start)
        elif packet["type"] == "move":
            rotations = {"right":Rotation.from_quat(packet["rotation_xyzw"])} if "rotation_xyzw" in packet else {"right":self.planner.fk("right",qs["right"])[1]}
            plan = self.planner.trajectory(qs,{"right":packet["right_tcp"]},rotations=rotations,
                duration=packet.get("duration",2.),finger_start=start,finger_end=start,
                max_joint_velocity=.35,max_joint_acceleration=.5,clearance_margin=.008,validate_samples=401)
        else: raise ValueError("Unknown command type")
        if not np.isfinite(plan.duration) or plan.duration <= 0:
            raise ValueError("Command duration must be finite and positive")
        if packet["type"] != "move":
            for when in np.linspace(0.,plan.duration,401):
                sample = plan.sample(float(when))
                if min(v["distance"] for v in self.planner.clearance(sample["q"],sample["finger"]).values()) < .008:
                    raise RuntimeError("Hold/opening rigid clearance below original 8 mm")
        if is_release:
            self.released = True
            self.release_tcp = state["tcp_right"].copy()
            if packet["type"] == "release_direct":
                self.report.update(requested_action="direct_gripper_release",
                    release_policy="Explicit user instruction: release without COM or node-containment prerequisites",
                    containment_used_to_authorize_release=False,
                    release_action_completed=False)
        self.events.append({"name":name,"command":packet,"start":self.time,"duration":plan.duration})
        print("PHASE "+name+" duration="+str(plan.duration),flush=True)
        last_progress = time.monotonic()
        error = None
        for step in range(1,int(np.ceil(plan.duration/self.dt))+1):
            sample = plan.sample(min(step*self.dt,plan.duration))
            command_pd(self.robots,self.dofs,sample)
            self.scene.step()
            self.time += self.dt
            self.qcmd = {s:sample["q"][s].copy() for s in self.robots}
            self.fingers = sample["finger"].copy()
            error = self.observe()
            if step % 10 == 0: self.capture(name)
            if (self.out/"SNAPSHOT").exists():
                self.capture("latest",True)
                write_json(self.out/"latest_physics.json",{"time_s":self.time,"physical_metrics":self.metrics[-1],"reaction":self.reactions[-1],"containment":self.inside()})
                (self.out/"SNAPSHOT").unlink()
            if time.monotonic()-last_progress > 15:
                print(f"PROGRESS {name} t={self.time:.3f}",flush=True)
                last_progress = time.monotonic()
            if error or (self.out/"PAUSE").exists(): break
        self.capture(name,True)
        if (self.out/"PAUSE").exists():
            (self.out/"PAUSE").unlink()
            error = error or "Requested pause"
        self.events[-1].update(end=self.time,error=error)
        if not error and is_release:
            if np.min(self.current()["q_right"][-2:]) >= .038:
                self.release_completed_time = self.time
                self.report.update(release_action_completed=True,
                    release_completed_time_s=self.time,
                    actual_finger_openings_m=self.current()["q_right"][-2:].tolist())
            else:
                error = "Release ended without both actual fingers reaching 38 mm"
                self.events[-1]["error"] = error
        if not error and packet["type"] == "move" and self.release_completed_time is not None:
            now = self.current()
            if (np.linalg.norm(now["tcp_right"]-self.release_tcp) >= .10
                    and now["tcp_right"][2] >= now["bear"][:,2].max()+.04
                    and np.min(now["q_right"][-2:]) >= .038):
                self.withdraw_completed_time = self.time
        self.paused = bool(error)
        self.save("paused_after_error" if error else "ready",error)
        print("PAUSED "+str(error) if error else "READY "+name,flush=True)

    def loop(self):
        while not self.finished:
            pending = sorted(p for p in self.commands.glob("*.json") if not p.with_suffix(".done").exists())
            if not pending:
                time.sleep(.2)
                continue
            path = pending[0]
            try:
                packet = json.loads(path.read_text())
                if self.paused and packet["type"] not in ("resume","stop","finish","finish_placement"):
                    # Existing queued motions must not run through a failed guard.
                    controls = [(p,json.loads(p.read_text())) for p in pending]
                    controls = [(p,c) for p,c in controls if c.get("type") in ("resume","stop","finish","finish_placement")]
                    if not controls:
                        time.sleep(.2)
                        continue
                    path,packet = controls[0]
                if packet["type"] == "stop":
                    self.save("stopped")
                    self.finished = True
                elif packet["type"] == "finish_placement":
                    # User requested stopping once placement is achieved. This
                    # is explicitly distinct from the full withdrawal/4 s test.
                    recent=[r for r in self.frames if r["time"]>=self.time-1.-1e-8]
                    after_release=(self.release_completed_time is not None and self.time-self.release_completed_time>=1.-1e-8)
                    passed=bool(not self.paused and after_release and len(recent)>=10 and self.time-recent[0]["time"]>=.95)
                    checks=[self.inside(r) for r in recent]
                    passed=bool(passed and all(c["all_bear_vertices_inside"] and c["bear_center_of_mass_inside"] and np.min(r["q_right"][-2:])>=.038 for r,c in zip(recent,checks)))
                    contact=final_contact_audit(self.current()["bag"],self.current()["bear"],self.rest["bag_faces"],self.rest["bear_tets"])
                    passed=bool(passed and contact["penetration_over_tolerance"]==0)
                    self.report.update(task_success=passed,placement_at_stop_passed=passed,
                        whole_sequence_withdrawal_and_four_second_test_passed=False,
                        placement_stop_checks={"released_at_s":self.release_completed_time,"one_second_after_release":bool(after_release),"sample_count":len(recent),"all_sampled_nodes_and_COM_inside":bool(checks and all(c["all_bear_vertices_inside"] for c in checks)),"final_contact":contact})
                    self.save("placement_completed_at_stop" if passed else "placement_check_failed")
                    self.finished=passed
                elif packet["type"] == "finish":
                    recent = [r for r in self.frames if r["time"] >= self.time-4.-1e-8]
                    ready = (not self.paused and self.release_completed_time is not None
                        and self.withdraw_completed_time is not None
                        and self.time-self.withdraw_completed_time >= 4.-1e-8)
                    passed = bool(ready and len(recent)>=3 and self.time-recent[0]["time"] >= 3.9)
                    passed = bool(passed and all(self.inside(r)["all_bear_vertices_inside"]
                        and self.inside(r)["bear_center_of_mass_inside"]
                        and np.min(r["q_right"][-2:])>=.038 for r in recent))
                    self.report["final_retention"] = {"released_at_s":self.release_completed_time,
                        "withdrawn_at_s":self.withdraw_completed_time,"sample_count":len(recent),
                        "four_seconds_after_withdrawal":bool(ready),"all_samples_full_containment_and_open_fingers":passed}
                    self.report["task_success"] = passed
                    self.report["final_contact"] = final_contact_audit(self.current()["bag"], self.current()["bear"], self.rest["bag_faces"], self.rest["bear_tets"])
                    passed = passed and self.report["final_contact"]["penetration_over_tolerance"] == 0
                    self.report["task_success"] = passed
                    self.save("completed" if passed else "final_check_failed")
                    self.finished = passed
                elif packet["type"] == "resume":
                    self.paused = False
                    self.save("ready")
                elif packet["type"] == "snapshot":
                    self.capture("latest",True)
                    self.save("ready" if not self.paused else "paused_after_error")
                else: self.execute(packet)
            except Exception as exc:
                traceback.print_exc()
                self.paused = True
                self.save("paused_after_error",f"{type(exc).__name__}: {exc}")
            path.with_suffix(".done").write_text("processed\n")
        self.writer.close()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--source",required=True)
    ap.add_argument("--time",type=float,default=46.815)
    ap.add_argument("--output",required=True)
    ap.add_argument("--native-checkpoint-source")
    ap.add_argument("--torque-limit",type=float,default=.15)
    ap.add_argument("--direct-release-and-stop",action="store_true")
    args=ap.parse_args()
    live=LiveStage(args)
    if args.direct_release_and_stop:
        live.execute({"type":"release_direct", "name":"direct_release_teddy", "duration":.8})
        if live.release_completed_time is not None and not live.paused:
            live.save("release_completed")
            live.writer.close()
            print("DIRECT_RELEASE_COMPLETED_AND_STOPPED",flush=True)
            return
    live.loop()

if __name__ == "__main__": main()
