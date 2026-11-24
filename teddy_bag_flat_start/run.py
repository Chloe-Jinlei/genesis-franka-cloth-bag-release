"""Run and record the native-contact teddy bag experiment with fail-closed gates."""
from __future__ import annotations
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

# Linux's visible pyglet window uses GLX. EGL remains the scene default for
# offscreen runs; mixing EGL dispatch with the window left textures unbound.
if "--viewer" in sys.argv and sys.platform.startswith("linux"):
    os.environ["PYOPENGL_PLATFORM"]="glx"

from scene import HERE, build, read_config, positions, physics_metadata
from flat_start import enabled as flat_start_enabled, initial_control_state, run_flat_pickup, validate_task
from motion import Planner, DOWN, TCP, hold_trajectory, command_pd
from physics_config import coupling_diagnostics
from contact_grasp import cloth_grasp_frame
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from PIL import Image
import imageio.v2 as imageio


class GateFailure(RuntimeError):
    pass


def tcp(robot):
    hand = robot.get_link("hand")
    quat = hand.get_quat().cpu().numpy().reshape(4)
    return hand.get_pos().cpu().numpy().reshape(3) + Rotation.from_quat(quat[[1,2,3,0]]).apply(TCP)


class Experiment:
    def __init__(self, config, args):
        validate_task(config, args.task)
        from gravity_release import release_policy
        release_policy(config)
        from capture_recovery import recovery_policy
        recovery_policy(config)
        from partial_gravity_geometry import MODE, planning_mode, partial_policy
        if planning_mode(config)==MODE:
            partial_policy(config)
            import contact_insertion, gravity_release, partial_gravity_geometry
            import capture_recovery, capture_recovery_geometry
            loaded={"run.py":Path(__file__),
                "contact_insertion.py":Path(contact_insertion.__file__),
                "gravity_release.py":Path(gravity_release.__file__),
                "partial_gravity_geometry.py":Path(partial_gravity_geometry.__file__),
                "capture_recovery.py":Path(capture_recovery.__file__),
                "capture_recovery_geometry.py":Path(capture_recovery_geometry.__file__)}
            if any(not (HERE/name).is_file() or path.resolve()!=(HERE/name).resolve()
                    for name,path in loaded.items()):
                raise GateFailure("Partial mode requires the complete reviewed modules in the scene directory; mixed proposal/production loading is forbidden")
        self.config, self.args = config, args
        self.out = HERE/"runs"/(args.output or datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.out.mkdir(parents=True, exist_ok=False)
        (self.out/"config.json").write_text(json.dumps(config,indent=2))
        src=self.out/"source"
        src.mkdir()
        for f in ("scene.py","run.py","motion.py","validate.py","assets_builder.py","two_panel_builder.py","build_tall_bag.py","build_wide_tall_bag.py","physics_config.py","contact_grasp.py","insertion_candidates.py","contact_insertion.py","staged_contact_insertion.py","staged_hand_monitor.py","contact_slice_geometry.py","contact_robot_geometry.py","contact_guard.py","postlift_yaw.py","postlift_orientation.py","pickup_uprighting.py","remesh_bear.py","physical_monitor.py","gravity_release.py","partial_gravity_geometry.py","capture_recovery.py","capture_recovery_geometry.py","flat_start.py","build_flat_bag.py","initial_views.py"):
            if (HERE/f).exists(): shutil.copy2(HERE/f,src/f)
        self.meta=json.loads((HERE/"assets"/config["bag"].get("metadata","metadata.json")).read_text())
        mesh_audit=self.meta["bag"]["mesh"]
        if mesh_audit["connected_components"]!=1 or mesh_audit["nonmanifold_edges"]!=0:
            raise GateFailure("Bag must be one connected manifold cloth shell before simulation")
        asset_snapshot=src/"assets"
        asset_snapshot.mkdir()
        bag_file=config["bag"].get("mesh","two_panel_bag.obj")
        bear_file=config["bear"].get("mesh","bear_physics.obj")
        for name in (bag_file,Path(bag_file).with_suffix(".mtl").name,
                     config["bag"].get("npz_file","two_panel_bag.npz"),
                     config["bag"].get("metadata","metadata_two_panel.json"),
                     bear_file,Path(bear_file).with_suffix(".mtl").name,"teddy_fur.png","cotton_canvas.png","panda_tactile.xml"):
            shutil.copy2(HERE/"assets"/name,asset_snapshot/name)
        frozen_inputs=[self.out/"config.json",*sorted(p for p in src.rglob("*") if p.is_file())]
        (self.out/"input_sha256.json").write_text(json.dumps({
            str(p.relative_to(self.out)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in frozen_inputs},indent=2)+"\n")
        self.planner=Planner(config)
        bear_grasp_quat=config.get("task",{}).get("bear_grasp_rotation_xyzw")
        self.bear_grasp_rotation=Rotation.from_quat(bear_grasp_quat) if bear_grasp_quat is not None else DOWN
        self.bear_grasp_tracking_radius=float(config.get("task",{}).get("bear_grasp_tracking_radius_m",.032))
        if not np.isfinite(self.bear_grasp_tracking_radius) or not .005<=self.bear_grasp_tracking_radius<=.05:
            raise GateFailure("Teddy grasp tracking radius must be finite and between 5 and 50 mm")
        lift_height=config.get("task",{}).get("bear_lift_height_m",1.37)
        if type(lift_height) not in (int,float) or not np.isfinite(lift_height) \
                or lift_height<=float(config["table"]["height"])+.04:
            raise GateFailure("Teddy lift TCP height must be finite and above the table by over 40 mm")
        self.bear_lift_height=float(lift_height)
        self.qs={}
        self.finger=dict.fromkeys(("left","right"),.001)
        self.rotations=dict.fromkeys(("left","right"),DOWN)
        self.left_hold_rotation=None
        bag_source=np.load(HERE/"assets"/config["bag"]["npz_file"])
        grasp_frames={}
        self.flat_start_mode=flat_start_enabled(config)
        if self.flat_start_mode:
            self.qs,self.finger,self.rotations,grasp_frames=initial_control_state(config,self.planner)
        else:
            # Configuration is assigned once before integration. The initial hands
            # surround the two free handles; grasping occurs through finite-force PD.
            for side,group in (("left","grasp_front"),("right","grasp_back")):
                grasp_spec=config.get("task",{}).get("handle_grasps",{}).get(side,{})
                group=grasp_spec.get("group",group)
                frame=cloth_grasp_frame(bag_source["points"],bag_source["faces"],bag_source[group],
                    shell_radius=config["bag"]["thickness"],
                    preferred_approach=grasp_spec.get("preferred_approach",[0,0,-1]))
                grasp_frames[side]=frame.as_dict()
                hold_approach=config.get("task",{}).get("left_hold_preferred_approach")
                if side=="left" and hold_approach is not None:
                    self.left_hold_rotation=cloth_grasp_frame(
                        bag_source["points"],bag_source["faces"],bag_source[group],
                        shell_radius=config["bag"]["thickness"],
                        preferred_approach=hold_approach).rotation
                self.rotations[side]=frame.rotation
                self.finger[side]=max(.001,frame.minimum_half_gap)
                target=np.asarray(config["bag"]["position"])+frame.position
                if args.task in ("grasp","grasp_rotate") and side=="right":
                    target=np.asarray(config["bear"]["position"])+config.get("task",{}).get("bear_grasp_local",[.08,-.014,.034])
                    target[2]+=.09
                    self.finger[side]=.04
                    self.rotations[side]=self.bear_grasp_rotation
                q,err,angle=self.planner.solve(side,target,self.rotations[side])
                if err>.001 or angle>.005: raise GateFailure(f"Initial {side} pose unreachable: {err}, {angle}")
                self.qs[side]=q
        initial_q={s:np.r_[q,self.finger[s],self.finger[s]] for s,q in self.qs.items()}
        self.scene,self.robots,self.bag,self.bear,self.cameras=build(config,
            viewer=args.viewer,width=args.width,height=args.height,initial_q=initial_q)
        self.dofs={s:np.arange(9) for s in self.robots}
        self.rest={"bag":positions(self.bag),"bear":positions(self.bear),
            "bag_faces":np.asarray(self.bag.elems),"bear_tets":np.asarray(self.bear.elems)}
        left_group=config.get("task",{}).get("handle_grasps",{}).get("left",{}).get("group","grasp_front")
        material_points=bag_source["points"][bag_source[left_group]]+np.asarray(config["bag"]["position"])
        mapping_error,self.left_grasp_material_ids=cKDTree(self.rest["bag"]).query(material_points)
        if np.max(mapping_error)>1e-6:
            raise GateFailure("Cannot map the held cloth-handle material to the runtime mesh")
        self.left_grasp_material_ids=np.unique(self.left_grasp_material_ids)
        right_group=config.get("task",{}).get("handle_grasps",{}).get("right",{}).get("group","grasp_back")
        right_material_points=bag_source["points"][bag_source[right_group]]+np.asarray(config["bag"]["position"])
        right_error,self.right_grasp_material_ids=cKDTree(self.rest["bag"]).query(right_material_points)
        if np.max(right_error)>1e-6:
            raise GateFailure("Cannot map the right cloth-handle material to the runtime mesh")
        self.right_grasp_material_ids=np.unique(self.right_grasp_material_ids)
        self.handle_coupling_data=self.scene.sim.coupler._coupling_data
        self.handle_coupling_indices={}
        for side in ("left","right"):
            ids=[]
            for name in ("left_finger","right_finger"):
                matches=[i for i,link in enumerate(self.handle_coupling_data.links)
                    if link.entity.name==f"{side}_franka" and link.name==name]
                if len(matches)!=1:
                    raise GateFailure("Cannot identify both fingers for the cloth-handle contact monitor")
                ids.append(matches[0])
            self.handle_coupling_indices[side]=np.asarray(ids)
        np.savez_compressed(self.out/"rest.npz",**self.rest)
        from validate import signed_volumes
        self.rest_tetra_volumes=signed_volumes(self.rest["bear"],self.rest["bear_tets"])
        self.dt=config["simulation"]["dt"]
        self.time=0.
        self.frames=[]
        self.joint_trace=[]
        self.soft_metrics=[]
        self.must_hold_teddy=False
        self.must_hold_bag=False
        self.must_hold_right_handle=False
        self.contact_guard=None
        self.contact_guard_trace=[]
        self.staged_hand_monitor=None
        self.require_bear_below_left_hand=False
        self.staged_hand_trace=[]
        self.handle_grip_trace=[]
        from physical_monitor import PhysicalMonitor
        self.physical_monitor=PhysicalMonitor(self.rest,config)
        self.events=[]
        self.report={"status":"running","requested_task":args.task,"task_passed":False,
            "physics":physics_metadata(config,self.bag,self.bear)}
        self.report["initialization"]=self.scene.teddy_initialization_report
        self.report["initial_cloth_grasp_geometry"]=grasp_frames
        self.report["shell_conversion"]=self.scene.teddy_shell_report
        self.report["stc_reaction_feedback"]=self.scene.teddy_stc_feedback_report
        self.report["rigid_contact_pairing"]=self.scene.teddy_rigid_contact_report
        self.coupling_trace=[]
        self.writer=imageio.get_writer(str(self.out/"simulation.mp4"),fps=20,codec="libx264",quality=7)
        self.last_frame_time=-1.
        self.wall=time.monotonic()
        self.last_progress=self.wall
        if self.flat_start_mode:
            from initial_views import render_initial_views
            self.report["initial_views"]=render_initial_views(
                self.scene,self.cameras,self.out/"initial_views",config=config)
        self.observe_physics()
        self.capture("initial",True)

    def observe_physics(self):
        bag,bear=positions(self.bag),positions(self.bear)
        row=self.physical_monitor.measure(self.time,bag,bear)
        self.soft_metrics.append(row)
        if np.isfinite(row[1]):
            self.report["minimum_observed_tetra_volume_ratio"]=min(
                self.report.get("minimum_observed_tetra_volume_ratio",1.),float(row[1]))
        error=None
        if not bool(row[5]): error="Nonfinite soft-body state"
        elif row[1]<.05: error=f"Teddy element excessively compressed or inverted: volume ratio {row[1]:.4f}"
        elif max(row[2:4])>.002: error="A soft body penetrated the actual tabletop footprint by over 2 mm"
        elif row[4]>1.15: error=f"Cloth edge stretch exceeded 15 percent: ratio {row[4]:.4f}"
        left_distance=None
        right_distance=None
        if self.must_hold_bag:
            left_distance=float(np.linalg.norm(bag[self.left_grasp_material_ids]-tcp(self.robots["left"]),axis=1).min())
            self.report["maximum_left_held_material_distance_m"]=max(
                self.report.get("maximum_left_held_material_distance_m",0.),left_distance)
            if left_distance>.018 and error is None:
                error="The designated held cloth-handle material slipped from the left gripper"
        if self.must_hold_right_handle:
            handle_distance=float(np.linalg.norm(bag[self.right_grasp_material_ids]-tcp(self.robots["right"]),axis=1).min())
            self.report["maximum_right_held_handle_material_distance_m"]=max(
                self.report.get("maximum_right_held_handle_material_distance_m",0.),handle_distance)
            if handle_distance>.018 and error is None:
                error="The designated right cloth handle slipped before the handoff"
            grip_row={"time_s":self.time,"left_material_distance_m":left_distance,
                "right_material_distance_m":handle_distance}
            data=self.handle_coupling_data
            for side,ids in self.handle_coupling_indices.items():
                reactions=np.einsum("ij,ij->i",data.out_forces[0,ids],data.aim_transforms[0,ids,:3,1])
                grip_row[side+"_opening_reactions_n"]=reactions.tolist()
                if (not np.isfinite(reactions).all() or np.min(reactions)<=.1) and error is None:
                    error=f"The {side} hand lost two-sided soft contact before the handle handoff"
            self.handle_grip_trace.append(grip_row)
        if self.flat_start_mode and self.must_hold_bag and not self.must_hold_right_handle:
            ids=self.handle_coupling_indices["left"]
            data=self.handle_coupling_data
            reactions=np.einsum("ij,ij->i",data.out_forces[0,ids],data.aim_transforms[0,ids,:3,1])
            self.handle_grip_trace.append({"time_s":self.time,
                "left_material_distance_m":left_distance,
                "left_opening_reactions_n":reactions.tolist(),
                "single_handle_hold":True})
            if (not np.isfinite(reactions).all() or np.min(reactions)<=.1) and error is None:
                error="Left hand lost two-sided soft contact during single-handle hold"
        if self.must_hold_teddy:
            right_distance=float(np.linalg.norm(bear-tcp(self.robots["right"]),axis=1).min())
            if right_distance>.045 and error is None:
                error="Teddy slipped out of the right finger pads"
        if self.contact_guard is not None:
            contact_row,contact_error=self.contact_guard.sample(self.time)
            contact_row["left_designated_material_distance_m"]=left_distance
            contact_row["right_nearest_teddy_distance_m"]=right_distance
            contact_row["right_two_sided_contact_required"]=self.must_hold_teddy
            if self.must_hold_teddy and min(contact_row["right_finger_opening_projection_n"])<=.1:
                contact_error="; ".join(filter(None,(contact_error,"Right fingers lost two-sided soft-contact opening reaction before release")))
                contact_row["stop_reason"]=contact_error
            self.contact_guard_trace.append(contact_row)
            if contact_error and error is None: error=contact_error
        if self.require_bear_below_left_hand:
            hand_row,hand_error=self.staged_hand_monitor.measure(self.time,bear)
            self.staged_hand_trace.append(hand_row)
            if hand_error and error is None: error=hand_error
        if error:
            np.savez_compressed(self.out/"failed_soft_state.npz",time=self.time,bag=bag,bear=bear,metrics=row)
            raise GateFailure(error)

    def capture(self, phase, image=False):
        bp, tp=positions(self.bag), positions(self.bear)
        if not np.isfinite(bp).all() or not np.isfinite(tp).all(): raise GateFailure("Nonfinite soft-body state")
        from validate import signed_volumes
        minimum_volume_ratio=float(np.min(signed_volumes(tp,self.rest["bear_tets"])/self.rest_tetra_volumes))
        self.report["minimum_observed_tetra_volume_ratio"]=min(
            self.report.get("minimum_observed_tetra_volume_ratio",1.),minimum_volume_ratio)
        row={"time":self.time,"bag":bp,"bear":tp}
        for side,robot in self.robots.items():
            row["q_"+side]=robot.get_qpos().cpu().numpy().copy()
            row["tcp_"+side]=tcp(robot)
            row["force_"+side]=robot.get_dofs_control_force().cpu().numpy().copy()
        self.frames.append(row)
        self.coupling_trace.append({"time":self.time,**coupling_diagnostics(self.scene)})
        rgb=np.asarray(self.cameras["overview"].render()[0])
        self.writer.append_data(rgb)
        if image:
            Image.fromarray(rgb).save(self.out/(phase+".png"))
            detail=np.asarray(self.cameras["detail"].render()[0])
            Image.fromarray(detail).save(self.out/(phase+"-detail.png"))
            np.savez_compressed(self.out/(phase+".npz"),**row)
        self.last_frame_time=self.time
        if minimum_volume_ratio<.05:
            raise GateFailure(f"Teddy element excessively compressed or inverted: minimum volume ratio {minimum_volume_ratio:.4f}")
        if self.must_hold_teddy and np.linalg.norm(tp-tcp(self.robots["right"]),axis=1).min()>.045:
            raise GateFailure(f"Teddy slipped out of the right finger pads during {phase}")
        if self.must_hold_bag:
            distance=float(np.linalg.norm(bp[self.left_grasp_material_ids]-row["tcp_left"],axis=1).min())
            self.report["maximum_left_held_material_distance_m"]=max(
                self.report.get("maximum_left_held_material_distance_m",0.),distance)
            if distance>.018:
                raise GateFailure(f"The held cloth-handle material slipped from the left gripper during {phase}")
        return row

    def execute(self,label,plan):
        print(f"PHASE {label} duration={plan.duration:.3f}s",flush=True)
        self.events.append({"phase":label,"start":self.time,"duration":plan.duration})
        for i in range(1,int(np.ceil(plan.duration/self.dt))+1):
            sample=plan.sample(min(i*self.dt,plan.duration))
            command_pd(self.robots,self.dofs,sample)
            self.scene.step()
            self.time+=self.dt
            self.joint_trace.append(np.r_[self.time,*[r.get_qpos().cpu().numpy().reshape(-1) for r in self.robots.values()]])
            self.observe_physics()
            if self.time-self.last_frame_time>=.0499: self.capture(label)
            now=time.monotonic()
            if now-self.last_progress>15:
                print(f"PROGRESS {label} simulation={self.time:.2f}s wall={now-self.wall:.1f}s",flush=True)
                self.last_progress=now
            if self.args.max_seconds and self.time>=self.args.max_seconds:
                raise GateFailure("Stopped at the requested diagnostic duration; task not complete")
        self.qs=plan.end_q()
        self.finger=plan.finger_end.copy()
        self.capture(label,True)
        (self.out/"events.json").write_text(json.dumps(self.events,indent=2))
        (self.out/"latest_contact_diagnostics.json").write_text(json.dumps(self.coupling_trace[-1],indent=2))

    def hold(self,label,duration,**fingers):
        finish=self.finger|fingers
        self.execute(label,hold_trajectory(self.qs,duration,self.finger,finish))

    def close_teddy_gripper(self):
        task=self.config.get("task",{})
        force=float(task.get("bear_finger_force_limit",4.))
        gap=float(task.get("bear_finger_target",.027))
        self.robots["right"].set_dofs_force_range(
            np.array([-force,-force]),np.array([force,force]),dofs_idx_local=np.array([7,8]))
        self.report["teddy_grip_limits"]={"force_per_finger_n":force,"target_per_finger_m":gap}
        self.hold("close_teddy_gripper",1.,right=gap)
        self.must_hold_teddy=True

    def move(self,label,targets,duration=2.5,rotations=None):
        finish_rotations={s:self.rotations[s] for s in targets}
        if rotations: finish_rotations.update(rotations)
        speed=float(self.config.get("task",{}).get("motion_speed_scale",1.))
        plan=self.planner.trajectory(self.qs,targets,duration=duration/speed,
            rotations=finish_rotations,finger_start=self.finger,
            finger_end=self.finger,clearance_margin=.008,
            max_joint_velocity=.55*speed,max_joint_acceleration=speed**2)
        self.execute(label,plan)
        self.rotations.update(finish_rotations)

    def near_material(self,entity,side,radius=.028):
        p=positions(entity)
        distances=np.linalg.norm(p-tcp(self.robots[side]),axis=1)
        return {"near_vertices":int(np.sum(distances<radius)),"nearest_m":float(distances.min())}

    def move_guarded(self,label,stage):
        """Slow physical PD motion for the explicitly monitored contact trial."""
        from partial_gravity_geometry import planning_mode, MODE
        if planning_mode(self.config)==MODE:
            return self.move_partial_reference(label,stage)
        rotation=Rotation.from_quat(stage["rotation_xyzw"])
        # The finite-force closing target is not the measured finger opening
        # around the teddy. Check rigid clearance at the observed opening, then
        # retain the original motor target so physical grip pressure persists.
        observed_fingers={side:float(np.clip(robot.get_qpos().cpu().numpy().reshape(-1)[7:].mean(),0.,.04))
            for side,robot in self.robots.items()}
        plan=self.planner.trajectory(self.qs,{"right":stage["right_tcp_m"]},
            duration=stage["duration_s"],rotations={"right":rotation},
            finger_start=observed_fingers,finger_end=observed_fingers,clearance_margin=.008,
            validate_samples=201,max_joint_velocity=.35,max_joint_acceleration=.5)
        plan.finger_start=self.finger.copy()
        plan.finger_end=self.finger.copy()
        self.execute(label,plan)
        self.rotations["right"]=rotation

    def move_partial_reference(self,label,stage):
        """Execute the saved, checked partial-path reference without replanning."""
        from scipy.interpolate import PPoly
        from motion import MotionPlan, minimum_jerk
        from postlift_yaw import continuous_rate_bounds, _clearance_rows, _finite_tree
        packet=stage.get("exact_reference_curve")
        if not isinstance(packet,dict) or packet.get("execution_must_keep_existing_finite_motor_targets") is not True \
                or packet.get("spline_independent_variable")!="path fraction in [0,1]" \
                or packet.get("time_scaling")!="original MotionPlan minimum_jerk(t/duration), not a seconds-domain spline":
            raise GateFailure("Partial movement requires its exact checked reference and time scaling")
        duration=float(packet["duration_s"])
        if not np.isfinite(duration) or duration<=0 or duration!=float(stage["duration_s"]):
            raise GateFailure("Partial reference duration changed")
        curves={}
        for side in ("left","right"):
            x=np.asarray(packet["curves"][side]["x"],float)
            c=np.asarray(packet["curves"][side]["c"],float)
            if x.ndim!=1 or len(x)<2 or c.shape!=(4,len(x)-1,7) \
                    or not np.isfinite(x).all() or not np.isfinite(c).all() \
                    or x[0]!=0 or x[-1]!=1 or np.any(np.diff(x)<=0):
                raise GateFailure("Invalid partial reference polynomial")
            curves[side]=PPoly(c,x,extrapolate=False)
            # Continuous-rate bounds assume the serialized path is still C2.
            # Allow only floating-point roundoff in q, dq/ds and d2q/ds2;
            # no coefficients are repaired or replaced before execution.
            for order,tolerance in enumerate((1e-10,1e-9,1e-8)):
                derivative=curves[side].derivative(order)
                for knot in range(1,len(x)-1):
                    left=np.polynomial.polynomial.polyval(x[knot]-x[knot-1],
                        derivative.c[:,knot-1, :][::-1])
                    right=derivative.c[-1,knot,:]
                    if np.max(np.abs(left-right))>tolerance:
                        raise GateFailure("Partial reference polynomial is not C2 across its saved knots")
        plan=MotionPlan(curves,duration,self.finger.copy(),self.finger.copy())
        plan.validate_joint_limits(self.planner.limits)
        if any(not np.array_equal(plan.sample(0.)["q"][s],self.qs[s]) \
                or not np.array_equal(plan.end_q()[s],np.asarray(stage["command_q_end"][s]))
                for s in ("left","right")):
            raise GateFailure("Partial reference does not preserve declared command endpoints")
        rates=continuous_rate_bounds(plan)
        if not _finite_tree(rates) or any(np.max(r["velocity_upper_bound_rad_s"])>.35 \
                or np.max(r["acceleration_upper_bound_rad_s2"])>.5 for r in rates.values()):
            raise GateFailure("Partial reference exceeds original continuous rate bounds")
        state=self.frames[-1]
        from contact_insertion import _state_context, _policy
        from partial_gravity_geometry import validate_commands
        context=_state_context(self.rest,state,self.config,_policy(self.config))
        if context["reasons"] or abs(float(state["time"])-self.time)>1e-9:
            raise GateFailure("Invalid fresh actual state before partial reference execution")
        validate_commands(context,self.qs)
        start,unused=self.planner.fk("right",self.qs["right"])
        target=np.asarray(stage["right_tcp_m"],float)
        rotation=Rotation.from_quat(stage["rotation_xyzw"])
        minimum=float("inf"); maximum_position=0.; maximum_angle=0.
        for t in np.linspace(0.,duration,401):
            sample=plan.sample(float(t))
            if not _finite_tree(sample) or not np.array_equal(sample["q"]["left"],self.qs["left"]):
                raise GateFailure("Partial reference changes the held-bag command")
            for left in (self.qs["left"],context["qs"]["left"]):
                _,row=_clearance_rows(self.planner.clearance(dict(sample["q"],left=left),context["fingers"]),"partial reference")
                minimum=min(minimum,float(row["distance"]))
            p,r=self.planner.fk("right",sample["q"]["right"])
            expected=start+float(minimum_jerk(t/duration)[0])*(target-start)
            pe=float(np.linalg.norm(p-expected)); ae=float((r*rotation.inv()).magnitude())
            maximum_position=max(maximum_position,pe); maximum_angle=max(maximum_angle,ae)
            if not np.isfinite([pe,ae]).all() or pe>.0005 or ae>.002:
                raise GateFailure("Partial reference exceeds original straight-path FK tolerances")
        np.savez_compressed(self.out/(label+"_partial_reference.npz"),duration=duration,
            left_x=curves["left"].x,left_c=curves["left"].c,
            right_x=curves["right"].x,right_c=curves["right"].c,
            finger_left=self.finger["left"],finger_right=self.finger["right"])
        (self.out/(label+"_partial_reference_review.json")).write_text(json.dumps({
            "observed_time_s":float(state["time"]),"minimum_sampled_rigid_clearance_m":minimum,
            "maximum_sampled_position_error_m":maximum_position,"maximum_sampled_angle_error_rad":maximum_angle,
            "sample_count":401,"exact_checked_curve_preserved":True,
            "physical_success":False,"native_contact_and_deformation_checks_required":True},indent=2)+"\n")
        self.execute(label,plan)
        self.rotations["right"]=rotation

    def run_staged_contact_insertion(self,choice):
        """Observe the actual soft state again before each contact motion."""
        from staged_contact_insertion import plan_staged_contact
        from staged_hand_monitor import NativeHandHeightMonitor
        settings=self.config["task"]["contact_trial"]["staged"]
        individual_limit=float(settings.get("maximum_individual_xy_m",.03))
        cumulative_limit=float(settings.get("maximum_cumulative_xy_m",.05))
        if not np.isfinite([individual_limit,cumulative_limit]).all() \
                or not .03<=individual_limit<=.10 or not .05<=cumulative_limit<=.12 \
                or cumulative_limit<individual_limit:
            raise GateFailure("Invalid explicit staged individual/cumulative XY budgets")
        self.report["staged_xy_budget"]={"maximum_individual_xy_m":individual_limit,
            "maximum_cumulative_xy_m":cumulative_limit,
            "scope":"Measured XY endpoint displacements; physical and robot clearance gates unchanged"}
        self.staged_hand_monitor=NativeHandHeightMonitor(self.scene,minimum_gap_m=.005)
        self.report["staged_native_hand_monitor"]=self.staged_hand_monitor.report()
        np.savez_compressed(self.out/"staged_native_hand_reference.npz",
            **self.staged_hand_monitor.reference_geometry())
        self.report["staged_contact_plans"]=[]
        names={"entry":"lower_teddy_to_rim","center":"center_teddy_at_rim",
            "insert":"insert_teddy"}
        original_rotation=Rotation.from_quat(choice["stages_to_execute"][0]["rotation_xyzw"])
        previous=choice
        previous_time=-np.inf
        cumulative_xy=0.
        for phase in ("entry","center","insert"):
            state=self.frames[-1]
            observed_time=float(state["time"])
            start=np.asarray(state["tcp_right"],float).copy()
            if not np.isfinite(observed_time) or abs(observed_time-self.time)>1e-9 \
                    or observed_time<=previous_time:
                raise GateFailure("Staged insertion requires a fresh actual endpoint observation")
            fresh=plan_staged_contact(self.rest,state,self.config,phase=phase,previous=previous,
                output=self.out/f"staged_contact_{phase}.json")
            self.report["staged_contact_plans"].append(fresh)
            if fresh["status"]!="candidate":
                raise GateFailure(f"Fresh staged-contact {phase} plan rejected; see staged_contact_{phase}.json")
            stage=fresh["stage"]
            target=np.asarray(stage["right_tcp_m"],float)
            quat=np.asarray(stage["rotation_xyzw"],float)
            reference=np.asarray(fresh["original_rotation_xyzw"],float)
            duration=float(stage["duration_s"])
            if fresh["phase"]!=phase or stage["name"]!=names[phase] \
                    or not np.isfinite(float(fresh["observed_time_s"])) \
                    or abs(float(fresh["observed_time_s"])-observed_time)>1e-9 \
                    or target.shape!=(3,) or quat.shape!=(4,) or reference.shape!=(4,) \
                    or not all(np.isfinite(v).all() for v in (target,quat,reference)) \
                    or abs(np.linalg.norm(quat)-1.)>1e-8 or abs(np.linalg.norm(reference)-1.)>1e-8 \
                    or not np.isfinite(duration) or duration<4.:
                raise GateFailure("Invalid staged contact phase, endpoint, time or orientation record")
            if (Rotation.from_quat(quat).inv()*original_rotation).magnitude()>1e-8 \
                    or (Rotation.from_quat(reference).inv()*original_rotation).magnitude()>1e-8:
                raise GateFailure("Staged contact plan changed the complete original wrist orientation")
            delta=target-start
            declared_before=float(fresh["cumulative_xy_m_before"])
            declared_after=float(fresh["cumulative_xy_m_after"])
            if not np.isfinite([declared_before,declared_after]).all() \
                    or abs(declared_before-cumulative_xy)>1e-9 \
                    or abs(declared_after-declared_before-np.linalg.norm(delta[:2]))>1e-9 \
                    or declared_after>cumulative_limit+1e-9:
                raise GateFailure("Staged plan and actual cumulative XY budget disagree")
            if phase=="center":
                adaptive="center_offsets_xy_m" in settings
                configured=np.asarray(settings.get("center_offsets_xy_m",
                    [settings.get("center_offset_xy_m",[-.027,-.010])]),float)
                if configured.ndim!=2 or configured.shape[1]!=2 or not 1<=len(configured)<=8 \
                        or not np.isfinite(configured).all() \
                        or np.any(np.linalg.norm(configured,axis=1)<1e-6) \
                        or np.any(np.linalg.norm(configured,axis=1)>individual_limit) \
                        or len(np.unique(configured,axis=0))!=len(configured) \
                        or not np.any(np.linalg.norm(configured-delta[:2],axis=1)<1e-9) \
                        or abs(delta[2])>1e-9 \
                        or not 0.<np.linalg.norm(delta[:2])<=individual_limit+1e-9:
                    raise GateFailure("Low contact centering disagrees with its bounded declared XY move")
                if adaptive:
                    selection=fresh.get("center_selection",{})
                    index=selection.get("selected_index")
                    picked=np.asarray(fresh.get("selected_center_offset_xy_m",[]),float)
                    reported=np.asarray(selection.get("selected_offset_xy_m",[]),float)
                    if type(index) is not int or not 0<=index<len(configured) \
                            or picked.shape!=(2,) or reported.shape!=(2,) \
                            or not np.isfinite([picked,reported]).all() \
                            or np.linalg.norm(picked-configured[index])>1e-9 \
                            or np.linalg.norm(picked-reported)>1e-9 \
                            or np.linalg.norm(picked-delta[:2])>1e-9 \
                            or selection.get("actual_observed_time_s")!=observed_time \
                            or selection.get("later_insert_requires_fresh_actual_observation") is not True:
                        raise GateFailure("Adaptive center selection disagrees with the actual state and frozen candidate list")
                self.require_bear_below_left_hand=True
                self.check_staged_hand_clearance(positions(self.bear))
            elif np.linalg.norm(delta[:2])>1e-9 or delta[2]>=0.:
                raise GateFailure("Contact entry and insertion must descend at the measured current XY")
            self.move_guarded(names[phase],stage)
            actual=float(np.linalg.norm(np.asarray(self.frames[-1]["tcp_right"])[:2]-start[:2]))
            cumulative_xy+=actual
            fresh["runtime_observed_endpoint_displacement_m"]=actual
            fresh["runtime_observed_cumulative_endpoint_displacement_m"]=cumulative_xy
            fresh["runtime_distance_scope"]="Sum of measured XY endpoint displacements of all staged motions"
            if not np.isfinite(actual) or actual>individual_limit+1e-9 or cumulative_xy>cumulative_limit+1e-9:
                raise GateFailure("Actual staged motion exceeded its explicit individual/cumulative XY budget")
            fresh["runtime_endpoint_time_s"]=self.time
            fresh["runtime_endpoint_tcp_m"]=np.asarray(self.frames[-1]["tcp_right"]).tolist()
            previous_time=observed_time
            previous=fresh
        self.require_bear_below_left_hand=False
        self.must_hold_teddy=False
        self.hold("release_teddy",.5,right=.04)
        self.move_guarded("withdraw_right_arm",{
            "right_tcp_m":fresh["outside_tcp_m"],
            "rotation_xyzw":fresh["stage"]["rotation_xyzw"],"duration_s":4.})
        self.hold("final_settle",2.)
        self.report["sequence_completed"]=True

    def check_staged_hand_clearance(self,bear):
        row,reason=self.staged_hand_monitor.measure(self.time,bear)
        self.staged_hand_trace.append(row)
        if reason: raise GateFailure(reason)

    def begin_guarded_contact_monitor(self):
        """Keep a single force baseline from before any optional held-bear yaw."""
        from contact_guard import ContactGuard
        if self.contact_guard is not None:
            return
        self.contact_guard=ContactGuard(self.scene)
        self.report["contact_guard_baseline"]=self.contact_guard.set_baseline(self.time)
        row,reason=self.contact_guard.sample(self.time)
        row["left_designated_material_distance_m"]=float(np.linalg.norm(
            positions(self.bag)[self.left_grasp_material_ids]-tcp(self.robots["left"]),axis=1).min())
        row["right_nearest_teddy_distance_m"]=float(np.linalg.norm(
            positions(self.bear)-tcp(self.robots["right"]),axis=1).min())
        row["right_two_sided_contact_required"]=True
        if min(row["right_finger_opening_projection_n"])<=.1:
            reason="; ".join(filter(None,(reason,"Right fingers lack two-sided soft contact at trial entry")))
            row["stop_reason"]=reason
        self.contact_guard_trace.append(row)
        if reason: raise GateFailure(reason)

    def run_postlift_orientation(self):
        """Execute the exact checked joint curve, then observe the real soft body."""
        from postlift_orientation import plan_postlift_orientation, target_rotation as orientation_target_rotation, _finite_tree, _clearance_rows, _serial
        from contact_insertion import _state_context, _policy
        state=self.frames[-1]
        observed=float(np.asarray(state["time"]))
        if not np.isfinite(observed) or abs(observed-self.time)>1e-9:
            raise GateFailure("Postlift orientation requires a fresh actual state")
        if not self.must_hold_bag or not self.must_hold_teddy or self.contact_guard is None:
            raise GateFailure("Postlift orientation requires both real grasps and the contact monitor")
        choice,plan=plan_postlift_orientation(self.rest,state,self.config,self.qs)
        self.report["postlift_orientation"]=choice
        (self.out/"postlift_orientation_candidate.json").write_text(json.dumps(choice,indent=2,allow_nan=False)+"\n")
        if choice.get("status")!="candidate" or plan is None:
            raise GateFailure("No admissible guarded postlift orientation; see postlift_orientation_candidate.json")
        option=self.config["task"]["contact_trial"]["postlift_orientation"]
        _,actual_rotation=self.planner.fk("right",np.asarray(state["q_right"],float)[:7])
        target_rotation=orientation_target_rotation(option,actual_rotation)
        target_tcp=np.asarray(state["tcp_right"],float)
        required_checks={"actual_original_physical_gates","lift_above_table_40mm",
            "command_actual_pose_consistent","analytic_joint_limits","continuous_reference_rate_bounds",
            "sampled_all_raw_rigid_checks_finite_and_above_8mm","sampled_actual_reference_fk_tolerances",
            "actual_and_commanded_left_frozen_rotation_above_5mm","returns_exact_checked_plan",
            "frozen_rotation_bear_table_above_40mm"}
        if not _finite_tree(choice) or choice.get("kind")!="guarded_postlift_orientation" \
                or choice.get("parameters")!=option \
                or abs(float(choice.get("observed_time_s",-1))-observed)>1e-9 \
                or choice.get("requires_fresh_actual_preparation_after_execution") is not True \
                or choice.get("physical_success") is not False \
                or choice.get("runtime_execution_performed") is not False \
                or choice.get("modifies_soft_points") is not False \
                or set(choice.get("checks",{}))!=required_checks or not all(value is True for value in choice["checks"].values()) \
                or np.linalg.norm(np.asarray(choice["target_tcp_m"])-target_tcp)>1e-9 \
                or (Rotation.from_quat(choice["target_rotation_xyzw"])*target_rotation.inv()).magnitude()>1e-9 \
                or not np.isfinite(plan.duration) or plan.duration<float(option["duration_s"]) \
                or abs(plan.duration-float(choice["duration_s"]))>1e-9:
            raise GateFailure("Postlift orientation result disagrees with the fresh configured motion")
        start_sample=plan.sample(0.)
        end_q=plan.end_q()
        for side in ("left","right"):
            if not np.array_equal(start_sample["q"][side],self.qs[side]) \
                    or not np.array_equal(end_q[side],np.asarray(choice[side+"_q_end"])):
                raise GateFailure("Postlift orientation changed its checked command endpoints")
        if not np.array_equal(end_q["left"],self.qs["left"]):
            raise GateFailure("Postlift orientation must keep the left command fixed")
        # Only the finite-force finger motor targets replace measured geometry;
        # the checked arm curves and their timing are executed without replanning.
        plan.finger_start=self.finger.copy()
        plan.finger_end=self.finger.copy()
        np.savez_compressed(self.out/"postlift_orientation_plan.npz",duration=plan.duration,
            observed_time=observed,left_x=plan.curves["left"].x,left_c=plan.curves["left"].c,
            right_x=plan.curves["right"].x,right_c=plan.curves["right"].c,
            finger_left=self.finger["left"],finger_right=self.finger["right"])
        execution=dict(status="executing",start_time_s=self.time,planned_duration_s=plan.duration,
            actual_step_count=int(np.ceil(plan.duration/self.dt)),physical_success=False,
            exact_checked_plan_executed=True,contact_guard_baseline_reset=False)
        self.report["postlift_orientation_execution"]=execution
        try:
            self.execute("orient_teddy_postlift_orientation",plan)
            self.rotations["right"]=target_rotation
            end_state=self.frames[-1]
            context=_state_context(self.rest,end_state,self.config,_policy(self.config))
            endpoint=np.asarray(end_state["tcp_right"],float)
            _,end_rotation=self.planner.fk("right",np.asarray(end_state["q_right"],float)[:7])
            position_error=float(np.linalg.norm(endpoint-target_tcp))
            rotation_error=float((end_rotation*target_rotation.inv()).magnitude())
            execution.update(end_time_s=self.time,endpoint_time_s=float(end_state["time"]),
                endpoint_tcp_m=endpoint.tolist(),endpoint_rotation_xyzw=end_rotation.as_quat().tolist(),
                endpoint_position_error_m=position_error,endpoint_rotation_error_rad=rotation_error,
                actual_physical_state=context["physical"],endpoint_rejections=context["reasons"])
            if not _finite_tree(execution): raise GateFailure("Nonfinite postlift orientation endpoint evidence")
            _clearance_rows(context["physical"]["actual_rigid_clearances"],"postlift orientation actual endpoint")
            expected_time=observed+execution["actual_step_count"]*self.dt
            if context["reasons"] or abs(self.time-expected_time)>1e-8 \
                    or abs(float(end_state["time"])-self.time)>1e-9 \
                    or position_error>.002 or rotation_error>.015 \
                    or float(np.asarray(end_state["bear"])[:,2].min())<=float(self.config["table"]["height"])+.04:
                raise GateFailure("Actual postlift orientation endpoint failed its fresh physical/pose checks")
            execution["status"]="executed_endpoint_passed"
        except Exception as error:
            execution.update(status="execution_rejected",error_type=type(error).__name__,error=str(error))
            raise
        finally:
            self.report["postlift_orientation_execution"]=_serial(execution)
            (self.out/"postlift_orientation_execution.json").write_text(json.dumps(_serial(execution),indent=2,allow_nan=False)+"\n")

    def run_postlift_yaw(self):
        """Execute the exact checked joint curve, then observe the real soft body."""
        from postlift_yaw import plan_postlift_yaw, _finite_tree, _clearance_rows, _serial
        from contact_insertion import _state_context, _policy
        state=self.frames[-1]
        observed=float(np.asarray(state["time"]))
        if not np.isfinite(observed) or abs(observed-self.time)>1e-9:
            raise GateFailure("Postlift yaw requires a fresh actual state")
        if not self.must_hold_bag or not self.must_hold_teddy or self.contact_guard is None:
            raise GateFailure("Postlift yaw requires both real grasps and the contact monitor")
        choice,plan=plan_postlift_yaw(self.rest,state,self.config,self.qs)
        self.report["postlift_yaw"]=choice
        (self.out/"postlift_yaw_candidate.json").write_text(json.dumps(choice,indent=2,allow_nan=False)+"\n")
        if choice.get("status")!="candidate" or plan is None:
            raise GateFailure("No admissible guarded postlift yaw; see postlift_yaw_candidate.json")
        option=self.config["task"]["contact_trial"]["postlift_yaw"]
        _,actual_rotation=self.planner.fk("right",np.asarray(state["q_right"],float)[:7])
        target_rotation=Rotation.from_rotvec([0.,0.,np.deg2rad(option["world_z_degrees"])])*actual_rotation
        target_tcp=np.asarray(state["tcp_right"],float)
        required_checks={"actual_original_physical_gates","lift_above_table_40mm",
            "command_actual_pose_consistent","analytic_joint_limits","continuous_reference_rate_bounds",
            "sampled_all_raw_rigid_checks_finite_and_above_8mm","sampled_actual_reference_fk_tolerances",
            "actual_and_commanded_left_frozen_rotation_above_5mm","returns_exact_checked_plan"}
        if not _finite_tree(choice) or choice.get("kind")!="guarded_postlift_yaw" \
                or choice.get("parameters")!=option \
                or abs(float(choice.get("observed_time_s",-1))-observed)>1e-9 \
                or choice.get("requires_fresh_actual_preparation_after_execution") is not True \
                or choice.get("physical_success") is not False \
                or choice.get("runtime_execution_performed") is not False \
                or choice.get("modifies_soft_points") is not False \
                or set(choice.get("checks",{}))!=required_checks or not all(value is True for value in choice["checks"].values()) \
                or np.linalg.norm(np.asarray(choice["target_tcp_m"])-target_tcp)>1e-9 \
                or (Rotation.from_quat(choice["target_rotation_xyzw"])*target_rotation.inv()).magnitude()>1e-9 \
                or not np.isfinite(plan.duration) or plan.duration<float(option["duration_s"]) \
                or abs(plan.duration-float(choice["duration_s"]))>1e-9:
            raise GateFailure("Postlift yaw result disagrees with the fresh configured motion")
        start_sample=plan.sample(0.)
        end_q=plan.end_q()
        for side in ("left","right"):
            if not np.array_equal(start_sample["q"][side],self.qs[side]) \
                    or not np.array_equal(end_q[side],np.asarray(choice[side+"_q_end"])):
                raise GateFailure("Postlift yaw changed its checked command endpoints")
        if not np.array_equal(end_q["left"],self.qs["left"]):
            raise GateFailure("Postlift yaw must keep the left command fixed")
        # Only the finite-force finger motor targets replace measured geometry;
        # the checked arm curves and their timing are executed without replanning.
        plan.finger_start=self.finger.copy()
        plan.finger_end=self.finger.copy()
        np.savez_compressed(self.out/"postlift_yaw_plan.npz",duration=plan.duration,
            observed_time=observed,left_x=plan.curves["left"].x,left_c=plan.curves["left"].c,
            right_x=plan.curves["right"].x,right_c=plan.curves["right"].c,
            finger_left=self.finger["left"],finger_right=self.finger["right"])
        execution=dict(status="executing",start_time_s=self.time,planned_duration_s=plan.duration,
            actual_step_count=int(np.ceil(plan.duration/self.dt)),physical_success=False,
            exact_checked_plan_executed=True,contact_guard_baseline_reset=False)
        self.report["postlift_yaw_execution"]=execution
        try:
            self.execute("orient_teddy_postlift",plan)
            self.rotations["right"]=target_rotation
            end_state=self.frames[-1]
            context=_state_context(self.rest,end_state,self.config,_policy(self.config))
            endpoint=np.asarray(end_state["tcp_right"],float)
            _,end_rotation=self.planner.fk("right",np.asarray(end_state["q_right"],float)[:7])
            position_error=float(np.linalg.norm(endpoint-target_tcp))
            rotation_error=float((end_rotation*target_rotation.inv()).magnitude())
            execution.update(end_time_s=self.time,endpoint_time_s=float(end_state["time"]),
                endpoint_tcp_m=endpoint.tolist(),endpoint_rotation_xyzw=end_rotation.as_quat().tolist(),
                endpoint_position_error_m=position_error,endpoint_rotation_error_rad=rotation_error,
                actual_physical_state=context["physical"],endpoint_rejections=context["reasons"])
            if not _finite_tree(execution): raise GateFailure("Nonfinite postlift yaw endpoint evidence")
            _clearance_rows(context["physical"]["actual_rigid_clearances"],"postlift yaw actual endpoint")
            expected_time=observed+execution["actual_step_count"]*self.dt
            if context["reasons"] or abs(self.time-expected_time)>1e-8 \
                    or abs(float(end_state["time"])-self.time)>1e-9 \
                    or position_error>.002 or rotation_error>.015 \
                    or float(np.asarray(end_state["bear"])[:,2].min())<=float(self.config["table"]["height"])+.04:
                raise GateFailure("Actual postlift yaw endpoint failed its fresh physical/pose checks")
            execution["status"]="executed_endpoint_passed"
        except Exception as error:
            execution.update(status="execution_rejected",error_type=type(error).__name__,error=str(error))
            raise
        finally:
            self.report["postlift_yaw_execution"]=_serial(execution)
            (self.out/"postlift_yaw_execution.json").write_text(json.dumps(_serial(execution),indent=2,allow_nan=False)+"\n")

    def run_capture_recovery(self, candidate, review, actual_cumulative_xy):
        """Keep the original grasp while making bounded, freshly observed trials."""
        from capture_recovery import (recovery_policy, initialize_recovery_budget,
            plan_capture_recovery, observe_recovery_endpoint)
        from gravity_release import review_actual_release
        from contact_insertion import _serial
        if recovery_policy(self.config) is None:
            raise GateFailure("Actual partial capture does not permit gravity release; recovery is disabled")
        budget=initialize_recovery_budget(candidate,self.frames[-1],actual_cumulative_xy)
        self.report["capture_recovery_steps"]=[]
        withdraw=None
        while review["status"]!="admissible_release":
            if budget["steps_completed"]>=3:
                raise GateFailure("Actual capture still does not permit release after three bounded contact trials")
            number=budget["steps_completed"]+1
            label=f"capture_recovery_{number:02d}"
            proposed,plan=plan_capture_recovery(self.rest,self.frames[-1],self.config,candidate,review,
                self.contact_guard_trace[-1],self.joint_trace,self.qs,budget=budget,return_plan=True)
            if proposed.get("kind")!="bounded_capture_recovery_step_v1" \
                    or proposed.get("status")!="candidate" or proposed.get("release_authorized") is not False \
                    or plan is None or not self.must_hold_teddy or not self.must_hold_bag \
                    or float(proposed["observed_time_s"])!=float(self.frames[-1]["time"]) \
                    or any(not np.array_equal(plan.sample(0.)["q"][s],self.qs[s]) for s in ("left","right")):
                raise GateFailure("Invalid fresh capture recovery motion or changed grasp state")
            packet=proposed["insert_stage"]
            if float(packet["duration_s"])!=float(plan.duration) or any(
                    not np.array_equal(np.asarray(packet["command_q_end"][s]),plan.end_q()[s])
                    or not np.array_equal(np.asarray(packet["exact_reference_curve"]["curves"][s]["x"]),plan.curves[s].x)
                    or not np.array_equal(np.asarray(packet["exact_reference_curve"]["curves"][s]["c"]),plan.curves[s].c)
                    for s in ("left","right")):
                raise GateFailure("Capture recovery packet differs from its checked motion")
            (self.out/(label+"_candidate.json")).write_text(json.dumps(_serial(proposed),indent=2,allow_nan=False)+"\n")
            execution={"phase":label,"start_time_s":self.time,"budget_before":dict(budget),
                "status":"executing","physical_success":False,"release_authorized":False,
                "contact_guard_baseline_reset":False,"finite_motor_targets_preserved":dict(self.finger)}
            self.report["capture_recovery_steps"].append(execution)
            try:
                self.move_partial_reference(label,packet)
                budget=observe_recovery_endpoint(proposed,self.frames[-1],budget,self.config)
                next_review,withdraw=review_actual_release(self.rest,self.frames[-1],self.config,proposed,
                    self.contact_guard_trace[-1],self.joint_trace,self.qs,return_plan=True)
                (self.out/(label+"_actual_review.json")).write_text(json.dumps(_serial(next_review),indent=2,allow_nan=False)+"\n")
                execution.update(status="observed",end_time_s=self.time,budget_after=dict(budget),
                    release_authorized=next_review["status"]=="admissible_release")
                self.report["gravity_release_actual_review"]=next_review
                # A continuing trial must improve actual capture; forecasts alone
                # cannot authorize more pushing or release.
                if next_review["status"]!="admissible_release" and (
                        next_review["actual_containment"]["bear_vertex_inside_fraction"]
                        <=review["actual_containment"]["bear_vertex_inside_fraction"]):
                    raise GateFailure("Capture recovery did not improve observed node capture")
                candidate,review=proposed,next_review
            except Exception as error:
                execution.update(status="rejected",error_type=type(error).__name__,error=str(error))
                raise
            finally:
                (self.out/(label+"_execution.json")).write_text(json.dumps(_serial(execution),indent=2,allow_nan=False)+"\n")
        return candidate,review,withdraw

    def run_guarded_contact_trial(self):
        recenter_enabled=self.config.get("task",{}).get("contact_trial",{}).get("recenter_enabled",True)
        if type(recenter_enabled) is not bool:
            raise GateFailure("contact_trial.recenter_enabled must be a boolean")
        from contact_insertion import (plan_guarded_contact_trial,
            reobserve_guarded_contact_trial,recenter_guarded_contact_trial)
        contact_options=self.config.get("task",{}).get("contact_trial",{})
        if "postlift_orientation" in contact_options and "postlift_yaw" in contact_options:
            raise GateFailure("Choose exactly one guarded postlift orientation policy")
        if "postlift_orientation" in contact_options:
            self.begin_guarded_contact_monitor()
            self.run_postlift_orientation()
        elif "postlift_yaw" in contact_options:
            self.begin_guarded_contact_monitor()
            self.run_postlift_yaw()
        choice=plan_guarded_contact_trial(self.rest,self.frames[-1],self.config,
            output=self.out/"guarded_contact_candidate.json",command_qs=self.qs)
        self.report["guarded_contact_candidate"]=choice
        if choice["status"]!="candidate":
            raise GateFailure("No admissible guarded-contact insertion candidate; see guarded_contact_candidate.json")
        self.begin_guarded_contact_monitor()
        labels={"height_at_current_xy":"adjust_teddy_height_contact_trial",
            "transport_y_at_current_x":"transport_teddy_along_y",
            "translate_outside_mouth":"transport_teddy"}
        initial_names=[stage["name"] for stage in choice["stages_to_execute"]]
        strategy=self.config.get("task",{}).get("contact_trial",{}).get("transport_xy_strategy","direct")
        expected_names=["height_at_current_xy","translate_outside_mouth"]
        if strategy=="y_then_x":
            expected_names.insert(1,"transport_y_at_current_x")
        if strategy not in ("direct","y_then_x") or initial_names!=expected_names:
            raise GateFailure("Initial contact route stages disagree with the declared transport strategy")
        initial_start=np.asarray(self.frames[-1]["tcp_right"],float)
        initial_targets=np.asarray([stage["right_tcp_m"] for stage in choice["stages_to_execute"]],float)
        if initial_targets.shape!=(len(expected_names),3) or not np.isfinite(initial_targets).all() \
                or np.linalg.norm(initial_targets[0,:2]-initial_start[:2])>1e-9 \
                or np.max(np.abs(initial_targets[:,2]-initial_targets[0,2]))>1e-9 \
                or (strategy=="y_then_x" and (abs(initial_targets[1,0]-initial_start[0])>1e-9 \
                    or abs(initial_targets[1,1]-initial_targets[-1,1])>1e-9)):
            raise GateFailure("Initial contact transport does not follow its declared measured-state route")
        for stage in choice["stages_to_execute"]:
            self.move_guarded(labels[stage["name"]],stage)
        insertion_strategy=self.config.get("task",{}).get("contact_trial",{}).get("insertion_strategy","vertical")
        if insertion_strategy=="lower_then_center":
            self.run_staged_contact_insertion(choice)
            return
        if insertion_strategy!="vertical":
            raise GateFailure("Unknown guarded contact insertion strategy")
        self.report["guarded_contact_reobservations"]=[]
        self.report["guarded_contact_corrections"]=[]
        cumulative_xy=0.
        actual_cumulative_xy=0.
        for attempt in range(3):
            filename="guarded_contact_reobserved.json" if attempt==0 else f"guarded_contact_reobserved_{attempt:02d}.json"
            fresh=reobserve_guarded_contact_trial(self.rest,self.frames[-1],self.config,choice,
                output=self.out/filename,command_qs=self.qs)
            self.report["guarded_contact_reobserved"]=fresh
            self.report["guarded_contact_reobservations"].append(fresh)
            if fresh["status"]=="candidate":
                break
            if not recenter_enabled:
                raise GateFailure("Fresh contact insertion rejected and recentering is disabled; see "+filename)
            if attempt==2:
                raise GateFailure("Fresh contact insertion remains rejected after two measured XY corrections")
            correction=recenter_guarded_contact_trial(self.rest,self.frames[-1],self.config,choice,
                attempt_index=attempt+1,cumulative_xy_m=cumulative_xy,
                output=self.out/f"guarded_contact_correction_{attempt+1:02d}.json",command_qs=self.qs)
            self.report["guarded_contact_corrections"].append(correction)
            if correction["status"]!="candidate":
                raise GateFailure("No admissible small XY correction from the actual observed state")
            distance=float(correction["correction_distance_m"])
            observed_before=float(correction["cumulative_xy_m_before"])
            new_total=float(correction["cumulative_xy_m_after"])
            stages=correction["stages_to_execute"]
            if [stage["name"] for stage in stages]!=["height_at_current_xy","translate_outside_mouth"]:
                raise GateFailure("A contact correction may only adjust height and translate above the mouth")
            observed_start=np.asarray(self.frames[-1]["tcp_right"],float).copy()
            stage_targets=np.asarray([stage["right_tcp_m"] for stage in stages],float)
            correction_xy=np.asarray(correction["correction_xy_m"],float)
            if stage_targets.shape!=(2,3) or correction_xy.shape!=(2,) or not np.isfinite(stage_targets).all() \
                    or not np.isfinite(correction_xy).all() or np.linalg.norm(stage_targets[0,:2]-observed_start[:2])>1e-9 \
                    or np.linalg.norm(stage_targets[1,:2]-observed_start[:2]-correction_xy)>1e-9:
                raise GateFailure("Contact correction targets disagree with the declared observed-state XY move")
            target_distance=float(np.linalg.norm(stage_targets[1,:2]-observed_start[:2]))
            if not np.isfinite([distance,observed_before,new_total]).all() or not 0.<=distance<=.030000001 \
                    or abs(distance-target_distance)>1e-9 or abs(observed_before-actual_cumulative_xy)>1e-9 \
                    or new_total>.050000001 or abs(new_total-observed_before-distance)>1e-9:
                raise GateFailure("Contact correction violated the bounded XY travel budget")
            for stage in stages:
                label=f"recenter_height_{attempt+1}" if stage["name"]=="height_at_current_xy" else f"recenter_teddy_xy_{attempt+1}"
                self.move_guarded(label,stage)
            observed_distance=float(np.linalg.norm(np.asarray(self.frames[-1]["tcp_right"])[:2]-observed_start[:2]))
            actual_cumulative_xy+=observed_distance
            correction["runtime_observed_endpoint_displacement_m"]=observed_distance
            correction["runtime_observed_cumulative_endpoint_displacement_m"]=actual_cumulative_xy
            correction["runtime_distance_scope"]="Sum of measured correction endpoint XY displacements, not the full oscillatory TCP arc length"
            if observed_distance>.030000001 or actual_cumulative_xy>.050000001:
                raise GateFailure("Observed correction endpoint displacement exceeded its XY budget")
            cumulative_xy=new_total
            choice=correction
        from gravity_release import release_policy, plan_gravity_release, review_actual_release
        gravity_policy=release_policy(self.config)
        gravity_candidate=None
        gravity_plan=None
        gravity_withdraw=None
        stage=fresh["insert_stage"]
        if gravity_policy is not None:
            from contact_insertion import _serial
            gravity_candidate,gravity_plan=plan_gravity_release(self.rest,self.frames[-1],self.config,fresh,
                self.qs,return_plan=True)
            if gravity_candidate.get("status")!="candidate" or gravity_plan is None:
                raise GateFailure("Gravity-release policy requires its accepted exact checked curve")
            self.report["gravity_release_candidate"]=gravity_candidate
            (self.out/"gravity_release_candidate.json").write_text(json.dumps(_serial(gravity_candidate),indent=2,allow_nan=False)+"\n")
            stage=gravity_candidate["insert_stage"]
        if gravity_plan is None:
            self.move_guarded("insert_teddy",stage)
        else:
            if not gravity_candidate["continuous_reference_zero_endpoint_velocity_acceleration"] \
                    or any(not np.array_equal(gravity_plan.sample(0.)["q"][s],self.qs[s]) for s in ("left","right")):
                raise GateFailure("Gravity-release curve does not preserve the existing command endpoint")
            gravity_plan.finger_start=self.finger.copy()
            gravity_plan.finger_end=self.finger.copy()
            np.savez_compressed(self.out/"gravity_insert_plan.npz",duration=gravity_plan.duration,
                left_x=gravity_plan.curves["left"].x,left_c=gravity_plan.curves["left"].c,
                right_x=gravity_plan.curves["right"].x,right_c=gravity_plan.curves["right"].c,
                finger_left=self.finger["left"],finger_right=self.finger["right"])
            self.execute("insert_teddy",gravity_plan)
            self.rotations["right"]=Rotation.from_quat(stage["rotation_xyzw"])
        if gravity_candidate is not None:
            review,gravity_withdraw=review_actual_release(self.rest,self.frames[-1],self.config,gravity_candidate,
                self.contact_guard_trace[-1],self.joint_trace,self.qs,return_plan=True)
            self.report["gravity_release_actual_review"]=review
            self.report["gravity_release_initial_review"]=review
            (self.out/"gravity_release_actual_review.json").write_text(json.dumps(_serial(review),indent=2,allow_nan=False)+"\n")
            if review["status"]!="admissible_release":
                gravity_candidate,review,gravity_withdraw=self.run_capture_recovery(
                    gravity_candidate,review,actual_cumulative_xy)
                self.report["effective_release_candidate"]=gravity_candidate
            if review["status"]!="admissible_release" or gravity_withdraw is None:
                raise GateFailure("Actual partial capture does not permit gravity release; see gravity_release_actual_review.json")
        self.must_hold_teddy=False
        self.hold("release_teddy",.5,right=.04)
        if gravity_withdraw is None:
            self.move_guarded("withdraw_right_arm",{
                "right_tcp_m":fresh["outside_tcp_m"],
                "rotation_xyzw":fresh["insert_stage"]["rotation_xyzw"],"duration_s":4.})
        else:
            if any(not np.array_equal(gravity_withdraw.sample(0.)["q"][s],self.qs[s]) for s in ("left","right")):
                raise GateFailure("Gravity-release withdrawal changed its checked command start")
            gravity_withdraw.finger_start=self.finger.copy()
            gravity_withdraw.finger_end=self.finger.copy()
            np.savez_compressed(self.out/"gravity_withdraw_plan.npz",duration=gravity_withdraw.duration,
                left_x=gravity_withdraw.curves["left"].x,left_c=gravity_withdraw.curves["left"].c,
                right_x=gravity_withdraw.curves["right"].x,right_c=gravity_withdraw.curves["right"].c,
                finger_left=self.finger["left"],finger_right=self.finger["right"])
            self.execute("withdraw_right_arm",gravity_withdraw)
        self.hold("final_settle",gravity_policy["settle_seconds"] if gravity_policy is not None else 2.)
        self.report["sequence_completed"]=True

    def orient_held_bag(self,target):
        """Reorient the held strap by real joint motion while both hands grasp it."""
        if self.left_hold_rotation is None:
            return
        duration=float(self.config["task"].get("left_hold_rotation_duration_s",4.))
        if not np.isfinite(duration) or duration<4.:
            raise GateFailure("Held-bag reorientation requires a finite duration of at least 4 seconds")
        observed_fingers={side:float(np.clip(robot.get_qpos().cpu().numpy().reshape(-1)[7:].mean(),0.,.04))
            for side,robot in self.robots.items()}
        plan=self.planner.trajectory(self.qs,{"left":target},duration=duration,
            rotations={"left":self.left_hold_rotation},finger_start=observed_fingers,
            finger_end=observed_fingers,clearance_margin=.008,
            max_joint_velocity=.25,max_joint_acceleration=.4)
        plan.finger_start=self.finger.copy()
        plan.finger_end=self.finger.copy()
        self.report["held_bag_reorientation"]={
            "target_left_tcp_m":np.asarray(target).tolist(),
            "target_rotation_xyzw":self.left_hold_rotation.as_quat().tolist(),
            "physical_pd_motion":True,"both_handle_motor_targets_remain_closed":True}
        self.execute("orient_held_bag",plan)
        self.rotations["left"]=self.left_hold_rotation
        self.hold("settle_oriented_bag",.5)
        proximity={side:self.near_material(self.bag,side) for side in self.robots}
        self.report["oriented_handle_proximity"]=proximity
        if any(row["nearest_m"]>.018 for row in proximity.values()):
            raise GateFailure("A cloth handle slipped during physical held-bag reorientation")

    def run(self):
        if self.args.task=="settle":
            self.hold("settle",self.args.settle_seconds)
            return
        if self.args.task in ("grasp","grasp_rotate"):
            self.hold("grasp_probe_settle",.5,left=0.)
            center=np.asarray(self.config.get("task",{}).get("bear_grasp_local",[.08,-.014,.034]))
            rest=self.rest["bear"]-np.asarray(self.config["bear"]["position"])
            arm=np.where(np.linalg.norm(rest-center,axis=1)<self.bear_grasp_tracking_radius)[0]
            if len(arm)<3: raise GateFailure("No narrow teddy upper-arm grasp region")
            target=np.asarray(self.config["bear"]["position"])+center+(positions(self.bear)[arm]-self.rest["bear"][arm]).mean(0)
            target[2]=max(target[2],self.config["table"]["height"]+.025)
            self.move("teddy_grasp_approach",{"right":target},1.5,
                rotations={"right":self.bear_grasp_rotation})
            self.close_teddy_gripper()
            self.move("lift_teddy",{"right":[target[0],target[1],1.17]},3.)
            self.hold("grasp_hold",float(self.config.get("task",{}).get("grasp_probe_hold_seconds",1.)))
            if self.args.task=="grasp_rotate":
                self.move("grasp_rotate",{"right":[target[0],target[1],1.17]},6.,
                    rotations={"right":Rotation.from_euler("x",90,degrees=True)})
                self.hold("rotated_grasp_hold",1.)
            self.report["grasp_probe_lift_minimum_z"]=float(positions(self.bear)[:,2].min())
            if positions(self.bear)[:,2].min()<self.config["table"]["height"]+.04:
                raise GateFailure("Teddy friction grasp did not lift it clear of the table")
            self.report["grasp_probe_lifted"]=True
            return
        if self.flat_start_mode:
            run_flat_pickup(self)
            if self.args.task in ("open","lift"):
                return
        else:
            self.hold("close_both_handles",float(self.config.get("task",{}).get("handle_close_seconds",.5)),left=0.,right=0.)
            evidence={s:self.near_material(self.bag,s) for s in self.robots}
            self.report["initial_handle_proximity"]=evidence
            if any(x["near_vertices"]<2 for x in evidence.values()):
                raise GateFailure("Handle slipped away before physical closure")
            self.hold("drape_under_gravity",.6)
            self.report["settled_handle_proximity"]={s:self.near_material(self.bag,s) for s in self.robots}
            if any(v["nearest_m"]>.018 for v in self.report["settled_handle_proximity"].values()):
                raise GateFailure("A handle slipped out of the finger pads while settling")
            self.must_hold_bag=True
            self.must_hold_right_handle=True
            open_targets=self.config.get("task",{}).get("open_targets",{"left":[-.25,-.13,1.25],"right":[-.10,.13,1.25]})
            self.move("spread_bag",open_targets,2.5)
            self.report["spread_handle_proximity"]={s:self.near_material(self.bag,s) for s in self.robots}
            if any(x["near_vertices"]<2 for x in self.report["spread_handle_proximity"].values()):
                raise GateFailure("One of the handles is not held after spreading")
            self.orient_held_bag(open_targets["left"])
            if self.args.task=="open": return
            self.must_hold_right_handle=False
            self.hold("release_right_handle",.4,right=.04)
            self.move("right_clear_handle",{"right":[.12,.20,1.32]},2.,rotations={"right":DOWN})
            if self.args.task=="handoff":
                self.hold("settle_handoff",10.)
                self.report["handoff_probe_completed"]=True
                self.report["handoff_probe_scope"]="Native cloth opening and one-hand hold only; no teddy grasp or insertion attempted."
                return
        # Follow the actual soft upper arm, not a scripted teddy pose.
        rest=self.rest["bear"]-np.asarray(self.config["bear"]["position"])
        center=np.asarray(self.config.get("task",{}).get("bear_grasp_local",[.08,-.014,.034]))
        arm=np.where(np.linalg.norm(rest-center,axis=1)<self.bear_grasp_tracking_radius)[0]
        if len(arm)<3: raise GateFailure("No narrow teddy upper-arm grasp region")
        target=np.asarray(self.config["bear"]["position"])+center+(positions(self.bear)[arm]-self.rest["bear"][arm]).mean(0)
        target[2]=max(target[2],self.config["table"]["height"]+.025)
        self.report["teddy_grasp_target"]=target.tolist()
        self.move("teddy_pregrasp",{"right":target+[0,0,.15]},3.,
            rotations={"right":self.bear_grasp_rotation})
        self.move("teddy_grasp_approach",{"right":target},2.,
            rotations={"right":self.bear_grasp_rotation})
        self.close_teddy_gripper()
        if "pickup_uprighting" in self.config.get("task",{}):
            self.begin_guarded_contact_monitor()
            from pickup_uprighting import upright_pickup
            upright_pickup(self)
        self.move("lift_teddy",{"right":[target[0],target[1],self.bear_lift_height]},4.)
        bear=positions(self.bear)
        self.report["lift_minimum_z"]=float(bear[:,2].min())
        if bear[:,2].min()<self.config["table"]["height"]+.04:
            raise GateFailure("Teddy did not lift clear of the table; real friction grasp failed")
        if self.config.get("task",{}).get("contact_trial",{}).get("enabled",False):
            self.run_guarded_contact_trial()
            return
        from insertion_candidates import (choose_insertion_route,reobserve_insertion,
            replan_after_orientation,NoInsertionRoute)
        phase_names={"height_at_current_xy":"adjust_teddy_height",
            "rotate_held_teddy":"orient_teddy","translate_outside_mouth":"transport_teddy"}
        # A head-end grasp can hang naturally in a suitable pose. Test the
        # actual pose before requesting a large wrist turn. Every correction
        # is followed by a fresh soft-body observation before transport.
        try:
            alignment=replan_after_orientation(self.rest,self.frames[-1],self.config,
                output=self.out/"alignment_initial.json")
        except NoInsertionRoute:
            choice=choose_insertion_route(self.rest,self.frames[-1],self.config,
                output=self.out/"insertion_candidates.json")
            route=choice["route"]
            for stage in route["stages"][:2]:
                self.move(phase_names[stage["name"]],{"right":stage["right_tcp_m"]},
                    stage["duration_s"],rotations={"right":Rotation.from_quat(stage["rotation_xyzw"])})
            alignment=replan_after_orientation(self.rest,self.frames[-1],self.config,route,
                output=self.out/"alignment_after_initial_orientation.json")
        self.report["observed_shape_alignment"]=[]
        iteration=0
        while True:
            self.report["observed_shape_alignment"].append(alignment)
            for stage in alignment["stages_to_execute"]:
                label=phase_names[stage["name"]]
                if stage["name"]!="translate_outside_mouth": label+=f"_alignment_{iteration}"
                self.move(label,{"right":stage["right_tcp_m"]},stage["duration_s"],
                    rotations={"right":Rotation.from_quat(stage["rotation_xyzw"])})
            route=alignment["route"]
            if alignment["next_observation"]=="before_insertion": break
            iteration+=1
            alignment=replan_after_orientation(self.rest,self.frames[-1],self.config,alignment,
                attempt=alignment["next_attempt"],max_attempts=alignment["max_attempts"],
                output=self.out/f"alignment_observation_{iteration}.json")
        self.report["selected_insertion_candidate"]=route
        fresh=reobserve_insertion(self.rest,self.frames[-1],self.config,route,
            output=self.out/"reobserved_insertion_candidates.json")
        self.report["reobserved_insertion"]=fresh
        insert=fresh["insert_stage"]
        self.move("insert_teddy",{"right":insert["right_tcp_m"]},insert["duration_s"],
            rotations={"right":Rotation.from_quat(insert["rotation_xyzw"])})
        self.must_hold_teddy=False
        self.hold("release_teddy",.5,right=.04)
        self.move("withdraw_right_arm",{"right":fresh["outside_tcp_m"]},3.)
        self.hold("final_settle",2.)
        self.report["sequence_completed"]=True

    def finish(self,error=None):
        self.writer.close()
        self.report.update(status="failed" if error else "completed",error=error,
            simulated_seconds=self.time,wall_seconds=time.monotonic()-self.wall,
            events=self.events)
        frames={k:np.asarray([f[k] for f in self.frames]) for k in self.frames[0]}
        trace=np.asarray(self.joint_trace)
        np.savez_compressed(self.out/"states.npz",**frames)
        np.save(self.out/"joint_trace.npy",trace)
        np.save(self.out/"soft_metrics.npy",np.asarray(self.soft_metrics))
        (self.out/"soft_metrics_columns.json").write_text(json.dumps(self.physical_monitor.columns,indent=2))
        (self.out/"events.json").write_text(json.dumps(self.events,indent=2))
        (self.out/"handle_grip_trace.json").write_text(json.dumps(self.handle_grip_trace,indent=2))
        if self.staged_hand_monitor is not None:
            (self.out/"staged_hand_trace.json").write_text(json.dumps(self.staged_hand_trace,indent=2))
        if self.contact_guard is not None:
            self.report["contact_guard"]=self.contact_guard.report()
            self.report["contact_guard"]["minimum_right_opening_reaction_until_release_n"]=.1
            self.report["contact_guard"]["holding_distances_checked_every_physics_step"]=True
            (self.out/"contact_guard_trace.json").write_text(json.dumps(self.contact_guard_trace,indent=2))
        if len(trace)>3:
            from validate import audit_arrays
            try:
                audit=audit_arrays(self.rest,frames,self.config,
                    task_completed=bool(self.report.get("sequence_completed",False)),
                    joint_trace=trace,events=self.events,soft_metrics=np.asarray(self.soft_metrics))
                (self.out/"validation.json").write_text(json.dumps(audit,indent=2))
                self.report["task_passed"]=bool(audit.get("task_success",False))
                if self.flat_start_mode and self.args.task in ("open","lift"):
                    checks=audit.get("checks",{})
                    full_only={"bear_fully_inside_bag_final","bear_remains_inside_during_final_hold_samples",
                        "requested_phases_recorded_in_order","bear_lifted_clear_of_table",
                        "right_fingers_open_after_release","controller_explicitly_reports_task_complete"}
                    probe_checks={key:value for key,value in checks.items() if key not in full_only}
                    native_rows=[r for r in self.handle_grip_trace if r.get("single_handle_hold")]
                    reaction_ok=bool(native_rows and all(np.isfinite(r["left_opening_reactions_n"]).all()
                        and min(r["left_opening_reactions_n"])>.1 for r in native_rows))
                    self.report["single_handle_lift_probe"]={"checks":probe_checks,
                        "native_two_sided_left_contact_retained":reaction_ok,
                        "native_contact_samples":len(native_rows),
                        "scope":"Flat initial state, single left grasp/lift/hold; teddy insertion not attempted"}
                    self.report["single_handle_lift_probe_passed"]=bool(
                        self.report.get("single_handle_lift_completed",False) and probe_checks
                        and all(probe_checks.values()) and reaction_ok)
                if self.args.task=="handoff":
                    checks=audit.get("checks",{})
                    physical_keys=("finite_saved_states","per_step_soft_metrics_valid_when_supplied",
                        "actual_open_bag_topology","two_cloth_panels_with_direct_u_seam_and_no_bottom_or_side_panels",
                        "no_tetrahedron_inversion","no_tetrahedron_volume_below_5_percent",
                        "table_penetration_below_2mm","cloth_edge_stretch_below_15_percent",
                        "no_final_sampled_bag_bear_penetration_over_1mm",
                        "joint_motion_within_configured_limits","motion_sample_interval_at_most_20ms",
                        "left_gripper_holds_pose_within_15mm_after_handoff",
                        "named_left_handle_material_within_18mm_after_handoff",
                        "both_left_fingers_command_clamping_after_handoff")
                    self.report["handoff_probe_passed"]=bool(self.report.get("handoff_probe_completed",False)
                        and all(checks.get(k,False) for k in physical_keys)
                        and np.min(trace[-1,17:19])>=.038)
                if self.args.task in ("grasp","grasp_rotate"):
                    checks=audit.get("checks",{})
                    self.report["grasp_probe_passed"]=bool(self.report.get("grasp_probe_lifted",False)
                        and all(checks.get(k,False) for k in ("finite_saved_states",
                        "per_step_soft_metrics_valid_when_supplied","no_tetrahedron_inversion",
                        "no_tetrahedron_volume_below_5_percent","table_penetration_below_2mm",
                        "cloth_edge_stretch_below_15_percent",
                        "joint_motion_within_configured_limits","motion_sample_interval_at_most_20ms")))
            except Exception as exc:
                self.report["validation_error"]=f"{type(exc).__name__}: {exc}"
        # Completing a motion script is not equivalent to passing its physical
        # acceptance checks. Preserve failed validation in both report and exit.
        scope_passed=not error and "validation_error" not in self.report
        if self.args.task=="full":
            scope_passed=scope_passed and self.report.get("task_passed",False)
        elif self.flat_start_mode and self.args.task in ("open","lift"):
            scope_passed=scope_passed and self.report.get("single_handle_lift_probe_passed",False)
        elif self.args.task=="handoff":
            scope_passed=scope_passed and self.report.get("handoff_probe_passed",False)
        elif self.args.task in ("grasp","grasp_rotate"):
            scope_passed=scope_passed and self.report.get("grasp_probe_passed",False)
        self.report["requested_task_passed"]=bool(scope_passed)
        if not error and not scope_passed:
            self.report["status"]="validation_failed"
        from contact_insertion import _serial
        (self.out/"report.json").write_text(json.dumps(_serial(self.report),indent=2,allow_nan=False))
        (self.out/"coupling_trace.json").write_text(json.dumps(self.coupling_trace,indent=2))
        print("OUTPUT "+str(self.out),flush=True)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",choices=("settle","open","lift","handoff","grasp","grasp_rotate","full"),default="full")
    ap.add_argument("--viewer",action="store_true")
    ap.add_argument("--config")
    ap.add_argument("--output")
    ap.add_argument("--width",type=int,default=960)
    ap.add_argument("--height",type=int,default=720)
    ap.add_argument("--max-seconds",type=float,default=0)
    ap.add_argument("--settle-seconds",type=float,default=.5)
    args=ap.parse_args()
    exp=Experiment(read_config(args.config),args)
    error=None
    try:
        exp.run()
    except (Exception, KeyboardInterrupt) as exc:
        error=f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    finally:
        exp.finish(error)
    if error or not exp.report.get("requested_task_passed",False): raise SystemExit(1)


if __name__=="__main__": main()
