"""Native Genesis IPC/FEM tabletop scene. No soft-body pins or scripted vertices."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import genesis as gs


def read_config(path=None):
    return json.loads(Path(path or HERE / "config.json").read_text())


def build(config, *, viewer=False, width=960, height=720, robots=True, initial_q=None):
    gs.init(backend=gs.cpu, precision=config["simulation"]["precision"],
            seed=config["seed"], logging_level="warning")
    sim = config["simulation"]
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=sim["dt"], substeps=sim["substeps"]),
        rigid_options=gs.options.RigidOptions(enable_self_collision=True,
            enable_neutral_collision=True,enable_adjacent_collision=False),
        coupler_options=gs.options.IPCCouplerOptions(
            contact_d_hat=sim["contact_distance"], contact_resistance=sim["contact_resistance"],
            contact_friction_enable=True, contact_eps_velocity=.001,
            newton_max_iterations=100, newton_tolerance=.005,
            newton_translation_tolerance=.1, newton_semi_implicit_enable=False,
            n_linesearch_iterations=16, linear_system_tolerance=1e-4,
            two_way_coupling=True, sanity_check_enable=True,
            enable_rigid_rigid_contact=False, enable_rigid_ground_contact=False,
            constraint_strength_translation=100., constraint_strength_rotation=100.),
        viewer_options=gs.options.ViewerOptions(res=(width, height),
            camera_pos=config["camera"]["position"], camera_lookat=config["camera"]["lookat"],
            camera_fov=config["camera"]["fov"], refresh_rate=30),
        vis_options=gs.options.VisOptions(ambient_light=(.35, .35, .35)),
        show_viewer=viewer,
    )
    def box(name, pos, size, color, contact=True):
        return scene.add_entity(name=name, morph=gs.morphs.Box(pos=pos, size=size, fixed=True),
            material=gs.materials.Rigid(coup_type="ipc_only" if contact else None, needs_coup=contact, coup_friction=.65),
            surface=gs.surfaces.Default(color=color, roughness=.9))

    th = config["table"]["height"]
    sx, sy, sz = config["table"]["size"]
    box("tabletop", (0,0,th-sz/2), (sx,sy,sz), (.50,.38,.28,1))
    for x in (-sx/2+.10, sx/2-.10):
        for y in (-sy/2+.10, sy/2-.10):
            box(f"leg_{x}_{y}", (x,y,(th-sz)/2), (.06,.06,th-sz), (.17,.20,.23,1))
    scene.add_entity(name="floor", morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(needs_coup=True,coup_type="ipc_only",coup_friction=.65),
        surface=gs.surfaces.Default(color=(.16,.19,.22,1)))

    arms = {}
    if robots:
        for side in ("left", "right"):
            rc = config["robots"]
            bp, yaw = rc[side+"_base"], rc[side+"_yaw"]
            box(side+"_mount", (bp[0],bp[1],(th+bp[2])/2), (.19,.19,bp[2]-th), (.24,.28,.31,1))
            arms[side] = scene.add_entity(name=side+"_franka",
                morph=gs.morphs.MJCF(file=rc.get("mjcf_path",str(HERE/"assets/panda_tactile.xml")),
                    pos=bp, quat=(np.cos(yaw/2),0,0,np.sin(yaw/2))),
                material=gs.materials.Rigid(coup_type="two_way_soft_constraint",
                    coup_friction=.85, friction=.7, gravity_compensation=1.0))

    bc = config["bag"]
    bag = scene.add_entity(name="open_canvas_tote",
        morph=gs.morphs.Mesh(file=str(HERE/"assets"/bc.get("mesh","bag.obj")), pos=bc["position"],
            file_meshes_are_zup=True, decimate=False, convexify=False),
        material=gs.materials.FEM.Cloth(**{k:bc[k] for k in
            ("E","nu","rho","thickness","bending_stiffness","friction_mu")}),
        surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(
            image_path=str(HERE/"assets/cotton_canvas.png")), roughness=1., double_sided=True))
    bc = config["bear"]
    bear_path = HERE/"assets"/bc.get("mesh","bear_physics.obj")
    bear = scene.add_entity(name="free_soft_teddy",
        morph=gs.morphs.Mesh(file=str(bear_path), pos=bc["position"],
            file_meshes_are_zup=True, decimate=False, convexify=False,
            maxvolume=bc.get("maxvolume",4e-6), mindihedral=bc.get("mindihedral",8),
            minratio=bc.get("minratio",1.3),nobisect=bc.get("nobisect",True)),
        material=gs.materials.FEM.Elastic(model="stable_neohookean",
            **{k:bc[k] for k in ("E","nu","rho","friction_mu")}),
        surface=gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(
            image_path=str(HERE/"assets/teddy_fur.png")), roughness=1.))
    cameras = {
        "overview":scene.add_camera(res=(width,height), GUI=False, pos=config["camera"]["position"],
            lookat=config["camera"]["lookat"], fov=config["camera"]["fov"]),
        "detail":scene.add_camera(res=(width,height), GUI=False, pos=(.7,-1.05,1.9),
            lookat=(-.02,-.04,1.0), fov=43),
    }
    from flat_start import enabled as flat_start_enabled
    if flat_start_enabled(config):
        from initial_views import add_initial_cameras
        cameras.update(add_initial_cameras(scene))
    from physics_config import initialize_before_ipc, configure_shell_shear, install_exact_stc_feedback, install_rigid_collision_fallback
    initial_q = initial_q or {s:np.r_[config["robots"]["home"],.04,.04] for s in arms}
    scene.teddy_initialization_report=initialize_before_ipc(scene,arms,initial_q)
    scene.teddy_shell_report=configure_shell_shear(scene,
        shear_stiffness_n_per_m=config["bag"].get("shear_stiffness",20.),
        handle_shear_stiffness_n_per_m=config["bag"].get("handle_shear_stiffness"),
        handle_source_npz=HERE/"assets"/config["bag"].get("npz_file","bag.npz"),
        handle_source_position=config["bag"]["position"], handle_entity=bag)
    scene.teddy_stc_feedback_report=install_exact_stc_feedback(scene) if sim.get("correct_stc_reaction",True) else {"enabled":False}
    scene.teddy_rigid_contact_report=install_rigid_collision_fallback(scene)
    print("BUILD_START", flush=True)
    scene.build()
    print("BUILD_DONE", flush=True)
    for side, robot in arms.items():
        rc = config["robots"]
        q = np.asarray(initial_q[side] if initial_q else [*rc["home"], .04,.04], dtype=float)
        robot.set_dofs_kp([4500,4500,3500,3500,2000,2000,2000,rc["finger_kp"],rc["finger_kp"]])
        robot.set_dofs_kv([180,180,150,150,90,90,90,rc["finger_kv"],rc["finger_kv"]])
        limit=np.asarray([87,87,87,87,12,12,12,rc["finger_force_limit"],rc["finger_force_limit"]])
        robot.set_dofs_force_range(-limit,limit)
        robot.control_dofs_position(q)
    return scene, arms, bag, bear, cameras


def positions(entity):
    return entity.get_state().pos.detach().cpu().numpy().reshape(-1,3)


def physics_metadata(config, bag, bear):
    return {"solver":"Genesis 1.4.3 + libuipc 0.0.29, IPC contact, FEM shell/volume",
        "bag_vertices":bag.n_vertices,"bear_vertices":bear.n_vertices,
        "bear_tetrahedra":bear.n_elements,
        "no_fixed_soft_vertices":True,"no_soft_grasp_attachments":True,
        "grasp_model":"finite-force finger PD and IPC friction, no ideal attachment",
        "robot_soft_contact_links":"all collision links",
        "rigid_contact_solver":"Genesis rigid solver with scene-local restoration of IPC-delegated collision pairs",
        "soft_environment_contact":"tabletop, legs, mounts and floor participate in native IPC contact",
        "other_arm_links_soft_contact":True,
        "two_way_contact_coupling":True,"material_experimentally_calibrated":False,
        "bear_material_scope":"homogeneous compressible elastic filling approximation; no separate textile skin",
        "bag_material_scope":"two cloth panels sewn on three edges; thin shell with separate effective stretch, shear and bending stiffness; no yarn model",
        "handle_material_scope":"Optional handle_shear_stiffness changes only the two sewn fabric strips' effective shear; uncalibrated, with unchanged stretch, bending, thickness and mass",
        "cloth_thickness_convention":"IPC thickness is a radius; physical thickness is twice the configured value",
        "bag":config["bag"],"bear":config["bear"],"simulation":config["simulation"]}
