"""Scene-local Genesis 1.4.3 / pyuipc 0.0.29 compatibility and diagnostics.

Only the supplied scene instance is changed. No engine installation is edited.
"""
from __future__ import annotations

from importlib.metadata import version
import numpy as np


def install_rigid_collision_fallback(scene):
    """Keep rigid contacts in Genesis when IPC's rigid contact pairs are off.

    Genesis 1.4.3 excludes every ipc_only link and pairs of IPC-delegated links
    from its rigid collider even when native IPC rigid-rigid contact is off.
    This scene-local Collider subclass hides only the delegation tag while
    the ordinary Genesis pair builder runs. All original adjacency, masks,
    fixed-body, weld and neutral-overlap filters remain in use. Materials are
    restored before any IPC geometry is built; native rigid contacts stay off.

    Only fixed ipc_only entities and two-way soft-constraint robot entities
    are supported. Dynamic ipc_only bodies need a different contact scheme.
    """
    if scene.is_built or version("genesis-world") != "1.4.3":
        raise RuntimeError("Rigid-contact fallback requires an unbuilt Genesis 1.4.3 scene")
    from genesis.engine.couplers import IPCCoupler
    from genesis.engine.solvers.rigid.collider.collider import Collider
    coupler, rigid = scene.sim.coupler, scene.sim.rigid_solver
    if not isinstance(coupler, IPCCoupler):
        raise RuntimeError("Rigid-contact fallback is only defined for the IPC coupler")
    if coupler.options.enable_rigid_rigid_contact or coupler.options.enable_rigid_ground_contact:
        raise RuntimeError("Native IPC rigid-rigid and rigid-ground contact must both remain disabled")
    if getattr(rigid, "_teddy_bag_rigid_collision_adapter", False):
        raise RuntimeError("Rigid-contact fallback already installed on this scene")
    report = {"enabled": True, "verified": False,
              "method": "Scene-local Collider subclass; original Genesis pair filters with temporary delegation tags hidden",
              "engine_installation_modified": False, "global_classes_modified": False,
              "native_ipc_rigid_rigid_contact": False, "native_ipc_rigid_ground_contact": False,
              "materials_restored": False, "before": {}, "after": {}, "required_entity_pairs": []}

    def count_pairs(pairs):
        counts = {}
        for ga, gb in pairs:
            key = " | ".join(sorted((rigid.geoms[int(ga)].entity.name,
                                     rigid.geoms[int(gb)].entity.name)))
            counts[key] = counts.get(key, 0) + 1
        return {"total_geometry_pairs": len(pairs), "geometry_pairs_by_entity_pair": counts}

    class SceneRigidCollider(Collider):
        def _compute_collision_pair_idx(self):
            if coupler.options.enable_rigid_rigid_contact or coupler.options.enable_rigid_ground_contact:
                raise RuntimeError("IPC rigid contacts changed after installing the Genesis fallback")
            baseline = super()._compute_collision_pair_idx()
            material_modes = {}
            moving, fixed = [], []
            for entity in rigid.entities:
                material = entity.material
                if not material.needs_coup or material.coup_type is None:
                    continue
                mode = material.coup_type
                if mode not in ("two_way_soft_constraint", "ipc_only"):
                    raise RuntimeError(f"Unsupported coupling mode in rigid fallback: {mode}")
                all_fixed = all(link.is_fixed for link in entity.links)
                if mode == "ipc_only" and not all_fixed:
                    raise RuntimeError("Dynamic ipc_only entities cannot use this rigid-contact fallback")
                material_modes[id(material)] = (material, mode)
                (fixed if all_fixed else moving).append(entity.name)
            try:
                for material, _ in material_modes.values():
                    material.coup_type = None
                corrected = super()._compute_collision_pair_idx()
            finally:
                for material, mode in material_modes.values():
                    material.coup_type = mode
                report["materials_restored"] = all(material.coup_type == mode
                    for material, mode in material_modes.values())
            before, after = baseline[2], corrected[2]
            report["before"], report["after"] = count_pairs(before), count_pairs(after)
            retained = {tuple(pair) for pair in before}.issubset({tuple(pair) for pair in after})
            required = {" | ".join(sorted((a, b))) for i, a in enumerate(moving)
                        for b in moving[i+1:] + fixed}
            counts = report["after"]["geometry_pairs_by_entity_pair"]
            report["required_entity_pairs"] = sorted(required)
            report["all_original_pairs_retained"] = retained
            report["restored_delegation_modes"] = [mode for _, mode in material_modes.values()]
            report["verified"] = bool(report["materials_restored"] and retained
                and all(counts.get(key, 0) > 0 for key in required))
            if not report["verified"]:
                raise RuntimeError(f"Rigid-contact fallback did not restore required entity pairs: {report}")
            return corrected

    def initialize_local_collider():
        rigid.collider = SceneRigidCollider(rigid)

    rigid._init_collider = initialize_local_collider
    rigid._teddy_bag_rigid_collision_adapter = True
    return report


def stc_reaction_wrench(ipc_transform, aim_transform, mass, first_moment,
                        second_moment, translation_scale, rotation_scale):
    """Exact rigid-target virtual-work pullback of pyuipc 0.0.29 STC energy.

    Scales are eta_p/h^2 and eta_a/h^2. The native ABD mass moments use the
    link's original reference frame, including a nonzero centre of mass.
    Return world force and world torque *about the aim link origin*. Retain
    the full affine A: polar projection would change the energy derivative.
    """
    ipc, aim = np.asarray(ipc_transform), np.asarray(aim_transform)
    mc, second = np.asarray(first_moment).reshape(3), np.asarray(second_moment).reshape(3, 3)
    m = float(mass)
    if not np.isfinite(m) or m <= 0:
        raise ValueError("STC feedback needs a finite positive native ABD mass")
    dp, da = ipc[:3, 3]-aim[:3, 3], ipc[:3, :3]-aim[:3, :3]
    effective_second = rotation_scale*second + (translation_scale-rotation_scale)*np.outer(mc, mc)/m
    force = translation_scale*(m*dp + da@mc)
    ga = translation_scale*np.outer(dp, mc) + da@effective_second
    torque = np.cross(aim[:3, :3].T, ga.T).sum(axis=0)
    return force, torque


def install_exact_stc_feedback(scene):
    """Opt in to scene-local, exact STC reaction; never patch installed code.

    Genesis 1.4.3's default approximation uses link-origin displacement and
    MJCF mass/inertia. libuipc's STC instead uses its ABD mass moments and COM.
    This adapter replaces that one scene's feedback callback with the actual
    energy derivative. Forces act at the Genesis link origin; the torque
    already includes the COM lever arm, so no extra r-cross-F is added.
    Only two_way_soft_constraint, nonfixed links receive reactions. A native
    animator substep of 1 is required so aim_transforms is the actual target.
    """
    if scene.is_built or version("genesis-world")!="1.4.3" or version("pyuipc")!="0.0.29":
        raise RuntimeError("Exact STC adapter requires an unbuilt, version-matched scene")
    import uipc
    from genesis.engine.couplers.ipc_coupler.data import COUPLING_TYPE
    coupler=scene.sim.coupler
    report={"method":"Exact native STC energy gradient, rigid-target virtual work",
            "enabled":True,"applied_steps":0,"links":[],
            "engine_installation_modified":False,"mass_source":"native IPC ABD dyadic moments",
            "wrench_reference":"world axes, torque about target link origin",
            "source":"libuipc v0.0.29 soft_transform_constraint.cu"}
    cache=None

    def exact_feedback():
        nonlocal cache
        if not coupler.options.two_way_coupling:
            return
        data=coupler._coupling_data
        if data is None:
            return
        if coupler._ipc_animator.substep()!=1:
            raise RuntimeError("Exact STC feedback requires native animator substep=1")
        if cache is None:
            cache=[]
            for index,link in enumerate(data.links):
                if link.is_fixed or coupler._coup_type_by_entity.get(link.entity)!=COUPLING_TYPE.TWO_WAY_SOFT_CONSTRAINT:
                    continue
                for env,slot in enumerate(coupler._abd_slots_by_link[link]):
                    meta=slot.geometry().meta()
                    m=float(np.asarray(uipc.view(meta.find("abd_mass"))).reshape(-1)[0])
                    mc=np.asarray(uipc.view(meta.find("abd_mass_x_bar"))).reshape(3).copy()
                    second=np.asarray(uipc.view(meta.find("abd_mass_x_bar_x_bar"))).reshape(3,3).copy()
                    if not np.isfinite(np.r_[m,mc,second.ravel()]).all() or m<=0:
                        raise RuntimeError("Invalid native ABD moments for exact STC feedback")
                    cache.append((env,index,m,mc,second))
                    report["links"].append({"entity":link.entity.name,"link":link.name,
                        "environment":env,"ipc_mass_kg":m,
                        "genesis_mass_kg":float(data.links_mass[index]),
                        "ipc_local_com_m":(mc/m).tolist()})
        data.out_forces.fill(0.)
        data.out_torques.fill(0.)
        for env,index,m,mc,second in cache:
            force,torque=stc_reaction_wrench(data.ipc_transforms[env,index],data.aim_transforms[env,index],
                m,mc,second,coupler._constraint_strength_translation_scaled,
                coupler._constraint_strength_rotation_scaled)
            data.out_forces[env,index]=force
            data.out_torques[env,index]=torque
        if not np.isfinite(data.out_forces).all() or not np.isfinite(data.out_torques).all():
            raise RuntimeError("Nonfinite exact STC feedback wrench")
        # During couple(), Genesis has not advanced since aim_transforms was
        # captured. Thus the current link origin is the target origin.
        coupler.rigid_solver.apply_links_external_wrench(
            data.out_forces if coupler.sim.n_envs>0 else data.out_forces[0],
            data.out_torques if coupler.sim.n_envs>0 else data.out_torques[0],
            links_idx=data.links_idx)
        report["applied_steps"]+=1

    coupler._apply_abd_coupling_forces=exact_feedback
    return report


def initialize_before_ipc(scene, robots, initial_q):
    """Install before scene.build(); preserve MJCF qpos0, set actual initial q.

Genesis joint.desc.init_qpos is the neutral reference qpos0, not an independent
start-pose field. The actual q must be applied after rigid buffers/FK exist but
before IPC constructs its world-space collision bodies.
"""
    if scene.is_built or version("genesis-world") != "1.4.3":
        raise RuntimeError("Initial-pose adapter requires an unbuilt Genesis 1.4.3 scene")
    coupler = scene.sim.coupler
    native_build = coupler.build
    report = {"method":"Actual qpos after rigid solver build, before IPC geometry build",
              "neutral_joint_reference_modified":False,"verified":False}

    def build_with_initial_pose():
        rigid = scene.sim.rigid_solver
        before = rigid.qpos0.to_numpy().copy()
        for side,robot in robots.items():
            q=np.asarray(initial_q[side],dtype=float)
            if q.shape != (robot.n_qs,):
                raise ValueError(f"{side} initial q has wrong shape: {q.shape}")
            rigid.set_qpos(q,qs_idx=np.arange(robot.q_start,robot.q_end),skip_forward=True)
        # Rigid update_forward_pos refreshes both links and collision geometry.
        rigid.update_forward_pos()
        if not np.array_equal(before,rigid.qpos0.to_numpy()):
            raise RuntimeError("Initial-pose adapter changed the neutral reference")
        native_build()
        # Populate IPC state buffers without advancing simulation time.
        coupler._store_gs_rigid_states()
        coupler._retrieve_rigid_states()
        data=coupler._coupling_data
        errors=np.linalg.norm(data.ipc_transforms[...,:3,3]-data.aim_transforms[...,:3,3],axis=-1)
        report["maximum_initial_ipc_genesis_translation_error_m"]=float(errors.max()) if errors.size else 0.
        report["verified"]=bool(not errors.size or errors.max()<1e-7)
        if not report["verified"]:
            raise RuntimeError(f"Initial IPC geometry disagrees with Genesis: {report}")

    coupler.build=build_with_initial_pose
    return report


def _map_shell_material_regions(source_npz, source_position, native_positions, native_faces):
    """Map named material regions without relying on either mesh's index order."""
    import hashlib
    from pathlib import Path
    from scipy.spatial import cKDTree

    source_path = Path(source_npz)
    with np.load(source_path, allow_pickle=False) as data:
        source = {name: np.array(data[name], copy=True) for name in
                  ("points", "faces", "body_face_indices", "handle_front", "handle_back")}
    points = np.asarray(source["points"], float)
    translation = np.asarray(source_position, float)
    native_points = np.asarray(native_positions, float).reshape(-1, 3)
    faces = np.asarray(native_faces).reshape(-1, 3)
    source_faces = source["faces"]
    if (points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3
            or native_points.shape != points.shape or translation.shape != (3,)
            or not all(np.isfinite(a).all() for a in (points, native_points, translation))):
        raise ValueError("Invalid source/native shell positions for material mapping")
    for name, array, upper in (("source faces", source_faces, len(points)),
                               ("native faces", faces, len(points))):
        if (array.ndim != 2 or array.shape[1:] != (3,) or not len(array)
                or not np.issubdtype(array.dtype, np.integer)
                or np.any(array < 0) or np.any(array >= upper)
                or np.any(np.diff(np.sort(array, axis=1), axis=1) == 0)):
            raise ValueError(f"Invalid {name} for shell material mapping")
    if len(faces) != len(source_faces):
        raise ValueError("Source/native shell face counts differ")

    tolerance = 1e-7
    distances, nearest = cKDTree(native_points).query(points + translation, k=2)
    vertex_map = nearest[:, 0]
    if (distances[:, 0].max() > tolerance or np.any(distances[:, 1] <= tolerance)
            or len(np.unique(vertex_map)) != len(points)):
        raise ValueError("Source/native shell vertices lack a unique exact bijection")
    native_keys = [tuple(row) for row in np.sort(faces, axis=1)]
    mapped_keys = [tuple(row) for row in np.sort(vertex_map[source_faces], axis=1)]
    if len(set(native_keys)) != len(faces) or len(set(mapped_keys)) != len(source_faces):
        raise ValueError("Duplicate shell triangles prevent material mapping")
    lookup = {key: index for index, key in enumerate(native_keys)}
    if set(lookup) != set(mapped_keys):
        raise ValueError("Source/native shell triangle topology differs")
    face_map = np.asarray([lookup[key] for key in mapped_keys], dtype=int)

    def indices(name, upper):
        value = source[name]
        if (value.ndim != 1 or not len(value) or not np.issubdtype(value.dtype, np.integer)
                or np.any(value < 0) or np.any(value >= upper)
                or len(np.unique(value)) != len(value)):
            raise ValueError(f"Invalid source material group {name}")
        return value

    body = indices("body_face_indices", len(source_faces))
    source_regions = {"body": body}
    for name in ("handle_front", "handle_back"):
        vertices = indices(name, len(points))
        source_regions[name] = np.flatnonzero(np.isin(source_faces, vertices).all(axis=1))
        if not len(source_regions[name]):
            raise ValueError(f"Source material group {name} has no complete triangles")
    coverage = np.zeros(len(source_faces), dtype=int)
    for region in source_regions.values():
        coverage[region] += 1
    if not np.all(coverage == 1):
        raise ValueError("Body and two handle regions must cover each shell triangle exactly once")
    regions = {name: face_map[region] for name, region in source_regions.items()}
    report = {
        "source_npz": str(source_path.resolve()),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "method": "Unique nearest-position vertex bijection, then unordered vertex-triplet face bijection",
        "maximum_vertex_mapping_error_m": float(distances[:, 0].max()),
        "mapping_tolerance_m": tolerance,
        "source_vertex_order_matches_native": bool(np.array_equal(vertex_map, np.arange(len(points)))),
        "source_face_order_matches_native": bool(np.array_equal(face_map, np.arange(len(faces)))),
        "source_translation_m": translation.tolist(),
        "vertex_bijection": True, "face_bijection": True,
        "body_and_handles_disjoint_complete_face_partition": True,
        "regions": {name: {"face_count": len(ids), "native_face_indices": ids.tolist()}
                    for name, ids in regions.items()},
    }
    return regions, report


def _set_shell_shear_coefficients(geometry, shear_stiffness_n_per_m, *,
                                 handle_shear_stiffness_n_per_m=None, regions=None):
    """Change only mu on an uninitialized native geometry; audit other attributes."""
    import uipc
    values = [shear_stiffness_n_per_m]
    if handle_shear_stiffness_n_per_m is not None:
        values.append(handle_shear_stiffness_n_per_m)
        if regions is None:
            raise ValueError("Local handle shear requires a verified material partition")
    if not np.isfinite(values).all() or np.any(np.asarray(values) <= 0):
        raise ValueError("Effective shell shear coefficients must be positive and finite")
    protected = {}
    for collection_name, names in (("triangles", ("lambda", "strain_rate")),
                                   ("vertices", ("thickness", "volume", "is_fixed")),
                                   ("edges", ("bending_stiffness",))):
        collection = getattr(geometry, collection_name)()
        for name in names:
            attribute = collection.find(name)
            if attribute is not None:
                protected[(collection_name, name)] = np.array(uipc.view(attribute), copy=True)
    meta_before = geometry.meta().to_json()
    mu = uipc.view(geometry.triangles().find("mu"))
    old = float(mu.mean())
    mu[:] = shear_stiffness_n_per_m
    if handle_shear_stiffness_n_per_m is not None:
        for name in ("handle_front", "handle_back"):
            mu[regions[name]] = handle_shear_stiffness_n_per_m
        if not np.all(mu[regions["body"]] == shear_stiffness_n_per_m):
            raise RuntimeError("Handle material override modified body shear")
    unchanged = {f"{collection}.{name}": bool(np.array_equal(
        before, uipc.view(getattr(geometry, collection)().find(name))))
        for (collection, name), before in protected.items()}
    unchanged["geometry_metadata_including_contact_material"] = meta_before == geometry.meta().to_json()
    if not all(unchanged.values()):
        raise RuntimeError("Shell shear conversion changed a protected physical attribute")
    report = {"previous_shear_coefficient": old,
              "new_shear_coefficient": float(mu.mean()),
              "new_shear_coefficient_scope": "Mean over all triangles; per-region values are authoritative",
              "protected_attributes_unchanged": unchanged}
    if regions is not None:
        report["region_coefficients"] = {name: {
            "face_count": len(ids), "shear_stiffness_n_per_m": float(mu[ids[0]]),
            "all_faces_equal": bool(np.all(mu[ids] == mu[ids[0]]))}
            for name, ids in regions.items()}
    return report


def configure_shell_shear(scene, *, shear_stiffness_n_per_m=20.0,
                          handle_shear_stiffness_n_per_m=None, handle_source_npz=None,
                          handle_source_position=None, handle_entity=None):
    """Set independent membrane shear stiffness for this scene's cloth shells.

In pyuipc 0.0.29's Baraff-Witkin implementation, the triangle shear coefficient
is E_shear/[2(1+nu)] and multiplies triangle area; unlike stretch, its conversion
does not include thickness. Genesis currently passes the same E to both modes.
That makes the default shear much harder than membrane stretch for thin fabric.
This narrow compatibility adapter replaces only the generated per-triangle mu
coefficient before World.init. 20 N/m is an uncalibrated effective cloth value,
not a measured property; change it explicitly when calibration is available.
An optional handle override changes only the two named, sewn cloth strips'
effective shear. Their membrane stretch, bending, mass and contact stay intact.
"""
    if scene.is_built or version("genesis-world")!="1.4.3" or version("pyuipc")!="0.0.29":
        raise RuntimeError("Shell shear adapter requires an unbuilt, version-matched scene")
    if not np.isfinite(shear_stiffness_n_per_m) or shear_stiffness_n_per_m<=0:
        raise ValueError("Effective shear stiffness must be positive")
    local_handles = handle_shear_stiffness_n_per_m is not None
    if local_handles and (not np.isfinite(handle_shear_stiffness_n_per_m)
            or handle_shear_stiffness_n_per_m <= 0 or handle_source_npz is None
            or handle_source_position is None or handle_entity is None):
        raise ValueError("Handle shear requires a positive coefficient and explicit source, pose and entity")
    from uipc.backend import SceneVisitor
    from uipc.geometry import SimplicialComplexSlot
    from genesis.engine.couplers.ipc_coupler.utils import read_ipc_geometry_metadata
    import uipc
    coupler=scene.sim.coupler
    native_add=coupler._add_fem_entities_to_ipc
    report={"scope":"Per-scene pre-initialization FEM shell coefficient conversion",
            "calibrated":False,"shear_stiffness_n_per_m":float(shear_stiffness_n_per_m),
            "handle_shear_override_enabled":local_handles,
            "handle_shear_stiffness_n_per_m":float(handle_shear_stiffness_n_per_m) if local_handles else None,
            "handle_material_scope":"Uncalibrated effective cloth-strip shear; no yarn, rigid support, pin or attachment model",
            "thickness_semantics":"IPC one-sided radius r; full thickness h=2r",
            "entities":[]}

    def add_with_independent_shear():
        native_add()
        matched_handles = 0
        for slot in SceneVisitor(coupler._ipc_scene).geometries():
            if not isinstance(slot,SimplicialComplexSlot):
                continue
            geometry=slot.geometry()
            meta=read_ipc_geometry_metadata(geometry)
            if meta is None or meta[0]!="cloth":
                continue
            _,env,index=meta
            entity=coupler.fem_solver.entities[index]
            regions = mapping_report = None
            if local_handles and entity is handle_entity:
                regions, mapping_report = _map_shell_material_regions(
                    handle_source_npz, handle_source_position,
                    uipc.view(geometry.positions()), uipc.view(geometry.triangles().topo()))
                matched_handles += 1
            coefficient_report = _set_shell_shear_coefficients(
                geometry, shear_stiffness_n_per_m,
                handle_shear_stiffness_n_per_m=handle_shear_stiffness_n_per_m if regions is not None else None,
                regions=regions)
            stretch=uipc.view(geometry.triangles().find("lambda"))
            volume=uipc.view(geometry.vertices().find("volume"))
            report["entities"].append({"entity_index":int(index),"environment":int(env),
                **coefficient_report, "material_region_mapping":mapping_report,
                "stretch_coefficient":float(stretch.mean()),
                "full_shell_thickness_m":float(2*entity.material.thickness),
                "native_rest_mass_kg":float(volume.sum()*entity.material.rho)})
        if not report["entities"]:
            raise RuntimeError("No IPC cloth geometry found for independent shear configuration")
        if local_handles and matched_handles != scene.sim._B:
            raise RuntimeError("Named handle entity was not uniquely mapped in every environment")

    coupler._add_fem_entities_to_ipc=add_with_independent_shear
    return report


def coupling_diagnostics(scene):
    """Read latest IPC contact-body transforms against current Genesis links.

Use after build and/or step. Initialization helper populates the otherwise
uninitialized IPC arrays before the first step. No state positions are changed.
"""
    from scipy.spatial.transform import Rotation
    coupler=scene.sim.coupler
    if scene.sim.cur_step_global==0:
        # Scene.build performs one compilation step then reset(). IPC reset
        # restores frame 0 but does not refresh these Python accessor buffers.
        # Read the recovered world so t=0 diagnostics are not stale warmup data.
        coupler._store_gs_rigid_states()
        coupler._retrieve_rigid_states()
    data=coupler._coupling_data
    values=[]
    for index,link in enumerate(data.links):
        if link.entity.n_dofs==0:
            continue
        ipc=data.ipc_transforms[0,index]
        pos=link.get_pos().detach().cpu().numpy().reshape(3)
        q=link.get_quat().detach().cpu().numpy().reshape(4)
        current=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        u,s,vh=np.linalg.svd(ipc[:3,:3])
        polar=u@vh
        if np.linalg.det(polar)<0:
            u[:,-1]*=-1
            polar=u@vh
        row={"entity":link.entity.name,"link":link.name,
            "translation_error_m":float(np.linalg.norm(ipc[:3,3]-pos)),
            "rotation_error_rad":float(Rotation.from_matrix(polar@current.T).magnitude()),
            "affine_singular_value_deviation":float(np.abs(s-1).max())}
        if "finger" in link.name:
            # Preserve the raw buffers needed to distinguish an origin-based
            # feedback spring from IPC's centre-of-mass STC energy. These are
            # observations only; this function never changes coupling forces.
            import uipc
            geometry=coupler._abd_slots_by_link[link][0].geometry()
            meta=geometry.meta()
            mass=float(np.asarray(uipc.view(meta.find("mass"))).reshape(-1)[0])
            center=np.asarray(uipc.view(meta.find("mass_center"))).reshape(3)
            aim=data.aim_transforms[0,index]
            dc=(ipc[:3,3]+ipc[:3,:3]@center)-(aim[:3,3]+aim[:3,:3]@center)
            origin_delta=ipc[:3,3]-aim[:3,3]
            scale=float(coupler._constraint_strength_translation_scaled)
            origin_formula=scale*float(data.links_mass[index])*origin_delta
            com_formula=scale*mass*dc
            # Panda finger joint axis is local +Y on each mirrored finger.
            opening_axis=current[:,1]
            progressed=scene.sim.cur_step_global>0
            row.update({
                "ipc_transform":ipc.tolist(), "aim_transform":aim.tolist(),
                "genesis_current_position_m":pos.tolist(),
                "genesis_current_rotation_matrix":current.tolist(),
                "ipc_geometry_mass_kg":mass,
                "genesis_link_mass_kg":float(data.links_mass[index]),
                "ipc_geometry_local_center_of_mass_m":center.tolist(),
                "ipc_geometry_local_inertia":np.asarray(uipc.view(meta.find("inertia"))).reshape(3,3).tolist(),
                "origin_delta_ipc_minus_aim_m":origin_delta.tolist(),
                "com_delta_ipc_minus_aim_m":dc.tolist(),
                "opening_axis_world":opening_axis.tolist(),
                "aim_opening_axis_world":aim[:3,1].tolist(),
                "origin_feedback_formula_n":origin_formula.tolist(),
                "com_translation_gradient_formula_n":com_formula.tolist(),
                "origin_formula_along_opening_n":float(origin_formula@opening_axis),
                "com_formula_along_opening_n":float(com_formula@opening_axis),
                "origin_formula_along_aim_opening_n":float(origin_formula@aim[:3,1]),
                "com_formula_along_aim_opening_n":float(com_formula@aim[:3,1]),
                "last_feedback_force_world_n":data.out_forces[0,index].tolist() if progressed else None,
                "last_feedback_torque_world_nm":data.out_torques[0,index].tolist() if progressed else None,
                "translation_gradient_is_not_complete_rigid_wrench":True,
            })
        values.append(row)
    return {"links":values,
        "max_translation_error_m":max((v["translation_error_m"] for v in values),default=0.),
        "max_rotation_error_rad":max((v["rotation_error_rad"] for v in values),default=0.),
        "max_affine_deviation":max((v["affine_singular_value_deviation"] for v in values),default=0.)}
