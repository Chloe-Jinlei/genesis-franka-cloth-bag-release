"""Offline insertion candidates from an actual saved bag/teddy/robot state.

This module never initializes Genesis or modifies a simulation. It predicts
the held teddy's footprint under a rigid wrist rotation only to screen routes;
the real teddy remains deformable and needs fresh runtime contact checks.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation

from motion import DOWN, Planner
from validate import mouth_structure, proper_segment_crossings, signed_volumes


def projected_inside(points, polygon):
    """Ray casting of XY points against the actual ordered rim projection."""
    points, polygon = np.asarray(points), np.asarray(polygon)
    inside = np.zeros(len(points), bool)
    distances = np.full(len(points), np.inf)
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        delta = b-a
        if np.dot(delta, delta) < 1e-16:
            continue
        fraction = np.clip(((points-a)@delta)/np.dot(delta, delta), 0., 1.)
        distances = np.minimum(distances, np.linalg.norm(points-a-fraction[:,None]*delta, axis=1))
        crossing = (a[1] > points[:,1]) != (b[1] > points[:,1])
        if abs(delta[1]) > 1e-14:
            x_cross = a[0] + (points[:,1]-a[1])*delta[0]/delta[1]
            inside ^= crossing & (points[:,0] < x_cross)
    return inside | (distances < 1e-8), distances


def footprint_containment(points, polygon, *, margin=.002):
    """Conservatively contain the full teddy projection, not only its nodes.

    A convex envelope can reject a genuinely fitting concave teddy, but cannot
    falsely pass an edge bridging a concave/folded mouth. Segment distances
    include both directions, and crossings are checked analytically.
    """
    try:
        points = np.unique(np.asarray(points, float), axis=0)
        hull = points[ConvexHull(points).vertices]
    except (QhullError, ValueError):
        return False, 0., float("inf")
    polygon = np.asarray(polygon, float)
    inside, distances = projected_inside(hull, polygon)
    _, reverse_distances = projected_inside(polygon, hull)
    a, b = hull, np.roll(hull,-1,axis=0)
    c, d = polygon, np.roll(polygon,-1,axis=0)
    def cross(x,y):
        return x[...,0]*y[...,1]-x[...,1]*y[...,0]
    first = cross((b-a)[:,None],c[None]-a[:,None])
    second = cross((b-a)[:,None],d[None]-a[:,None])
    third = cross((d-c)[None],a[:,None]-c[None])
    fourth = cross((d-c)[None],b[:,None]-c[None])
    crosses = bool(((first*second < -1e-20) & (third*fourth < -1e-20)).any())
    separation = min(float(distances.min()),float(reverse_distances.min()))
    fit = bool(inside.all() and not crosses and separation >= margin)
    if fit:
        return True, separation, 0.
    # Bound the outside excursion of hull edges, including parts between
    # mesh nodes. The distance-to-boundary function is 1-Lipschitz, so half
    # the maximum sample step bounds an unsampled excursion conservatively.
    sampled=[]
    maximum_step=0.
    for p,q in zip(a,b):
        intervals=max(1,int(np.ceil(np.linalg.norm(q-p)/.001)))
        maximum_step=max(maximum_step,float(np.linalg.norm(q-p)/intervals))
        sampled.append(p+(q-p)*np.linspace(0.,1.,intervals+1)[:,None])
    sample_inside, sample_distance=projected_inside(np.concatenate(sampled),polygon)
    outside=float(sample_distance[~sample_inside].max()) if (~sample_inside).any() else 0.
    return False, separation, outside+maximum_step/2


def minimum_nonadjacent_edge_distance(polygon):
    """Detect projected rim self-touch, including collinear overlapping edges."""
    polygon=np.asarray(polygon,float)
    minimum=float("inf")
    for i,(a,b) in enumerate(zip(polygon,np.roll(polygon,-1,axis=0))):
        for j in range(i+2,len(polygon)):
            if i==0 and j==len(polygon)-1:
                continue
            c,d=polygon[j],polygon[(j+1)%len(polygon)]
            for points,p,q in ((np.asarray([a,b]),c,d),(np.asarray([c,d]),a,b)):
                delta=q-p
                fraction=np.clip((points-p)@delta/max(float(delta@delta),1e-20),0.,1.)
                minimum=min(minimum,float(np.linalg.norm(points-p-fraction[:,None]*delta,axis=1).min()))
    return minimum


def mass_principal_axes(points, tetrahedra, rest_volume_weights, center_of_mass):
    """Exact covariance of uniformly filled tetrahedra with conserved mass."""
    vertices=np.asarray(points)[tetrahedra]-center_of_mass
    sums=vertices.sum(axis=1)
    moments=(np.einsum("tki,tkj->tij",vertices,vertices)
             +np.einsum("ti,tj->tij",sums,sums))/20.
    covariance=np.einsum("t,tij->ij",rest_volume_weights,moments)/rest_volume_weights.sum()
    values,vectors=np.linalg.eigh(covariance)
    long,middle=vectors[:,2],vectors[:,1]
    return np.column_stack((long,middle,np.cross(long,middle))),values[::-1]


def polygon_centroid(polygon):
    following=np.roll(polygon,-1,axis=0)
    cross=polygon[:,0]*following[:,1]-following[:,0]*polygon[:,1]
    if abs(cross.sum())<1e-12: return np.zeros(2)
    return ((polygon+following)*cross[:,None]).sum(0)/(3*cross.sum())


def best_footprint_offset(centered_projection, polygon, compression_limit, outside_limit,
                          *, fixed_delta=None, current_projection=None,current_center=None,
                          margin=.002):
    """Choose a small in-plane offset without moving simulated vertices."""
    # Reducing to the convex hull preserves the complete conservative envelope.
    shape=centered_projection[ConvexHull(centered_projection).vertices]
    anchor=polygon_centroid(polygon)
    deltas=([np.asarray(fixed_delta,float)] if fixed_delta is not None else
            [np.array([x,y]) for x in (0.,-.01,.01,-.02,.02) for y in (0.,-.01,.01,-.02,.02)])
    best=None
    for delta in deltas:
        offset2=anchor+delta
        projected=shape+offset2
        centers=np.broadcast_to(offset2,projected.shape)
        if current_projection is not None:
            current_hull=current_projection[ConvexHull(current_projection).vertices]
            projected=np.vstack((projected,current_hull))
            centers=np.vstack((centers,np.broadcast_to(current_center,current_hull.shape)))
        fit,separation,outside=footprint_containment(projected,polygon,margin=margin)
        compression=0. if fit else None
        if not fit and outside <= outside_limit:
            for amount in np.linspace(0.,compression_limit,9)[1:]:
                compressed=(projected-centers)*(1-amount)+centers
                fits,compressed_margin,_=footprint_containment(compressed,polygon,margin=margin)
                if fits:
                    compression=float(amount);separation=compressed_margin;break
        rank=(compression is not None,-(compression if compression is not None else 1.),
              separation if compression is not None else -outside,-float(np.linalg.norm(delta)))
        if best is None or rank>best[0]:
            best=(rank,dict(offset2=offset2,delta=delta,compression=compression,
                fit=fit,margin=separation,outside=outside,projected=projected))
    return best[1]


def candidates(rest, state, config, *, yaws=(0., 90., -90.), tilts=(0., 90., -90.),
               clearances=(.04, .08),
               approaches=("vertical", "mouth_normal"), screen_geometry_first=False,
               align_yaw=True, plan_mode="full", fixed_rotation_xyzw=None,
               fixed_offset_strategy=None,max_feasible_routes=None):
    if plan_mode not in ("full","insert_only"):
        raise ValueError("Unknown insertion planning mode")
    bag, bear = np.asarray(state["bag"]), np.asarray(state["bear"])
    for name,points in (("bag",bag),("bear",bear)):
        if (points.ndim != 2 or points.shape[1] != 3 or len(points) < 4
                or points.shape != np.asarray(rest[name]).shape or not np.isfinite(points).all()
                or not np.isfinite(rest[name]).all()):
            raise ValueError(f"Observed and rest {name} must have matching finite (n,3) coordinates")
    for side in ("left","right"):
        q=np.asarray(state["q_"+side]).reshape(-1)
        tcp=np.asarray(state["tcp_"+side])
        if len(q)!=9 or not np.isfinite(q).all() or tcp.shape!=(3,) or not np.isfinite(tcp).all():
            raise ValueError(f"Invalid observed {side} robot state")
    if not np.isfinite(float(state["time"])):
        raise ValueError("Observed time must be finite")
    for key,width,count in (("bag_faces",3,len(bag)),("bear_tets",4,len(bear))):
        cells=np.asarray(rest[key])
        if (cells.ndim!=2 or cells.shape[1]!=width or not len(cells)
                or cells.dtype.kind not in "iu" or cells.min()<0 or cells.max()>=count
                or np.any(np.diff(np.sort(cells,axis=1),axis=1)==0)):
            raise ValueError(f"Invalid integer simplex topology: {key}")
    _, rim, topology = mouth_structure(rest["bag"], rest["bag_faces"], config)
    if (len(rim) < 4 or not topology["body_boundary_only_at_top"]
            or not topology["two_panels_directly_sewn_along_u"]):
        raise ValueError("Could not identify a valid sewn-bag mouth")
    rp = bag[rim]
    mouth_center = rp.mean(0)
    centered = rp-mouth_center
    _, singular_values, basis = np.linalg.svd(centered, full_matrices=False)
    if singular_values[1]<1e-5:
        raise ValueError("Observed mouth has no two-dimensional opening")
    normal = basis[-1].copy()
    if normal[2] < 0:
        normal = -normal
    major=basis[0].copy()
    reference=(np.asarray(fixed_offset_strategy["reference_axes_world"])[0]
               if fixed_offset_strategy is not None else np.array([1.,0.,0.]))
    if major@reference<0: major=-major
    basis=np.vstack((major,np.cross(normal,major),normal))
    area_xy = abs(np.sum(rp[:,0]*np.roll(rp[:,1],-1)-rp[:,1]*np.roll(rp[:,0],-1)))/2
    crossings = proper_segment_crossings(rp[:,:2])
    left_distance = float(np.linalg.norm(bag-np.asarray(state["tcp_left"]),axis=1).min())
    planner = Planner(config)
    qs = {s:np.asarray(state["q_"+s],float).reshape(-1)[:7].copy() for s in ("left","right")}
    current_tcp, current_rotation = planner.fk("right", qs["right"])
    measured_tcp = np.asarray(state["tcp_right"],float)
    relative_bear = bear-measured_tcp
    # A sphere about the actual grasp encloses every orientation during the
    # wrist flip, including intermediate rotations. It is deliberately more
    # conservative than checking only the two endpoint teddy poses.
    held_radius = float(np.linalg.norm(relative_bear, axis=1).max())
    tets = np.asarray(rest["bear_tets"],int)
    if tets.ndim!=2 or tets.shape[1]!=4 or not len(tets) or tets.min()<0 or tets.max()>=len(bear):
        raise ValueError("Invalid teddy tetrahedron topology")
    rest_volumes=signed_volumes(np.asarray(rest["bear"]),tets)
    if np.any(np.abs(rest_volumes)<1e-15):
        raise ValueError("Rest teddy has degenerate tetrahedra")
    weights = np.abs(rest_volumes)
    volume_ratio = signed_volumes(bear,tets)/rest_volumes
    center_of_mass = np.average(bear[tets].mean(1),axis=0,weights=weights)
    body_axes,body_variances=mass_principal_axes(bear,tets,weights,center_of_mass)
    offset = center_of_mass-measured_tcp
    fingers = {s:float(np.clip(np.asarray(state["q_"+s]).reshape(-1)[7:9].mean(),0.,.04)) for s in qs}
    warning = []
    if left_distance > .03:
        warning.append("Left gripper is farther than 30 mm from every bag vertex; this state does not demonstrate a held bag.")
    if volume_ratio.min() < .05:
        warning.append("Current teddy has a tetrahedron volume ratio below 0.05; this state is not an admissible insertion start.")
    tcp_errors={side:float(np.linalg.norm(planner.fk(side,qs[side])[0]-state["tcp_"+side]))
                for side in ("left","right")}
    if max(tcp_errors.values()) > .002:
        warning.append("Recorded TCP differs from the measured-joint rigid model by over 2 mm; refresh the robot state before planning.")
    raw_fingers={side:np.asarray(state["q_"+side])[7:9].tolist() for side in ("left","right")}
    if any(min(q)<-.0005 or max(q)>.0405 for q in raw_fingers.values()):
        warning.append("Observed finger position is outside the physical travel range by over 0.5 mm.")
    limits = config.get("task", {}).get("insertion_limits", {})
    compression_limit = float(limits.get("max_planar_compression",.08))
    outside_limit = float(limits.get("max_outside_projection_m",.008))
    if not 0 <= compression_limit <= .12 or not 0 <= outside_limit <= .012:
        raise ValueError("Insertion compression allowances exceed the bounded offline screening range")
    _,_,bear_xy_basis = np.linalg.svd(bear[:,:2]-bear[:,:2].mean(0),full_matrices=False)
    aligned_yaw = (np.degrees(np.arctan2(basis[0,1],basis[0,0])
                    -np.arctan2(bear_xy_basis[0,1],bear_xy_basis[0,0]))+90)%180-90
    yaws = list(yaws)
    if align_yaw and all(abs(aligned_yaw-yaw)>.5 for yaw in yaws):
        yaws.append(float(aligned_yaw))
    orientation_options=[]
    if fixed_rotation_xyzw is not None:
        orientation_options=[{"kind":"fixed_executed_orientation",
            "rotation":Rotation.from_quat(fixed_rotation_xyzw)}]
    else:
        for long_sign in (1.,-1.):
            for middle_sign in (1.,-1.):
                target_long=long_sign*normal;target_middle=middle_sign*basis[0]
                target_axes=np.column_stack((target_long,target_middle,np.cross(target_long,target_middle)))
                delta_rotation=Rotation.from_matrix(target_axes@body_axes.T)
                for spin in (0.,-10.,10.,-20.,20.):
                    rotation=Rotation.from_rotvec(normal*np.deg2rad(spin))*delta_rotation*current_rotation
                    orientation_options.append({"kind":"observed_mass_principal_axes_to_mouth",
                        "long_axis_sign":long_sign,"middle_axis_sign":middle_sign,
                        "spin_about_mouth_normal_deg":spin,"rotation":rotation})
        for yaw in yaws:
            for tilt in tilts:
                orientation_options.append({"kind":"fixed_world_tilt_yaw",
                    "wrist_world_z_yaw_deg":float(yaw),"wrist_world_x_tilt_deg":float(tilt),
                    "rotation":Rotation.from_euler("z",yaw,degrees=True)*Rotation.from_euler("x",tilt,degrees=True)*DOWN})
        orientation_options.sort(key=lambda o:float((o["rotation"]*current_rotation.inv()).magnitude()))
    report = {
        "kind":"offline_geometry_candidates_from_observed_state",
        "robot_planning_scope":"Only candidates passing geometric screens" if screen_geometry_first else "All requested geometric candidates",
        "robot_planning_mode":plan_mode,
        "observed_time_s":float(state["time"]), "warnings":warning,
        "mouth":{"center_m":mouth_center.tolist(),"rim_maximum_z_m":float(rp[:,2].max()),
            "xy_projected_area_m2":float(area_xy),"xy_projected_crossings":crossings,
            "fitted_plane_normal_abs_z":float(abs(basis[-1,2])),"topology":topology},
        "left_nearest_bag_vertex_m":left_distance,
        "observed_bear_dimensions_xyz_m":np.ptp(bear,axis=0).tolist(),
        "observed_bear_center_of_mass_offset_m":offset.tolist(),
        "observed_mass_principal_axes_world":body_axes.tolist(),
        "observed_mass_principal_variances_m2":body_variances.tolist(),
        "motion_search_stop_after_valid_routes":max_feasible_routes,
        "current_minimum_bear_tetrahedron_volume_ratio":float(volume_ratio.min()),
        "screening_allowances":{"max_planar_compression":compression_limit,
            "max_outside_projection_m":outside_limit},
        "orientation_sampling":{"world_z_yaw_deg":[float(v) for v in yaws],
            "world_x_tilt_deg":[float(v) for v in tilts],
            "composition":"Observed mass principal axes to mouth normal/major axis, both axis signs and +/-20 degree normal spins; plus legacy world tilt/yaw",
            "orientation_count":len(orientation_options)},
        "fk_vs_recorded_right_tcp_error_m":float(np.linalg.norm(current_tcp-measured_tcp)),
        "fk_vs_recorded_tcp_error_m":tcp_errors,
        "observed_finger_q_m":raw_fingers,
        "rigid_preflight_finger_q_m":fingers,
        "candidates":[],
        "limitations":[
            "This is an offline rigid-arm check, not a successful physical insertion or grasp certificate.",
            "Bear footprint predictions rotate the observed shape rigidly around the hand; real gravity, deformation and slip can change it.",
            "Rim and held bear observations become stale while the robot moves; observe again before insertion.",
            "A footprint-fit result assumes translation through the chosen projection plane; it does not establish volume containment.",
            "Sampled rigid clearance does not prove continuous collision freedom; runtime replanning and physical collision checks are required.",
        ]}
    report["mouth"].update(fitted_normal_up=normal.tolist(),
        maximum_nonplanarity_m=float(np.abs(centered@normal).max()))
    for approach in approaches:
        if approach not in ("vertical", "mouth_normal"):
            raise ValueError(f"Unknown approach {approach}")
        projection = np.eye(3)[:2] if approach == "vertical" else basis[:2]
        direction = np.array([0.,0.,1.]) if approach == "vertical" else normal
        polygon = centered@projection.T
        projected_crossings = proper_segment_crossings(polygon)
        rim_clearance=minimum_nonadjacent_edge_distance(polygon)
        projected_area=abs(float(np.sum(polygon[:,0]*np.roll(polygon[:,1],-1)
                                    -polygon[:,1]*np.roll(polygon[:,0],-1))))/2
        mode_warnings = warning.copy()
        if projected_crossings:
            mode_warnings.append("Mouth contour crosses itself in this approach projection; no footprint-fit claim is made.")
        if rim_clearance < max(.0005,2*float(config["bag"].get("thickness",.0003))):
            mode_warnings.append("Nonadjacent mouth edges nearly touch in this projection; no open passage is established.")
        if projected_area<.0001:
            mode_warnings.append("Projected mouth area is under 1 square cm; no usable opening is established.")
        if approach == "vertical" and normal[2] < .45:
            mode_warnings.append("Mouth is too steep for a simple downward insertion.")
        if np.abs(centered@normal).max() > .025:
            mode_warnings.append("Mouth deviates from its fitted plane by over 25 mm; one fitted normal does not describe the folds.")
        for orientation in orientation_options:
            target_rotation = orientation["rotation"]
            delta = target_rotation*current_rotation.inv()
            predicted_relative = delta.apply(relative_bear)
            rotated_offset = delta.apply(offset)
            centered_prediction = predicted_relative-rotated_offset
            projected = centered_prediction@projection.T
            current_projected=current_center=None
            if plan_mode=="insert_only":
                # Fixed-orientation translation is a convex combination of
                # endpoint projections. Enclose the full sweep to avoid
                # approving a centered endpoint reached through the rim.
                current_projected=(bear-mouth_center)@projection.T
                current_center=(center_of_mass-mouth_center)@projection.T
            rotation_padding=(2*held_radius*np.sin(delta.magnitude()/2)
                              if plan_mode=="insert_only" else 0.)
            fixed_delta=(None if fixed_offset_strategy is None else
                projection@np.asarray(fixed_offset_strategy["delta_world_m"])
                if "delta_world_m" in fixed_offset_strategy else fixed_offset_strategy["delta_m"])
            footprint=best_footprint_offset(projected,polygon,compression_limit,outside_limit,
                fixed_delta=fixed_delta,current_projection=current_projected,current_center=current_center,
                margin=.002+rotation_padding)
            projected=footprint["projected"]
            center_target=mouth_center+projection.T@footprint["offset2"]
            offset_strategy={"anchor":"fresh_projected_mouth_area_centroid",
                "delta_m":footprint["delta"].tolist(),"reference_axes_world":projection.tolist(),
                "delta_world_m":(projection.T@footprint["delta"]).tolist()}
            inside, edge_distance = projected_inside(projected,polygon)
            hull_margin,maximum_outside=footprint["margin"],footprint["outside"]
            fit = bool(projected_crossings == 0 and footprint["fit"])
            needed_compression=footprint["compression"] if not projected_crossings else None
            for clearance in clearances:
                if approach == "vertical":
                    above_z = max(float(rp[:,2].max()-predicted_relative[:,2].min()+clearance),
                                  float(config["table"]["height"]+.30))
                    above = np.r_[center_target[:2]-rotated_offset[:2],above_z]
                    insert = above.copy()
                    insert[2] = max(float(config["table"]["height"]+.03-predicted_relative[:,2].min()),
                                    float(center_target[2]-.08-rotated_offset[2]))
                else:
                    distance = max(0.,float((centered@direction).max()
                                    -(centered_prediction@direction).min()+clearance))
                    above = center_target-rotated_offset+direction*distance
                    max_depth = (center_target[2]+centered_prediction[:,2].min()
                                 -config["table"]["height"]-.03)/max(direction[2],1e-6)
                    depth = min(.08,max(0.,float(max_depth)))
                    insert = center_target-rotated_offset-direction*depth
                staging_z = max(float(current_tcp[2]),float(above[2]),
                                float(config["table"]["height"]+.03+held_radius))
                staging = np.r_[current_tcp[:2],staging_z]
                entry = {"approach":approach,"outward_approach_axis":direction.tolist(),
                    "warnings":mode_warnings,"orientation_source":{k:v for k,v in orientation.items() if k!="rotation"},
                    "wrist_world_z_yaw_deg":orientation.get("wrist_world_z_yaw_deg"),
                    "mouth_projected_area_m2":projected_area,
                    "mouth_nonadjacent_edge_clearance_m":rim_clearance,
                    "wrist_world_x_tilt_deg":orientation.get("wrist_world_x_tilt_deg"),
                    "in_plane_offset_strategy":offset_strategy,"mouth_entry_center_m":center_target.tolist(),
                    "fresh_rotation_sweep_padding_m":float(rotation_padding),
                    "bear_clearance_above_rim_m":float(clearance),
                    "rotation_xyzw":target_rotation.as_quat().tolist(),
                    "predicted_rotated_bear_dimensions_xyz_m":np.ptp(predicted_relative,axis=0).tolist(),
                    "predicted_projected_vertices_inside_fraction":float(inside.mean()),
                    "predicted_footprint_fits_with_2mm_margin":fit,
                    "predicted_convex_footprint_boundary_separation_m":hull_margin,
                    "predicted_maximum_outside_rim_distance_m":maximum_outside,
                    "required_predicted_planar_compression":needed_compression,
                    "bounded_compression_screen_passes":needed_compression is not None,
                    "rotation_staging_tcp_m":staging.tolist(),
                    "rotation_enclosing_sphere_radius_m":held_radius,
                    "rotation_sphere_table_clearance_m":float(staging_z-held_radius-config["table"]["height"]),
                    "above_tcp_m":above.tolist(),"insert_tcp_m":insert.tolist(),
                    "robot_route_feasible":False,"robot_planning_attempted":False,"stages":[]}
                current = {s:q.copy() for s,q in qs.items()}
                try:
                    if screen_geometry_first and (mode_warnings or needed_compression is None):
                        raise ValueError("Observed mouth or bounded teddy compression screen rejects this candidate before robot planning")
                    if max_feasible_routes is not None and sum(v["geometry_screen_passes"] for v in report["candidates"])>=max_feasible_routes:
                        raise ValueError("Sufficient valid routes already found; additional robot planning was not attempted")
                    if above[2]+predicted_relative[:,2].min() < config["table"]["height"]+.01:
                        raise ValueError("Predicted teddy would intersect the tabletop at the outside approach point")
                    if insert[2]+predicted_relative[:,2].min() < config["table"]["height"]+.03-1e-10:
                        raise ValueError("Predicted teddy would have less than 30 mm tabletop clearance at the insertion endpoint")
                    if plan_mode=="insert_only":
                        outside_distance=float(((bear-mouth_center)@direction).min()
                            -(centered@direction).max())
                        entry["observed_bear_clearance_outside_mouth_m"]=outside_distance
                        if outside_distance < .005:
                            raise ValueError("Observed teddy is not fully outside the current mouth plane with 5 mm clearance")
                    entry["robot_planning_attempted"] = True
                    stages=(("height_at_current_xy",staging,current_rotation),
                            ("rotate_held_teddy",staging,target_rotation),
                            ("translate_outside_mouth",above,target_rotation),
                            ("insert_candidate",insert,target_rotation))
                    if plan_mode=="insert_only": stages=stages[-1:]
                    for name, target, rotation in stages:
                        plan = planner.trajectory(current,{"right":target},rotations={"right":rotation},
                            samples=35,validate_samples=61,duration=2.5,
                            finger_start=fingers,finger_end=fingers,clearance_margin=.008)
                        current = plan.end_q()
                        entry["stages"].append({"name":name,"right_tcp_m":target.tolist(),
                            "rotation_xyzw":rotation.as_quat().tolist(),"duration_s":plan.duration,
                            "right_q_end":current["right"].tolist()})
                    entry["robot_route_feasible"] = True
                    entry["total_motion_duration_s"] = sum(s["duration_s"] for s in entry["stages"])
                except ValueError as error:
                    entry["rejection"] = str(error)
                entry["geometry_screen_passes"] = bool(entry["robot_route_feasible"]
                    and needed_compression is not None and not mode_warnings)
                report["candidates"].append(entry)
    report["robot_feasible_candidates"] = sum(v["robot_route_feasible"] for v in report["candidates"])
    report["geometry_screen_candidates"] = sum(v["geometry_screen_passes"] for v in report["candidates"])
    return report


class NoInsertionRoute(ValueError):
    """Fail closed while retaining the complete rejection report for logging."""
    def __init__(self, report):
        super().__init__("No observed-state insertion route passed mouth, bounded-compression and rigid-motion checks")
        self.report = report


def choose_insertion_route(rest, state, config, *, output=None):
    """Return a reviewable route for fresh actual state; never executes it.

    Returned stages use world TCP positions and SciPy xyzw hand rotations.
    Candidates include world-X tilts of 0 and +/-90 degrees (then a world-Z
    yaw), so the observed long teddy axis can become vertical before entry.
    Runtime must reobserve the mouth immediately before insert_candidate; this
    prediction cannot keep a collapsing mouth open or guarantee contact grasp.
    """
    report = candidates(rest,state,config,screen_geometry_first=True,max_feasible_routes=4)
    viable = [c for c in report["candidates"] if c["geometry_screen_passes"]]
    if output is not None:
        Path(output).write_text(json.dumps(report,indent=2)+"\n")
    if not viable:
        raise NoInsertionRoute(report)
    tilted = report["mouth"]["fitted_plane_normal_abs_z"] < np.cos(np.deg2rad(15))
    viable.sort(key=lambda c:(c["required_predicted_planar_compression"],
        int(tilted and c["approach"]!="mouth_normal"),c["total_motion_duration_s"]))
    return {"route":viable[0],"report":report,
        "physical_success":False,"requires_physical_validation":True,
        "requires_reobservation_before_insertion":True,
        "requires_runtime_grasp_and_deformation_gates":True,
        "requires_contact_deformation_trial":viable[0]["required_predicted_planar_compression"]>0,
        "reobserve_before_every_stage":False,
        "recommended_additional_observation":"after_orientation_before_transport",
        "observe_before_stage":"insert_candidate",
        "compression_is_a_prediction_not_applied_deformation":True}


def replan_after_orientation(rest, state, config, previous_route=None, *,
                             attempt=0, max_attempts=2, output=None):
    """Plan bounded corrections from the current measured pose and soft shape.

    The actual hand orientation is always tested first.  Only if that fails do
    we try rotations of 5, 10, ..., 35 degrees about world vertical and the
    measured mouth normal.  They multiply the *current* hand rotation; no grasp
    location or nominal wrist pose is assumed.  All existing physical, shape,
    IK, continuous joint-limit and rigid-clearance gates remain in force.

    ``attempt`` counts previously executed correction rotations, not calls.
    Execute only ``stages_to_execute``.  When ``next_observation`` is
    ``after_orientation``, capture new actual state and call this function with
    the full returned result as ``previous_route`` and ``attempt=next_attempt``.
    Transport is deliberately withheld until the current measured orientation
    itself fits.  After transport, call ``reobserve_insertion`` with ``route``.
    Reaching the correction budget still permits a current-pose transport;
    otherwise it fails closed without proposing another wrist correction.
    """
    if (type(attempt) is not int or type(max_attempts) is not int
            or not 0 <= attempt <= max_attempts or not 0 <= max_attempts <= 4):
        raise ValueError("Reorientation attempts must be integers with 0 <= attempt <= max_attempts <= 4")
    previous = previous_route or {}
    route_before = previous.get("route", previous)
    report = {"kind":"observed_shape_bounded_reorientation",
        "observed_time_s":float(state["time"]), "attempt":attempt,
        "max_attempts":max_attempts, "warnings":[], "orientation_trials":[],
        "physical_success":False, "runtime_execution_performed":False,
        "maximum_single_correction_degrees":35.,
        "candidate_axes":"world vertical and current mouth normal",
        "current_orientation_has_priority":True}

    def reject(message):
        report["warnings"].append(message)
        report["geometry_screen_candidates"] = 0
        if output is not None:
            Path(output).write_text(json.dumps(report,indent=2)+"\n")
        raise NoInsertionRoute(report)

    if previous.get("next_observation") == "after_orientation":
        if attempt != previous.get("next_attempt"):
            reject("Correction counter must continue from the previous plan; it cannot be reset.")
        if max_attempts != previous.get("max_attempts"):
            reject("The correction budget cannot change during the observation loop.")
        if float(state["time"]) <= float(previous["observed_time_s"]):
            reject("A new physical state is required after the correction; the prior snapshot is stale.")

    # candidates() validates complete mesh topology/state and applies its
    # unchanged admissibility gates before any robot route can be accepted.
    planner = Planner(config)
    _, actual_rotation = planner.fk("right",np.asarray(state["q_right"])[:7])
    if previous.get("next_observation") == "after_orientation":
        expected = Rotation.from_quat(route_before["rotation_xyzw"])
        error = float((expected*actual_rotation.inv()).magnitude())
        report["previous_correction_orientation_error_rad"] = error
        if error > .015:
            reject("The previous correction wrist target was not reached within 0.015 rad.")
    nearest = float(np.linalg.norm(np.asarray(state["bear"])-np.asarray(state["tcp_right"]),axis=1).min())
    report["nearest_bear_to_right_tcp_m"] = nearest
    if not np.isfinite(nearest) or nearest > .045:
        reject("The observed teddy is farther than 45 mm from the right TCP; a retained grasp is not established.")

    clearance = float(route_before.get("bear_clearance_above_rim_m",.04))
    preferred = route_before.get("approach","vertical")
    if preferred not in ("vertical","mouth_normal"):
        raise ValueError("Unknown previous approach")
    approaches = (preferred,"mouth_normal" if preferred=="vertical" else "vertical")
    axes = [("world_vertical",np.array([0.,0.,1.]))]
    selected = None
    angles = (0.,5.,10.,15.,20.,25.,30.,35.) if attempt < max_attempts else (0.,)
    for angle in angles:
        options = [("current_orientation",0.,actual_rotation)] if angle==0 else [
            (name,sign*angle,Rotation.from_rotvec(axis*np.deg2rad(sign*angle))*actual_rotation)
            for name,axis in axes for sign in (-1.,1.)]
        viable = []
        for axis_name,signed_angle,rotation in options:
            observed = candidates(rest,state,config,clearances=(clearance,),
                approaches=approaches,align_yaw=False,
                fixed_rotation_xyzw=rotation.as_quat(),screen_geometry_first=True,
                plan_mode="full",max_feasible_routes=2)
            trial = {"axis":axis_name,"angle_degrees":signed_angle,
                "rotation_xyzw":rotation.as_quat().tolist(),"report":observed}
            report["orientation_trials"].append(trial)
            if angle==0:
                report["current_rotation_xyzw"] = actual_rotation.as_quat().tolist()
                report["mouth"] = observed["mouth"]
                # State-wide warnings cannot be repaired by a wrist rotation.
                if observed["warnings"]:
                    reject("Observed state is inadmissible: "+" ".join(observed["warnings"]))
                normal=np.asarray(observed["mouth"]["fitted_normal_up"])
                if abs(float(normal@axes[0][1])) < 1.-1e-10:
                    axes.append(("mouth_normal",normal))
            for candidate in observed["candidates"]:
                if candidate["geometry_screen_passes"]:
                    viable.append((candidate,trial))
        if viable:
            # Angular levels are visited in ascending order and current pose
            # short-circuits all correction searches.  Tie-break only within
            # the same smallest successful angular level.
            viable.sort(key=lambda pair:(pair[0]["required_predicted_planar_compression"],
                int(pair[0]["approach"]!=preferred),pair[0]["total_motion_duration_s"]))
            selected, selected_trial = viable[0]
            break
    if selected is None:
        reason = ("Correction budget exhausted and the current measured pose has no admissible route."
                  if attempt >= max_attempts else
                  "No current-pose or bounded small-rotation route passed the existing gates.")
        reject(reason)

    rotation_change=float((Rotation.from_quat(selected["rotation_xyzw"])*actual_rotation.inv()).magnitude())
    correction = rotation_change > .002
    stages = selected["stages"]
    # Keep the validated height stage, including the tiny same-position hold,
    # so execution follows the branch that was actually preflighted.
    executable = stages[:2] if correction else [stages[0],stages[2]]
    report["geometry_screen_candidates"] = 1
    report["selected_rotation_change_rad"] = rotation_change
    report["selected_axis"] = selected_trial["axis"]
    report["selected_angle_degrees"] = selected_trial["angle_degrees"]
    result = {"route":selected,"report":report,"stages_to_execute":executable,
        "observed_time_s":float(state["time"]),"attempt":attempt,
        "next_attempt":attempt+int(correction),"max_attempts":max_attempts,
        "orientation_correction_required":correction,
        "next_observation":"after_orientation" if correction else "before_insertion",
        "requires_reobservation_after_orientation":correction,
        "requires_reobservation_before_insertion":True,
        "reobserve_before_every_stage":False,
        "observe_after_stages":["rotate_held_teddy"] if correction else ["translate_outside_mouth"],
        "physical_success":False,"requires_physical_validation":True,
        "requires_runtime_grasp_and_deformation_gates":True,
        "compression_is_a_prediction_not_applied_deformation":True}
    if output is not None:
        Path(output).write_text(json.dumps(result,indent=2)+"\n")
    return result


def reobserve_insertion(rest, state, config, selected_route, *, output=None):
    """Recheck insertion from fresh observed state after the preparation moves.

    Keep the selected wrist orientation and approach, but recompute the mouth,
    payload offset, outside point and insertion point. Only the direct motion
    from the observed robot state to the new insertion point is preflighted.
    The actual payload remains a contact-driven soft body throughout.
    """
    route=selected_route.get("route",selected_route)
    requested_rotation=Rotation.from_quat(route["rotation_xyzw"])
    planner=Planner(config)
    qs={s:np.asarray(state["q_"+s],float).reshape(-1)[:7] for s in ("left","right")}
    _,actual_rotation=planner.fk("right",qs["right"])
    orientation_error=float((requested_rotation*actual_rotation.inv()).magnitude())
    if orientation_error > .015:
        report={"kind":"fresh_insertion_rejected","warnings":[
            "The preparation wrist orientation has not been reached within 0.015 rad"],
            "observed_orientation_error_rad":orientation_error,"geometry_screen_candidates":0}
        if output is not None: Path(output).write_text(json.dumps(report,indent=2)+"\n")
        raise NoInsertionRoute(report)
    report=candidates(rest,state,config,
        clearances=(route["bear_clearance_above_rim_m"],),
        approaches=(route["approach"],),align_yaw=False,
        fixed_rotation_xyzw=route["rotation_xyzw"],
        fixed_offset_strategy=route.get("in_plane_offset_strategy"),
        screen_geometry_first=True,plan_mode="insert_only",max_feasible_routes=1)
    report["kind"]="fresh_observed_state_insertion_geometry"
    report["observed_orientation_error_rad"]=orientation_error
    viable=[c for c in report["candidates"] if c["geometry_screen_passes"]]
    if output is not None: Path(output).write_text(json.dumps(report,indent=2)+"\n")
    if not viable: raise NoInsertionRoute(report)
    chosen=viable[0]
    return {"insert_stage":chosen["stages"][0],"outside_tcp_m":chosen["above_tcp_m"],
        "route":chosen,"report":report,"physical_success":False,
        "requires_physical_validation":True,
        "requires_runtime_grasp_and_deformation_gates":True,
        "requires_contact_deformation_trial":chosen["required_predicted_planar_compression"]>0,
        "compression_is_a_prediction_not_applied_deformation":True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path,help="Run directory containing rest.npz and config.json")
    parser.add_argument("--state",type=Path,help="A phase snapshot .npz; default is last frame of states.npz")
    parser.add_argument("--output",type=Path,help="Default: insertion_candidates.json in run directory")
    args=parser.parse_args()
    rest=dict(np.load(args.run/"rest.npz",allow_pickle=False))
    raw=dict(np.load(args.state or args.run/"states.npz",allow_pickle=False))
    state={k:raw[k][-1] if np.asarray(raw["time"]).ndim else raw[k]
           for k in ("time","bag","bear","q_left","q_right","tcp_left","tcp_right")}
    result=candidates(rest,state,json.loads((args.run/"config.json").read_text()))
    output=args.output or args.run/"insertion_candidates.json"
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"output":str(output),"warnings":result["warnings"],
        "robot_feasible_candidates":result["robot_feasible_candidates"],
        "geometry_screen_candidates":result["geometry_screen_candidates"]}))


if __name__=="__main__":
    main()
