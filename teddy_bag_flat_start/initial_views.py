"""Four native camera renders of the unstepped initial simulation state.

Integration (no camera can be added after Genesis Scene.build()):
    cameras.update(add_initial_cameras(scene))   # before scene.build()
    ... scene.build(); configure the existing robot controllers ...
    render_initial_views(scene, cameras, output_dir, config=config)

The render call must precede the first scene.step(). It never integrates,
repositions a camera after build, changes an entity, or edits rendered pixels.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

RESOLUTION = (1600, 1200)
CAMERA_SPECS = (
    ("initial_01_oblique", (1.9, -2.25, 1.95), (0., 0., .96), (0., 0., 1.)),
    ("initial_02_front_close", (.15, -1.8, 1.65), (-.03, 0., 1.08), (0., 0., 1.)),
    ("initial_03_top", (0., 0., 3.15), (0., 0., .74), (0., 1., 0.)),
    ("initial_04_opposite_front", (0., 2.4, 1.85), (0., 0., 1.), (0., 0., 1.)),
)


def add_initial_cameras(scene):
    """Add the four reference-angle cameras before scene.build(), and return them."""
    if scene.is_built:
        raise RuntimeError("Initial-view cameras must be added before scene.build()")
    return {
        name: scene.add_camera(
            model="pinhole", res=RESOLUTION, pos=pos, lookat=lookat, up=up,
            fov=42., GUI=False,
        )
        for name, pos, lookat, up in CAMERA_SPECS
    }


def _clock(scene):
    steps = scene.sim.steps
    if hasattr(steps, "detach"):
        steps = steps.detach().cpu().numpy()
    return float(scene.sim.cur_t), np.asarray(steps).reshape(-1).copy()


def render_initial_views(scene, cameras, output_dir, *, config=None):
    """Save four exact PNG RGB renders plus initial_views.json at time/step zero.

    Pass the dictionary returned by add_initial_cameras, or the full scene camera
    dictionary after updating it with those entries. Existing output files are
    refused so later invocations cannot overwrite previously captured evidence.
    The only writes are the four PNG files and their JSON provenance record.
    """
    if not scene.is_built:
        raise RuntimeError("Build the scene before rendering its initial views")
    start_time, start_steps = _clock(scene)
    if start_time != 0. or np.any(start_steps != 0):
        raise RuntimeError("Initial images require an unstepped scene at time zero")
    missing = [name for name, *_ in CAMERA_SPECS if name not in cameras]
    if missing:
        raise ValueError(f"Missing initial-view cameras: {missing}")
    out = Path(output_dir).expanduser().resolve()
    filenames = [name + ".png" for name, *_ in CAMERA_SPECS]
    metadata_path = out / "initial_views.json"
    existing = [str(path) for path in [metadata_path, *(out / name for name in filenames)] if path.exists()]
    if existing:
        raise FileExistsError(f"Initial-view outputs already exist: {existing}")
    out.mkdir(parents=True, exist_ok=True)
    images = []
    for name, pos, lookat, up in CAMERA_SPECS:
        rgb = cameras[name].render(
            rgb=True, depth=False, segmentation=False, normal=False,
            antialiasing=False, force_render=True,
        )[0]
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.shape != (RESOLUTION[1], RESOLUTION[0], 3):
            raise RuntimeError(f"Unexpected native RGB format for {name}: {rgb.dtype}, {rgb.shape}")
        now, steps = _clock(scene)
        if now != start_time or not np.array_equal(steps, start_steps):
            raise RuntimeError("Simulation time changed while rendering initial views")
        destination = out / (name + ".png")
        Image.fromarray(rgb).save(destination, format="PNG")
        images.append({
            "file": destination.name,
            "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            "width_px": RESOLUTION[0], "height_px": RESOLUTION[1],
            "camera": {"model": "pinhole", "position_m": list(pos),
                       "lookat_m": list(lookat), "up": list(up), "vertical_fov_degrees": 42.},
        })
    end_time, end_steps = _clock(scene)
    metadata = {
        "status": "rendered_unstepped_initial_state",
        "renderer": "Native Genesis camera RGB output",
        "simulation_time_before_s": start_time,
        "simulation_time_after_s": end_time,
        "simulation_steps_before": start_steps.tolist(),
        "simulation_steps_after": end_steps.tolist(),
        "physics_steps_executed_by_this_helper": 0,
        "physical_entity_mutations_by_this_helper": 0,
        "image_postprocessing": "None; lossless PNG encoding only",
        "images": images,
    }
    if config is not None:
        canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        metadata["config_sha256_canonical_json"] = hashlib.sha256(canonical).hexdigest()
        metadata["config"] = config
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return metadata
