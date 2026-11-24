#!/usr/bin/env python3
"""Check packaged source/assets and installed versions without simulating physics."""
from __future__ import annotations
import argparse
import ast
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "teddy_bag_flat_start"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets-only", action="store_true", help="Check portable files and MJCF only; do not import Genesis/Torch/uipc")
    ap.add_argument("--gpu", action="store_true", help="Also query CUDA availability; no simulation or scene build")
    args = ap.parse_args()
    if args.assets_only and args.gpu:
        ap.error("--assets-only and --gpu are mutually exclusive")
    checks, info = {}, {}
    cfg = json.loads((SCENE / "config.json").read_text())
    for path in list(SCENE.glob("*.py")) + list((SCENE / "runtime_ipc").glob("*.py")):
        ast.parse(path.read_text(), filename=str(path))
    checks["python_sources_parse"] = True
    for name in (cfg["bag"]["mesh"], cfg["bag"]["npz_file"], cfg["bag"]["metadata"],
                 cfg["bear"]["mesh"], "cotton_canvas.png", "teddy_fur.png"):
        checks["asset:" + name] = Path(name).name == name and (SCENE / "assets" / name).is_file()
    checks["native_vendor_is_not_external_symlink"] = not (SCENE / "vendor").is_symlink()
    xml = SCENE / "assets/panda_tactile.xml"
    tree = ET.parse(xml).getroot()
    meshdir = Path(tree.find("compiler").attrib.get("meshdir", "."))
    checks["mesh_directory_relative"] = not meshdir.is_absolute()
    for mesh in tree.findall(".//mesh"):
        if "file" in mesh.attrib:
            name = Path(mesh.attrib["file"])
            checks["mesh:" + name.name] = not name.is_absolute() and (xml.parent / meshdir / name).is_file()
    for path in (SCENE / "assets" / cfg["bag"]["mesh"], SCENE / "assets" / cfg["bear"]["mesh"]):
        for line in path.read_text().splitlines():
            if line.startswith("mtllib "):
                material = path.parent / line.split(maxsplit=1)[1]
                checks["material:" + material.name] = material.is_file()
                if material.is_file():
                    for row in material.read_text().splitlines():
                        if row.startswith("map_Kd "):
                            name = Path(row.split(maxsplit=1)[1])
                            checks["texture:" + name.name] = not name.is_absolute() and (material.parent / name).is_file()
    if not all(checks.values()):
        print(json.dumps({"status": "failed", "checks": checks}, indent=2))
        raise SystemExit(1)
    import mujoco
    sys.path.insert(0, str(SCENE))
    from motion import Planner
    model = mujoco.MjModel.from_xml_path(str(xml))
    planner = Planner(cfg)
    checks["single_robot_nq_9"] = model.nq == 9
    checks["dual_robot_nq_nv_18"] = planner.m.nq == 18 and planner.m.nv == 18
    if not args.assets_only:
        checks["python_3_12"] = sys.version_info[:2] == (3, 12)
        checks["linux_x86_64"] = platform.system() == "Linux" and platform.machine() == "x86_64"
        sys.path.insert(0, str(SCENE / "vendor"))
        import torch
        import genesis
        import uipc
        versions = {name: metadata.version(name) for name in ("genesis-world", "pyuipc", "torch")}
        checks["genesis_version"] = versions["genesis-world"] == "1.4.3"
        checks["native_uipc_version"] = versions["pyuipc"] == str(uipc.__version__) == "0.0.29"
        checks["torch_distribution_version"] = versions["torch"].split("+", 1)[0] == "2.14.0"
        checks["torch_cuda_build"] = torch.__version__ == "2.14.0+cu130" and torch.version.cuda == "13.0"
        checks["genesis_from_restored_source"] = Path(genesis.__file__).resolve().is_relative_to(ROOT / "genesis-world")
        checks["uipc_from_restored_vendor"] = Path(uipc.__file__).resolve().is_relative_to(SCENE / "vendor")
        info.update(versions=versions, python=platform.python_version(), torch_runtime=torch.__version__,
                    torch_cuda=torch.version.cuda, genesis_source=str(Path(genesis.__file__).resolve()),
                    uipc_source=str(Path(uipc.__file__).resolve()))
        if args.gpu:
            checks["cuda_available"] = torch.cuda.is_available()
            if checks["cuda_available"]:
                info["gpu"] = torch.cuda.get_device_name(0)
    result = {"status": "passed" if all(checks.values()) else "failed", "checks": checks,
              "environment": info,
              "scope": "Source, resource, MJCF and optional runtime imports only; no physics steps, clean-machine install test, or task-success certification."}
    print(json.dumps(result, indent=2))
    if not all(checks.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
