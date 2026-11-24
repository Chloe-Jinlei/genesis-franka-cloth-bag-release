#!/usr/bin/env python3
"""Execute recorded final-stage commands from the published state/checkpoints.

These are separate native simulations from published snapshots. They are not an
exact continuous replay of the complete scene. Stage 18 originally stopped at a
torque gate; stage 19 ended while grasping; stage 20 opens the fingers by explicit
user instruction, without a containment prerequisite or placement-success claim.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
SCENE = ROOT / "teddy_bag_flat_start"
RUNS = SCENE / "runs"


def require_evidence(stage):
    files = [RUNS / "flat_full_17/config.json", RUNS / f"final_stage_{stage}/status.json"]
    if stage == "18":
        files += [RUNS / "flat_full_17" / name for name in ("states.npz", "rest.npz", "joint_trace.npy")]
    else:
        previous = RUNS / f"final_stage_{int(stage)-1}"
        files += [previous / name for name in ("status.json", "config.json", "rest.npz", "live_control_state.npz")]
        files += [previous / "native_workspace/dump"]
    missing = [str(path.relative_to(ROOT)) for path in files if not path.exists()]
    if missing:
        raise RuntimeError("Missing recorded evidence: " + ", ".join(missing) +
            ". Run scripts/fetch_release.py --group evidence and scripts/restore_runtime.py --group evidence first.")


def run_one(stage, output):
    require_evidence(stage)
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a new output directory; preserving existing {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    os.environ.setdefault("GS_CACHE_FILE_PATH", str(SCENE / ".cache/genesis"))
    os.environ.setdefault("XDG_CACHE_HOME", str(SCENE / ".cache"))
    sys.path.insert(0, str(SCENE / "runtime_ipc"))
    sys.path.insert(0, str(SCENE))
    from final_stage_live import LiveStage
    source_status = json.loads((RUNS / f"final_stage_{stage}/status.json").read_text())
    native_source = None if stage == "18" else str(RUNS / f"final_stage_{int(stage)-1}")
    options = SimpleNamespace(source=str(RUNS / "flat_full_17"), time=46.815,
        output=str(output), native_checkpoint_source=native_source,
        torque_limit=float(source_status["engineering_contact_limits"]["max_tcp_torque_nm"]),
        direct_release_and_stop=stage == "20")
    live = LiveStage(options)
    live.report["recorded_command_reproduction"] = {
        "recorded_stage": int(stage), "source_status": source_status["status"],
        "source_commands": f"runs/final_stage_{stage}/status.json",
        "uses_published_source_snapshot": True,
        "exact_continuous_whole_scene_replay": False,
        "new_simulation_not_reuse_of_recorded_results": True}
    try:
        commands = [event["command"] for event in source_status["events"]]
        if not commands:
            raise ValueError("Published stage has no commands")
        if stage == "20" and (len(commands) != 1 or commands[0].get("type") != "release_direct"):
            raise ValueError("Published stage 20 is not the documented single direct-release action")
        for packet in commands:
            live.execute(packet)
            if live.paused:
                break
        if stage == "20" and live.release_completed_time is not None and not live.paused:
            live.save("release_completed")
        elif not live.paused:
            live.save("stopped")
        result = {"stage": int(stage), "output": str(output), "status": live.report["status"],
                  "error": live.report.get("error"),
                  "release_action_completed": bool(live.report.get("release_action_completed", False)),
                  "placement_success_claimed": False,
                  "scope": "Separate native run from the published source snapshot; not exact continuous whole-scene replay"}
        print(json.dumps(result, indent=2), flush=True)
        # A physical guard stop remains a nonzero result; it is never recast as success.
        return 2 if live.paused else 0
    except Exception as exc:
        live.save("paused_after_error", str(exc))
        raise
    finally:
        live.writer.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", choices=("18", "19", "20", "all"), default="20")
    ap.add_argument("--output", type=Path, required=True, help="New output directory; existing paths are never overwritten")
    ap.add_argument("--dry-run", action="store_true", help="Show sources and commands without importing the simulator")
    args = ap.parse_args()
    stages = ("18", "19", "20") if args.stage == "all" else (args.stage,)
    for stage in stages:
        require_evidence(stage)
    if args.output.exists():
        raise FileExistsError(f"Choose a new output directory; preserving existing {args.output}")
    if args.dry_run:
        print(json.dumps({"stages": [{"stage": s,
            "state_source": str(RUNS / ("flat_full_17" if s == "18" else f"final_stage_{int(s)-1}")),
            "commands": json.loads((RUNS / f"final_stage_{s}/status.json").read_text())["events"]}
            for s in stages], "exact_continuous_replay": False,
            "note": "Each stage starts from its published source state, not a newly rerun preceding stage."}, indent=2))
        return 0
    if args.stage != "all":
        return run_one(args.stage, args.output)
    args.output.mkdir(parents=True)
    codes = {}
    # Isolate native worlds and release CUDA resources between stages.
    for stage in stages:
        child = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--stage", stage,
                                "--output", str(args.output.resolve() / f"stage_{stage}")], cwd=ROOT)
        codes[stage] = child.returncode
        if child.returncode not in (0, 2):
            break
    result = {"stage_exit_codes": codes,
              "note": "Each stage used its published source snapshot. Exit 2 preserves a physical guard stop; recorded stage 18 originally stopped at its torque gate."}
    (args.output / "reproduction-summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if len(codes) == 3 and all(code == 0 for code in codes.values()) else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"Recorded-stage execution stopped: {exc}", file=sys.stderr)
        raise SystemExit(1)
