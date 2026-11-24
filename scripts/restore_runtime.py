#!/usr/bin/env python3
"""Restore verified runtime/evidence archives into their documented directories."""
from __future__ import annotations
import argparse
from pathlib import Path
import shutil
import sys
import tarfile
import tempfile
from fetch_release import ROOT, GROUP_KEYS, assets_for, file_matches, load_index

DESTINATIONS = {"genesis_source": Path("genesis-world"),
                "native_vendor": Path("teddy_bag_flat_start/vendor"),
                "evidence": Path("teddy_bag_flat_start/runs")}


def restore(key, asset, archive_dir):
    archive = archive_dir / asset["file"]
    if not file_matches(archive, asset):
        raise ValueError(f"Missing or mismatched archive: {archive}; run fetch_release.py first")
    relative = DESTINATIONS[key]
    destination = ROOT / relative
    if destination.is_symlink():
        raise ValueError(f"Refusing symbolic-link destination: {relative}")
    if destination.exists():
        print(f"Preserving existing {relative}; contents have not been revalidated against this archive")
        return
    with tempfile.TemporaryDirectory(prefix=".restore-", dir=ROOT) as temporary:
        staging = Path(temporary)
        with tarfile.open(archive, "r:gz") as stream:
            for member in stream.getmembers():
                path = Path(member.name)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Unsafe archive member: {member.name}")
                # Allow parent directory entries, but no files outside the selected subtree.
                inside = path == relative or relative in path.parents
                parent_directory = member.isdir() and path in relative.parents
                if not (inside or parent_directory):
                    raise ValueError(f"Unexpected archive member: {member.name}")
            stream.extractall(staging, filter="data")
        source = staging / relative
        if not source.is_dir() or source.is_symlink():
            raise ValueError(f"Archive lacks expected directory: {relative}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
    print(f"Restored {relative}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", choices=GROUP_KEYS, default="runtime")
    ap.add_argument("--archives", type=Path, default=ROOT / "release-assets")
    args = ap.parse_args()
    for key, asset in assets_for(load_index(), args.group):
        restore(key, asset, args.archives.resolve())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, tarfile.TarError) as exc:
        print(f"Restore failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
