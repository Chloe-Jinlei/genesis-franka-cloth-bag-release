#!/usr/bin/env python3
"""Download checksum-verified assets from this repository's pinned release."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "Chloe-Jinlei/genesis-franka-cloth-bag-release"
RELEASE_TAG = "v1.0.0-direct-release"
GROUP_KEYS = {"runtime": ("genesis_source", "native_vendor"),
              "evidence": ("evidence",),
              "all": ("genesis_source", "native_vendor", "evidence")}


def load_index():
    index = json.loads((ROOT / "environment/release-assets.json").read_text())
    if index.get("repository") != REPOSITORY or index.get("tag") != RELEASE_TAG:
        raise ValueError("Release index repository/tag does not match this package")
    return index


def assets_for(index, group):
    assets, seen = [], set()
    for key in GROUP_KEYS[group]:
        item = index[key]
        name, size, digest = (item.get(k) for k in ("file", "bytes", "sha256"))
        if not isinstance(name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) or name in seen:
            raise ValueError(f"Invalid or duplicate asset name in {key}")
        if type(size) is not int or size <= 0 or not isinstance(digest, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", digest):
            raise ValueError(f"Invalid asset size/digest in {key}")
        seen.add(name)
        assets.append((key, {"file": name, "bytes": size, "sha256": digest.lower()}))
    return assets


def file_matches(path, asset):
    if not path.is_file() or path.is_symlink() or path.stat().st_size != asset["bytes"]:
        return False
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest() == asset["sha256"]


def download(asset, destination):
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / asset["file"]
    if path.is_symlink():
        raise ValueError(f"Refusing symbolic-link destination: {path}")
    if file_matches(path, asset):
        print(f"Verified existing {path.name}", flush=True)
        return
    url = f"https://github.com/{REPOSITORY}/releases/download/{RELEASE_TAG}/{quote(asset['file'], safe='')}"
    request = Request(url, headers={"User-Agent": "genesis-franka-cloth-bag-release/1.0"})
    temporary = None
    print(f"Downloading {asset['file']} ({asset['bytes']:,} bytes)", flush=True)
    try:
        with tempfile.NamedTemporaryFile(prefix=path.name + ".", suffix=".part", dir=destination, delete=False) as output:
            temporary = Path(output.name)
            digest, received = hashlib.sha256(), 0
            with urlopen(request, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    received += len(block)
                    if received > asset["bytes"]:
                        raise ValueError("Download exceeds recorded byte count")
                    output.write(block)
                    digest.update(block)
            if received != asset["bytes"] or digest.hexdigest() != asset["sha256"]:
                raise ValueError(f"Downloaded size/SHA256 mismatch: {path.name}")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        temporary = None
        print(f"Verified {path.name}", flush=True)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--group", choices=GROUP_KEYS, default="runtime")
    ap.add_argument("--output", type=Path, default=ROOT / "release-assets")
    args = ap.parse_args()
    for _, asset in assets_for(load_index(), args.group):
        download(asset, args.output.resolve())
    print("Selected downloads verified; no archive has been extracted.")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as exc:
        print(f"Download failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
