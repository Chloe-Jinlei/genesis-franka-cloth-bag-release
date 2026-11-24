#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python_bin="$project_dir/.venv/bin/python"
if [[ ! -x "$python_bin" ]]; then
  echo 'Missing environment. Run: bash scripts/setup_environment.sh' >&2
  exit 1
fi
export PYTHONPATH="$project_dir/teddy_bag_flat_start/vendor${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export GS_CACHE_FILE_PATH="$project_dir/teddy_bag_flat_start/.cache/genesis"
export XDG_CACHE_HOME="$project_dir/teddy_bag_flat_start/.cache"
cd -- "$project_dir"
case "${1:-}" in
  release)
    shift
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    exec "$python_bin" "$project_dir/scripts/run_recorded_stages.py" --stage 20 "$@"
    ;;
  recorded)
    shift
    export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    exec "$python_bin" "$project_dir/scripts/run_recorded_stages.py" "$@"
    ;;
  *)
    show_viewer=0
    forward_args=()
    for argument in "$@"; do
      case "$argument" in
        --headless|--no-viewer) show_viewer=0 ;;
        --viewer) show_viewer=1 ;;
        *) forward_args+=("$argument") ;;
      esac
    done
    if (( show_viewer )); then
      forward_args+=(--viewer)
    else
      export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
    fi
    cd -- "$project_dir/teddy_bag_flat_start"
    exec "$python_bin" "$project_dir/teddy_bag_flat_start/run.py" "${forward_args[@]}"
    ;;
esac
