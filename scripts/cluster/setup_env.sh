#!/bin/bash
# One-time setup on the cluster login node: a venv with JAX on the CPU and MuJoCo / MJX.
#
#   MAZE_ROOT=/scratch/rohit/Projects/GRACE-maze bash scripts/cluster/setup_env.sh
set -euo pipefail
ROOT=${MAZE_ROOT:-/scratch/rohit/Projects/GRACE-maze}
VENV=${MAZE_VENV:-.venv}
cd "$ROOT"
[ -d "$VENV" ] || python3 -m venv "$VENV"
source "$VENV/bin/activate"
python -c "import sys; assert sys.version_info >= (3, 10), sys.version"
pip install --upgrade pip
pip install -e ".[dev]"
python -c "import jax, flax, mujoco; from mujoco import mjx; import maze.game; \
print('ok: jax', jax.__version__, '| flax', flax.__version__, '| mujoco', mujoco.__version__)"
