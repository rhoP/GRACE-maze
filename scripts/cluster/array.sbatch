#!/bin/bash
# One task of an array: python $SCRIPT "$@" --task $SLURM_ARRAY_TASK_ID. Submit through submit.sh.
#SBATCH --account=RPTU-HybSol
#SBATCH -J maze
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --mail-type=FAIL
set -euo pipefail
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] || { echo "submit through scripts/cluster/submit.sh" >&2; exit 2; }
echo "task ${SLURM_ARRAY_TASK_ID} of job ${SLURM_ARRAY_JOB_ID} on ${HOSTNAME}: ${SCRIPT} $*"
set +eu
source "${MAZE_VENV:-.venv}/bin/activate" || { echo "cannot activate ${MAZE_VENV:-.venv}" >&2; exit 1; }
set -eu
export PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=3
if [ -n "${NEED_GPU:-}" ]; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
    python -c "import jax, jax.numpy as jnp; assert jax.default_backend() == 'gpu', jax.devices(); \
print('jax', jax.__version__, jax.devices(), float((jnp.ones((256, 256)) @ jnp.ones((256, 256))).sum()))" \
        || { echo "JAX cannot use the GPU" >&2; exit 3; }
else
    export JAX_PLATFORMS=cpu  # CPU node: skip the CUDA plugin (its init error was only noise in the .err files)
fi
python "$SCRIPT" "$@" --task "${SLURM_ARRAY_TASK_ID}"
