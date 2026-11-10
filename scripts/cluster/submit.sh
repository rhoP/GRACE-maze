#!/bin/bash
# Presets on the SLURM cluster (docs/plan.md §6). All CPU arrays.
#
#   bash scripts/cluster/submit.sh p1      # P1 on P1_LEVELS (default 25) levels per split, one split per task
#   bash scripts/cluster/submit.sh bank    # skill bank: 1000 mtrain levels in 20 shards -> runs/bank/bank_<k>.pkl
#   bash scripts/cluster/submit.sh pilot   # pref.py, seeds 0-1 (needs runs/bank): SKILL_STEPS, FLAT_STEPS, MGR_STEPS
#   bash scripts/cluster/submit.sh mgr     # manager variants base nr aa nraa x seeds 0-1, abstract arms only
#   bash scripts/cluster/submit.sh final   # pref.py, fresh seeds 2-4, budgets from docs/plan.md (env vars)
#   DEPEND=<jobid> bash scripts/cluster/submit.sh pilot   # start after that job (e.g. the bank) succeeds
#   DRY_RUN=1 bash scripts/cluster/submit.sh pilot
set -euo pipefail
cd "$(dirname "$0")/../.."
exp=${1:?usage: submit.sh <p1|bank|pilot|mgr|final>}
mkdir -p runs/slurm
CPUS=4
PREF=(--bank runs/bank --skill-steps "${SKILL_STEPS:-30000000}" --flat-steps "${FLAT_STEPS:-200000000}"
      --manager-steps "${MGR_STEPS:-1000000}" --eval-levels 50 --manager-levels 1000 --flat-levels 1000)
case "$exp" in
    p1)    N=4; SCRIPT=scripts/gates.py; MEM=${MEM:-8G}
           ARGS=(--p1 --p1-levels "${P1_LEVELS:-25}" --split-task --out runs/gates) ;;
    bank)  N=20; SCRIPT=scripts/make_bank.py; MEM=${MEM:-8G}
           ARGS=(--levels 1000 --shards "$N" --out runs/bank) ;;
    pilot) N=2; SCRIPT=scripts/pref.py; MEM=${MEM:-32G}; CPUS=8
           ARGS=(--out "${OUT:-runs/pilot}" "${PREF[@]}") ;;
    mgr)   N=8; SCRIPT=scripts/pref.py; MEM=${MEM:-8G}
           ARGS=(--out "${OUT:-runs/mgr}" "${PREF[@]}" --arms abstract --variants base nr aa nraa) ;;
    final) N=3; SCRIPT=scripts/pref.py; MEM=${MEM:-32G}; CPUS=8
           ARGS=(--out "${OUT:-runs/final}" "${PREF[@]}" --seed-offset 2) ;;
    *) echo "unknown preset $exp" >&2; exit 1 ;;
esac
cmd=(sbatch --parsable --chdir="$PWD" --array="0-$((N - 1))%${MAX_PARALLEL:-20}" -o "runs/slurm/maze-${exp}-%A_%a.out"
     --mem="$MEM" --cpus-per-task="$CPUS" --time="${TIME:-24:00:00}" --export=ALL,SCRIPT=$SCRIPT
     ${DEPEND:+--dependency=afterok:$DEPEND}
     scripts/cluster/array.sbatch "${ARGS[@]}")
if [ -n "${DRY_RUN:-}" ]; then echo "${cmd[*]}"; else echo "$exp: job $("${cmd[@]}") ($N tasks)"; fi
