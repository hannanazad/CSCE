#!/bin/bash
# Submit Q-Drifting runs to an HPC cluster.
#
# Usage:
#   bash scripts/submit_qdrift.sh                                          # all envs, seeds 0-4
#   bash scripts/submit_qdrift.sh --env antmaze-large-navigate-v0 --seeds "0 1 2 3 4"
#   bash scripts/submit_qdrift.sh --dry-run
#   bash scripts/submit_qdrift.sh --debug --env antmaze-large-navigate-v0 --seeds "0"
#   bash scripts/submit_qdrift.sh --env antmaze-large-navigate-v0 --seeds "0 1" \
#       --run-group qdrift_custom -- --agent.q_drift_scale=0.3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SLURM_TEMPLATE="${ROOT_DIR}/slurm/train.slurm"

DRY_RUN=false
DEBUG=false
FILTER_ENV=""
SEEDS="0 1 2 3 4"
RUN_GROUP_OVERRIDE=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)    DRY_RUN=true; shift ;;
        --debug)      DEBUG=true; shift ;;
        --env)        FILTER_ENV="$2"; shift 2 ;;
        --seeds)      SEEDS="$2"; shift 2 ;;
        --run-group)  RUN_GROUP_OVERRIDE="$2"; shift 2 ;;
        --)           shift; EXTRA_ARGS="$*"; break ;;
        *)            echo "Unknown flag: $1"; exit 1 ;;
    esac
done

if $DRY_RUN; then echo "[DRY RUN] Commands will be printed but not submitted."; fi
if $DEBUG; then echo "[DEBUG] Jobs will run with --debug (100 steps)."; fi

AGENT="agents/qdrift.py"
RUN_GROUP="${RUN_GROUP_OVERRIDE:-qdrift}"
OGBENCH_DATA="/path/to/ogbench_data"
DEBUG_FLAG=""
SLURM_TIME="12:00:00"
if $DEBUG; then
    DEBUG_FLAG="--debug"
    SLURM_TIME="00:30:00"
fi

# Ensure slurm logs directory exists
mkdir -p "${ROOT_DIR}/slurm_logs"

submit() {
    local ENV="$1"
    local EXTRA="${2:-}"
    if [[ -n "$FILTER_ENV" && "$ENV" != "$FILTER_ENV" ]]; then return; fi
    for SEED in $SEEDS; do
        local JOB_NAME="qdrift_${ENV}_s${SEED}"
        local CMD="--agent=${AGENT} --env_name=${ENV} --eval_episodes=50 --seed=${SEED} --run_group=${RUN_GROUP} ${EXTRA} ${EXTRA_ARGS} ${DEBUG_FLAG}"
        if $DRY_RUN; then
            echo "  sbatch --job-name=${JOB_NAME} --time=${SLURM_TIME} --chdir=${ROOT_DIR} ${SLURM_TEMPLATE} ${CMD}"
        else
            OUT=$(sbatch --job-name="${JOB_NAME}" --time="${SLURM_TIME}" --chdir="${ROOT_DIR}" "${SLURM_TEMPLATE}" ${CMD} 2>&1)
            if echo "${OUT}" | grep -q 'Submitted batch job'; then
                echo "Submitted: ${JOB_NAME} ($(echo ${OUT} | grep -o '[0-9]*$'))"
            else
                echo "ERROR submitting ${JOB_NAME}: ${OUT}" >&2
                exit 1
            fi
        fi
    done
}

# ========================================================================
# Environments — OGBench evaluation domains (no action chunking)
# ========================================================================

echo "=== Antmaze Navigate ==="
submit "antmaze-medium-navigate-v0"
submit "antmaze-large-navigate-v0"
submit "antmaze-giant-navigate-v0" "--agent.discount=0.995"

echo "=== Antmaze Stitch ==="
submit "antmaze-medium-stitch-v0"
submit "antmaze-large-stitch-v0"
submit "antmaze-giant-stitch-v0" "--agent.discount=0.995"

echo "=== Humanoidmaze Navigate ==="
submit "humanoidmaze-medium-navigate-v0" "--agent.discount=0.995"
submit "humanoidmaze-large-navigate-v0" "--agent.discount=0.995"
submit "humanoidmaze-giant-navigate-v0" "--agent.discount=0.995"

echo "=== Humanoidmaze Stitch ==="
submit "humanoidmaze-medium-stitch-v0" "--agent.discount=0.995"
submit "humanoidmaze-large-stitch-v0" "--agent.discount=0.995"
submit "humanoidmaze-giant-stitch-v0" "--agent.discount=0.995"

echo "Done."
