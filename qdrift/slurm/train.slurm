#!/bin/bash
#SBATCH --job-name=qdrift
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --ntasks=1

set -euo pipefail

echo "=== Job Info ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Job Name:  $SLURM_JOB_NAME"
echo "Node:      $(hostname)"
echo "GPU:       $CUDA_VISIBLE_DEVICES"
echo "Date:      $(date)"
echo "Command:   $@"
echo "================"

# Load modules (adjust to your HPC cluster setup)
module purge
module load GCCcore/12.3.0
module load Anaconda3
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate qdrift
echo "Using conda env: qdrift ($(which python))"

# Navigate to project directory
# --chdir in sbatch sets both SLURM_SUBMIT_DIR and the initial cwd to the project root.
PROJECT_DIR="${SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR"

echo "Working dir: $(pwd)"

# Ensure slurm_logs directory exists (relative to project root)
mkdir -p "${PROJECT_DIR}/slurm_logs"

# Set JAX to use GPU
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export MUJOCO_GL=egl

# Force-load CUDA libraries before JAX starts (bypasses dlopen/RPATH issues)
CUDA_LIB="${EBROOTCUDA}/targets/x86_64-linux/lib"
CUDNN_LIB="${EBROOTCUDNN}/lib64"
export LD_PRELOAD="${CUDA_LIB}/libcusparse.so:${CUDA_LIB}/libcublas.so:${CUDA_LIB}/libcufft.so:${CUDA_LIB}/libcusolver.so:${CUDNN_LIB}/libcudnn.so:${LD_PRELOAD:-}"
export LD_LIBRARY_PATH="${CUDA_LIB}:${CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
export XLA_FLAGS="--xla_gpu_cuda_data_dir=${EBROOTCUDA}"

echo "LD_PRELOAD=${LD_PRELOAD}"

# Verify JAX sees GPU
python -c "import jax; devs=jax.devices(); print(f'JAX devices: {devs}'); assert any('gpu' in str(d).lower() or 'cuda' in str(d).lower() for d in devs), 'No GPU found!'"

# OGBench dataset location (pre-downloaded; no internet on compute nodes)
export OGBENCH_DATA_DIR="/path/to/ogbench_data"

# Run the training command (all positional args from sbatch)
python main.py "$@"

echo "=== Done: $(date) ==="
