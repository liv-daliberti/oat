#!/bin/bash
# GPU smoke test for the posit fork of OAT, on ionic.
#   sbatch --partition=all --account=allcs --gres=gpu:a6000:1 test/gpu_smoke.sbatch
#SBATCH --job-name=oat-gpu-smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=/n/fs/similarity/od2961/posit_cache/oat-gpu-smoke-%j.log
set -uo pipefail
export HF_HOME=/n/fs/similarity/od2961/posit_cache/hf
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export XDG_CACHE_HOME=/n/fs/similarity/od2961/posit_cache/.cache
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton-$USER-${SLURM_JOB_ID:-0}"
echo "node=$(hostname -s)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/n/fs/similarity/posit/src/.venv/bin/python /n/fs/similarity/oat/test/gpu_smoke.py
echo "exit=$?"
