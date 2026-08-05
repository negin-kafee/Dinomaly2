#!/usr/bin/env bash
#SBATCH --job-name=dino_diag
#SBATCH --partition=cpuq
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=/project/detectanomaly/training/repos/Dinomaly2/slurm/logs/diag_%j.out
#SBATCH --error=/project/detectanomaly/training/repos/Dinomaly2/slurm/logs/diag_%j.err

# READ-ONLY diagnosis: subject overlap + extended threshold grid.
set -euo pipefail
source /project/detectanomaly/envs/strega-cu118-py311/bin/activate
cd /project/detectanomaly/training/repos/Dinomaly2
python tools/diagnose_threshold.py
