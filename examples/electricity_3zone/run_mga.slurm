#!/bin/bash
#SBATCH --job-name=mga_3zone
#SBATCH --output=logs/mga_%j.out
#SBATCH --error=logs/mga_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=lw3319@princeton.edu

module purge
module load julia/1.11.3
module load gurobi/11.0.3

export JULIA_NUM_THREADS=${SLURM_CPUS_PER_TASK}

mkdir -p "${SLURM_SUBMIT_DIR}/logs"
cd "${SLURM_SUBMIT_DIR}"

julia run_mga.jl
