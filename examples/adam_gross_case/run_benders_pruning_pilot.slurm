#!/bin/bash
#SBATCH --job-name=adam_bprune50
#SBATCH --output=logs/bprune50_%j.out
#SBATCH --error=logs/bprune50_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=600G
#SBATCH --constraint=amd
#SBATCH --time=1-12:00:00

#SBATCH hetjob

#SBATCH --nodes=12
#SBATCH --ntasks=156
#SBATCH --ntasks-per-node=13
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --constraint=amd
#SBATCH --time=1-12:00:00

set -euo pipefail

: "${ADAM_PRUNE_ROOT:?Set ADAM_PRUNE_ROOT to the isolated scratch run root}"

module purge
module load julia/1.11.3
module load gurobi/13.0.0

export MACROENERGY_PROJECT="${ADAM_PRUNE_ROOT}/MacroEnergy"
export MACROENERGY_SOLVERS="${ADAM_PRUNE_ROOT}/MacroEnergySolvers"
export MACROENERGY_CASE_PATH="${ADAM_PRUNE_ROOT}/case"
export JULIA_CPU_TARGET=generic
export JULIA_DEPOT_PATH="${ADAM_PRUNE_ROOT}/julia_depot:${HOME}/.julia"
export JULIA_PKG_PRECOMPILE_AUTO=0
export JULIA_NUM_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export BENDERS_FARKAS_DEBUG=false

mkdir -p "${MACROENERGY_CASE_PATH}/logs"

# SlurmClusterManager must launch Julia workers only in heterogeneous group 1.
SRUN_WRAPPER=$(mktemp -d)
trap 'rm -rf "${SRUN_WRAPPER}"' EXIT
cat > "${SRUN_WRAPPER}/srun" <<'WRAPPER'
#!/bin/bash
exec /usr/bin/srun --het-group=1 "$@"
WRAPPER
chmod +x "${SRUN_WRAPPER}/srun"
export PATH="${SRUN_WRAPPER}:${PATH}"

# Present the worker component, rather than the driver component, to
# SlurmClusterManager. This mapping is required for heterogeneous Julia jobs.
export SLURM_NTASKS="${SLURM_NTASKS_HET_GROUP_1}"
export SLURM_CPUS_PER_TASK="${SLURM_CPUS_PER_TASK_HET_GROUP_1:-1}"
export SLURM_JOB_NODELIST="${SLURM_JOB_NODELIST_HET_GROUP_1}"
export SLURM_NTASKS_PER_NODE="${SLURM_NTASKS_PER_NODE_HET_GROUP_1}"

: "${SLURM_NTASKS:?SLURM_NTASKS is required}"
: "${SLURM_JOB_NODELIST:?SLURM_JOB_NODELIST is required}"

echo "=== ADAM BENDERS PRUNING PILOT PROVENANCE ==="
date --iso-8601=seconds
hostname
julia --version
gurobi_cl --version
printf 'driver_cpus=%s\n' "${SLURM_CPUS_PER_TASK_HET_GROUP_0:-unknown}"
printf 'worker_tasks=%s\n' "${SLURM_NTASKS_HET_GROUP_1:-unknown}"
printf 'worker_nodes=%s\n' "${SLURM_JOB_NODELIST_HET_GROUP_1:-unknown}"
printf 'mapped_worker_tasks=%s\n' "${SLURM_NTASKS}"
printf 'JULIA_DEPOT_PATH=%s\n' "${JULIA_DEPOT_PATH}"
git -C "${MACROENERGY_PROJECT}" rev-parse HEAD
git -C "${MACROENERGY_SOLVERS}" rev-parse HEAD
git -C "${MACROENERGY_PROJECT}" status --short
git -C "${MACROENERGY_SOLVERS}" status --short
sha256sum \
  "${MACROENERGY_CASE_PATH}/settings/benders_settings.json" \
  "${MACROENERGY_SOLVERS}/src/benders/algorithms.jl" \
  "${MACROENERGY_SOLVERS}/src/benders/subproblems.jl"
echo "=== END PROVENANCE ==="

stdbuf -oL -eL julia \
  --startup-file=no \
  --project="${MACROENERGY_PROJECT}" \
  "${MACROENERGY_SOLVERS}/examples/adam_gross_case/run_benders_pruning_pilot.jl"
