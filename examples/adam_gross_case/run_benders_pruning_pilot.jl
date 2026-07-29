using MacroEnergy
using MacroEnergySolvers
using Gurobi
using JSON3

const CASE_PATH = get(ENV, "MACROENERGY_CASE_PATH", @__DIR__)
const SETTINGS_PATH = joinpath(CASE_PATH, "settings", "benders_settings.json")
const JOB_ID = get(ENV, "SLURM_JOB_ID", "local")
const BACKUP_PATH = SETTINGS_PATH * ".pruning_backup." * JOB_ID

const PLANNING_ATTRIBUTES = (
    "Method" => 2,
    "Crossover" => 0,
    "BarConvTol" => 1e-6,
    "NumericFocus" => 2,
    "Threads" => 16,
)

const SUBPROBLEM_ATTRIBUTES = (
    "Method" => 1,
    "NumericFocus" => 2,
    "DualReductions" => 0,
    "InfUnbdInfo" => 1,
    "Threads" => 1,
)

println("=== ADAM BENDERS PRUNING PILOT ===")
println("case_path=$CASE_PATH")
println("julia_version=$VERSION")
println("macroenergy=$(pathof(MacroEnergy))")
println("macroenergysolvers=$(pathof(MacroEnergySolvers))")
println("gurobi_version=$(Gurobi._GUROBI_VERSION)")
println("driver_cpus=$(get(ENV, "SLURM_CPUS_PER_TASK_HET_GROUP_0", "unknown"))")
println("worker_tasks=$(get(ENV, "SLURM_NTASKS_HET_GROUP_1", "unknown"))")
println("planning_attributes=$PLANNING_ATTRIBUTES")
println("subproblem_attributes=$SUBPROBLEM_ATTRIBUTES")
println("BENDERS_FARKAS_DEBUG=$(get(ENV, "BENDERS_FARKAS_DEBUG", "unset"))")

settings_original = read(SETTINGS_PATH, String)
isfile(BACKUP_PATH) && error("Refusing to overwrite existing settings backup: $BACKUP_PATH")
write(BACKUP_PATH, settings_original)

try
    settings = JSON3.read(settings_original, Dict{String,Any})
    settings["Distributed"] = true
    settings["MaxIter"] = 50
    settings["MaxCpuTime"] = 115_200
    settings["ConvTol"] = 0.001

    # No cut can be deleted before k=39:
    # activity counting starts at k=35 and requires five consecutive solves.
    settings["CutPruning"] = true
    settings["CutPruningStartIter"] = 35
    settings["CutPruningMinAge"] = 15
    settings["CutPruningInactiveIters"] = 5
    settings["CutPruningKeepRecent"] = 10
    settings["CutPruningAbsTol"] = 1e-6
    settings["CutPruningRelTol"] = 1e-7
    settings["CutPruningMaxDeletesPerIter"] = 156

    open(SETTINGS_PATH, "w") do io
        JSON3.pretty(io, settings)
    end

    println("effective_benders_settings=")
    println(read(SETTINGS_PATH, String))

    (_, solution) = MacroEnergy.run_case(
        CASE_PATH;
        planning_optimizer=Gurobi.Optimizer,
        subproblem_optimizer=Gurobi.Optimizer,
        planning_optimizer_attributes=PLANNING_ATTRIBUTES,
        subproblem_optimizer_attributes=SUBPROBLEM_ATTRIBUTES,
        run_mga=false,
    )

    if hasproperty(solution, :convergence) && !isnothing(solution.convergence)
        convergence = solution.convergence
        println("=== PILOT ALGORITHM COMPLETE ===")
        println("termination_status=$(convergence.termination_status)")
        println("iterations=$(length(convergence.gap_hist))")
        println("final_LB=$(last(convergence.LB_hist))")
        println("final_UB=$(last(convergence.UB_hist))")
        println("final_gap=$(last(convergence.gap_hist))")
    end
catch error_value
    @error "Adam Benders pruning pilot failed" exception=(
        error_value,
        catch_backtrace(),
    )
    rethrow()
finally
    write(SETTINGS_PATH, settings_original)
    rm(BACKUP_PATH; force=true)
    println("Restored benders_settings.json")
end
