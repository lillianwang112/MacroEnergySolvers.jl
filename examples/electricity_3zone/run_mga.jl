using Pkg
Pkg.activate(joinpath(ENV["HOME"], "Documents", "MacroEnergy.jl"))
Pkg.develop(path=joinpath(ENV["HOME"], "Documents", "MacroEnergySolvers.jl"))

using MacroEnergy
using Gurobi
using CSV
using DataFrames
using JSON3
using Dates

(case, solution, mga_results, mga_vectors, mga_var_names) = MacroEnergy.run_case(
    @__DIR__;
    planning_optimizer=Gurobi.Optimizer,
    subproblem_optimizer=Gurobi.Optimizer,
    planning_optimizer_attributes=("Method" => 2, "Crossover" => 0, "BarConvTol" => 1e-4),
    subproblem_optimizer_attributes=("Method" => 2, "Crossover" => 1, "BarConvTol" => 1e-4),
    run_mga=true
)

# Serialize every MGA solution's capacity vector + cost so the ensemble survives
# past this process (needed for later spectral/DPP analysis, not just single-run inspection).
mga_settings = JSON3.read(read(joinpath(@__DIR__, "settings", "benders_settings.json"), String))
out_dir = joinpath(@__DIR__, "mga_ensembles")
mkpath(out_dir)

n_iter = length(mga_results)

# Fail loud rather than silently zero-filling: a missing key here means mga_var_names
# and the solution dict have diverged (typo, presolve, stale variable list), which is
# a bug to fix, not paper over.
for v in mga_var_names
    haskey(mga_results[1].planning_sol.values, v) || error("Variable '$v' in mga_var_names is missing from planning_sol.values — investigate before proceeding.")
end

df = DataFrame(mga_var_names .=> [[mga_results[it].planning_sol.values[v] for it in 1:n_iter] for v in mga_var_names])
insertcols!(df, 1, :iteration => 1:n_iter)
insertcols!(df, 2, :total_cost => [mga_results[it].planning_sol.planning_cost for it in 1:n_iter])

# Direction vectors used to generate each iteration (rows = variable, cols = iteration in
# mga_vectors as returned by benders_mga) — needed to later trace *why* a solution landed
# where it did in the spectral embedding, e.g. for the method-signature hypothesis.
vecs_df = DataFrame(mga_vectors', mga_var_names)
insertcols!(vecs_df, 1, :iteration => 1:n_iter)

tag = "method$(mga_settings.MGAMethod)_seed$(mga_settings.MGARandomSeed)_iters$(n_iter)"
CSV.write(joinpath(out_dir, "$(tag).csv"), df)
CSV.write(joinpath(out_dir, "$(tag)_directions.csv"), vecs_df)

# Provenance: without this, a CSV from today is untraceable once the model/solver settings
# change next year.
provenance = Dict(
    "mga_settings" => mga_settings,
    "n_iterations" => n_iter,
    "macroenergy_commit" => strip(read(`git -C $(dirname(@__DIR__)) rev-parse HEAD`, String)),
    "macroenergysolvers_commit" => strip(read(`git -C $(joinpath(ENV["HOME"], "Documents", "MacroEnergySolvers.jl")) rev-parse HEAD`, String)),
    "gurobi_version" => string(Gurobi.GRB_VERSION_MAJOR, ".", Gurobi.GRB_VERSION_MINOR, ".", Gurobi.GRB_VERSION_TECHNICAL),
    "timestamp" => string(now()),
)
open(joinpath(out_dir, "$(tag)_provenance.json"), "w") do io
    JSON3.write(io, provenance)
end

@info "Wrote $(n_iter) MGA solutions to $(joinpath(out_dir, "$(tag).csv"))"
