using MacroEnergy
using Gurobi
using JuMP

if !(1 <= length(ARGS) <= 2)
    error("Usage: julia dump_monolithic_variables.jl CASE_PATH [OUTPUT_CSV]")
end

case_path = abspath(ARGS[1])
output_path = length(ARGS) == 2 ? abspath(ARGS[2]) : joinpath(case_path, "linking_vars_mono.csv")

isdir(case_path) || error("Case directory does not exist: $case_path")

(_, model) = run_case(
    case_path;
    optimizer=Gurobi.Optimizer,
    optimizer_attributes=(
        "Method" => 2,
        "Crossover" => 1,
        "BarConvTol" => 1e-8,
    ),
)

has_values(model) || error(
    "Monolithic model has no primal solution: " *
    "termination=$(termination_status(model)) " *
    "primal=$(primal_status(model)) " *
    "raw=$(raw_status(model)) " *
    "result_count=$(result_count(model))",
)

variables = filter(v -> !isempty(name(v)), all_variables(model))
isempty(variables) && error(
    "The solved model contains no named variables. Enable " *
    "EnableJuMPStringNames in the case settings before generating the oracle.",
)
sort!(variables; by=name)

mkpath(dirname(output_path))
open(output_path, "w") do io
    println(io, "variable,value")
    for variable in variables
        variable_name = name(variable)
        # JuMP array-variable names can contain commas (for example x[1,2]).
        # Readers split on the final comma, which unambiguously separates the
        # numeric value from the variable name.
        println(io, variable_name, ',', value(variable))
    end
end

println("Wrote $(length(variables)) monolithic variable values to $output_path")
println("Objective value: $(objective_value(model))")
