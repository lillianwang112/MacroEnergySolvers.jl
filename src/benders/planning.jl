"""
    solve_planning_problem(m::Model, planning_variables::Vector{String})

Solves the planning (upper-level) problem in the Benders decomposition algorithm.

This function attempts to solve the planning problem and handles potential numerical issues,
particularly with negative capacities, by rounding small values to zero if needed.

# Arguments
- `m::Model`: The JuMP model representing the planning problem
- `planning_variables::Vector{String}`: Names of the variables of the planning problem

# Returns
A NamedTuple containing:
- `fixed_cost`: Fixed cost component of the solution 
- `values`: Dictionary mapping linking variable names to their optimal values

# Notes
If negative capacities are detected, the solver will be reconfigured with `Crossover = 1` 
and the problem will be re-solved. If the solution fails, the function will compute
and display conflicting constraints (if the solver supports it) before throwing an error.
"""
function solve_planning_problem(m::Model,planning_variables::Vector{String})

    optimize!(m)

    if !has_values(m)
        # Barrier with Crossover=0 can return INFEASIBLE_OR_UNBOUNDED when the problem
        # is actually feasible but numerically ambiguous. Retry with crossover enabled.
        status1 = termination_status(m)
        @info "Planning solve did not return values (status: $status1). Retrying with crossover enabled."
        try; set_attribute(m, "Crossover", 1);        catch; end
        try; set_attribute(m, "run_crossover", "on"); catch; end
        optimize!(m)
        # Capture status and values BEFORE resetting crossover — set_attribute invalidates solution state
        if has_values(m)
            planning_sol = process_planning_sol(m, planning_variables)
            LB = objective_value(m)
        else
            status2 = termination_status(m)
        end
        try; set_attribute(m, "Crossover", 0);         catch; end
        try; set_attribute(m, "run_crossover", "off"); catch; end

        if !@isdefined(planning_sol)
            # Barrier failed even with crossover — fall back to dual simplex
            @info "Crossover retry failed (status: $status2). Retrying with dual simplex (Method=1)."
            try; set_attribute(m, "Method", 1); catch; end
            optimize!(m)
            if has_values(m)
                planning_sol = process_planning_sol(m, planning_variables)
                LB = objective_value(m)
            else
                status3 = termination_status(m)
            end
            try; set_attribute(m, "Method", 2); catch; end
        end

        if !@isdefined(planning_sol)
            error("Planning solve failed with all methods (barrier: $status1, crossover: $status2, simplex: $status3).")
        end
    else
        planning_sol = process_planning_sol(m, planning_variables)
        LB = objective_value(m)
    end

    return planning_sol, LB
end

function add_approximate_variable_cost!(m::Model, number_of_subproblems::Int)

    if haskey(m, :vTHETA)
        
        @warn "Variable vTHETA already exists in the model. Skipping addition of approximate variable cost because it has already been added."

        unregister(m, :ePlanningCost)

        @expression(m, ePlanningCost, objective_function(m) - sum(m[:vTHETA]))

        drop_zeros!(m[:ePlanningCost])

        return
    end

    @variable(m, vTHETA[w in 1:number_of_subproblems])

    if haskey(m, :eLowerBoundOperatingCost)
        @constraint(m, cThetaLowerBound[w in 1:number_of_subproblems], vTHETA[w] >= m[:eLowerBoundOperatingCost][w])
    else
        @constraint(m, cThetaLowerBound[w in 1:number_of_subproblems], vTHETA[w] >= 0.0)
    end

    @expression(m, ePlanningCost, objective_function(m))

    @objective(m, Min, m[:ePlanningCost] + sum(m[:vTHETA][w] for w in 1:number_of_subproblems))

end


function process_planning_sol(m::Model,planning_variables::Vector{String})

    capacity_variables = values(m[:eAvailableCapacity])

    if any(value(vcap) < -1e-8 for vcap in capacity_variables)
        @info "Found negative capacity values, setting them to zero."
        planning_variables_values = Dict();
        all_planning_variables = all_variables(m)
        for v in all_planning_variables
            if in(v,capacity_variables)
                planning_variables_values[v] = round_small_values(value(v))
            else
                planning_variables_values[v] = value(v)
            end
        end
        planning_cost = value(x->planning_variables_values[x], m[:ePlanningCost])
        planning_variables_values = Dict([s => planning_variables_values[variable_by_name(m,s)] for s in planning_variables])
    else
        planning_cost =  value(m[:ePlanningCost])
        planning_variables_values = Dict([s=>value.(variable_by_name(m,s)) for s in planning_variables])
    end

    planning_sol =  (planning_cost = planning_cost, values = planning_variables_values)

    return planning_sol
    
end

function round_small_values(z::Float64)
    if z < -1e-8
        return 0.0
    else
        return z
    end
end