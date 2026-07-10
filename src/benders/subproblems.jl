
function add_slacks_to_subproblems!(m_subproblems::Vector{Dict{Any, Any}})
    
    add_slacks_to_local_subproblems!(m_subproblems); 

    return nothing
end

function add_slacks_to_subproblems!(m_subproblems::DArray{Dict{Any, Any}, 1, Vector{Dict{Any, Any}}})

    @sync for p in workers()
        @async @spawnat p begin
            add_slacks_to_local_subproblems!(localpart(m_subproblems));
        end
    end

    return nothing
end

function add_slacks_to_local_subproblems!(subproblem_local::Vector{Dict{Any,Any}})

    for sp in subproblem_local
        add_slacks_to_subproblem!(sp[:model]);
    end
    return nothing
end


function add_slacks_to_subproblem!(subproblem::Model)
    ### Slack variables are added to the subproblems and fixed to zero. 
    ### We will then allow slack variables to be non-zero to generate feasibility cuts when a subproblem is infeasible.

    eq_cons =  all_constraints(subproblem,AffExpr,MOI.EqualTo{Float64})
    less_ineq_cons = all_constraints(subproblem,AffExpr,MOI.LessThan{Float64})
    greater_ineq_cons = all_constraints(subproblem,AffExpr,MOI.GreaterThan{Float64})


    @variable(subproblem, slack_max)
    @constraint(subproblem, slack_max >= 0)

    if !isempty(less_ineq_cons)
        for c in less_ineq_cons
            set_normalized_coefficient(c, slack_max, -1)
        end
    end

    if !isempty(greater_ineq_cons)
        for c in greater_ineq_cons
            set_normalized_coefficient(c, slack_max, 1)
        end
    end

    if !isempty(eq_cons)
        n = length(eq_cons)
        @variable(subproblem, slack_eq[1:n])
        for i in 1:n
            set_normalized_coefficient(eq_cons[i], slack_eq[i], -1)
        end
        @constraint(subproblem, [i in 1:n], slack_eq[i] <= slack_max)
        @constraint(subproblem, [i in 1:n], -slack_eq[i] <= slack_max)
    end

        fix.(slack_max,0.0); 
    

    return nothing
end

function scale_local_subproblem_objectives!(subproblem_local::Vector{Dict{Any,Any}}, obj_scale::Float64)
    for sp in subproblem_local
        set_objective_function(sp[:model], objective_function(sp[:model]) / obj_scale)
    end
    return nothing
end

function scale_subproblem_objectives!(m_subproblems::Vector{Dict{Any, Any}}, obj_scale::Float64)
    scale_local_subproblem_objectives!(m_subproblems, obj_scale)
    return nothing
end

function scale_subproblem_objectives!(m_subproblems::DArray{Dict{Any, Any}, 1, Vector{Dict{Any, Any}}}, obj_scale::Float64)
    @sync for p in workers()
        @async @spawnat p begin
            scale_local_subproblem_objectives!(localpart(m_subproblems), obj_scale)
        end
    end
    return nothing
end

function fix_linking_variables!(m::Model,planning_sol::NamedTuple,linking_variables_sub::Vector{String})
    ### Fix linking variables in the subproblem to the values computed by the planning problem. 
	for y in linking_variables_sub
		vy = variable_by_name(m,y);
		fix(vy,planning_sol.values[y];force=true)
		if is_integer(vy)
			unset_integer(vy)
		elseif is_binary(vy)
			unset_binary(vy)
		end
	end
end

function solve_subproblem(m::Model,planning_sol::NamedTuple,linking_variables_sub::Vector{String},expect_feasible_subproblems::Bool)

    ### Solve the operational subproblem. If it is infeasible, compute feasibility cuts.

	fix_linking_variables!(m,planning_sol,linking_variables_sub)

    # Enable Farkas dual extraction on infeasible solves
    try; set_attribute(m, "InfUnbdInfo", 1); catch; end

	optimize!(m)

	if has_values(m)
		op_cost = objective_value(m);
		lambda = [dual(FixRef(variable_by_name(m,y))) for y in linking_variables_sub];
		theta_coeff = 1;
		@info "Subproblem feasible: op_cost=$(round(op_cost, sigdigits=4)), status=$(termination_status(m)), lambda_norm=$(round(norm(lambda), sigdigits=4)), lambda_max=$(round(maximum(abs.(lambda)), sigdigits=4))"
    elseif expect_feasible_subproblems==true
        compute_conflict!(m)
            list_of_conflicting_constraints = ConstraintRef[];
            for (F, S) in list_of_constraint_types(m)
                for con in all_constraints(m, F, S)
                    if get_attribute(con, MOI.ConstraintConflictStatus()) == MOI.IN_CONFLICT
                        push!(list_of_conflicting_constraints, con)
                    end
                end
            end
        display(list_of_conflicting_constraints)
        error("The subproblem is infeasible, but ExpectFeasibleSubproblems = true. Benders likely did not converge before MaxIter was reached. Check conflicting constraints above.")
    else
        @info "Subproblem is infeasible (status=$(termination_status(m)), primal=$(primal_status(m)), dual=$(dual_status(m))), attempting Farkas dual feasibility cut..."

        # Attempt Farkas dual approach: extract dual ray directly without re-solving.
        # op_cost must be the full Farkas objective pi^T*b + lambda^T*x_bar (> 0 by certificate).
        # Using only lambda^T*x_bar (the old formula) makes the cut 0 >= lambda^T*x — a hyperplane
        # through the origin that trivially passes all x >= 0 and builds zero capacity pressure.
        use_farkas = dual_status(m) == MOI.INFEASIBILITY_CERTIFICATE
        if use_farkas
            lambda = [dual(FixRef(variable_by_name(m,y))) for y in linking_variables_sub];
            # Build a set of linking variable refs for O(1) lookup
            linking_var_set = Set(variable_by_name(m, y) for y in linking_variables_sub)
            physical_farkas = 0.0
            for (F, S) in list_of_constraint_types(m)
                for con in all_constraints(m, F, S)
                    if F <: JuMP.AbstractVariableRef
                        # Variable bound/fix constraint: must include non-linking local variables
                        # (e.g. coal_gen >= 150, import_flow <= 500) in pi^T*b, but skip linking
                        # variable fix constraints since their contribution is already in linking_farkas.
                        v = jump_function(constraint_object(con))
                        v in linking_var_set && continue
                        rhs = 0.0
                        if S <: MOI.GreaterThan
                            rhs = constraint_object(con).set.lower
                        elseif S <: MOI.LessThan
                            rhs = constraint_object(con).set.upper
                        elseif S <: MOI.EqualTo
                            rhs = constraint_object(con).set.value
                        else
                            continue  # Integer/ZeroOne: no meaningful dual in LP relaxation
                        end
                        physical_farkas += dual(con) * rhs
                    elseif F <: JuMP.AbstractJuMPScalar
                        # Scalar affine constraint: normalized_rhs gives the RHS directly
                        physical_farkas += dual(con) * normalized_rhs(con)
                    elseif F <: AbstractVector
                        # Vector constraint (e.g. emission caps across zones):
                        # dual() returns a Vector{Float64}; RHS equivalent is -moi_f.constants
                        d = dual(con)
                        moi_f = MOI.get(backend(m), MOI.ConstraintFunction(), index(con))
                        physical_farkas += dot(d, -moi_f.constants)
                    end
                end
            end
            linking_farkas = sum(lambda[i] * planning_sol.values[linking_variables_sub[i]] for i in 1:length(linking_variables_sub))
            op_cost = physical_farkas + linking_farkas
            theta_coeff = 0;
            if op_cost > 0
                @info "Farkas cut (dual ray): op_cost=$(round(op_cost, sigdigits=4)) [physical=$(round(physical_farkas, sigdigits=4)), linking=$(round(linking_farkas, sigdigits=4))], lambda_norm=$(round(norm(lambda), sigdigits=4)), lambda_max=$(round(maximum(abs.(lambda)), sigdigits=4)), n_nonzero=$(sum(abs.(lambda) .> 1e-8))/$(length(lambda))"
            else
                @warn "Farkas objective = $(round(op_cost, sigdigits=4)) ≤ 0 — Farkas certificate may be degenerate or solver returned an invalid ray. Falling back to slack approach."
                use_farkas = false
            end
        end
        if !use_farkas
            # Farkas duals unavailable or invalid — fall back to slack-based feasibility subproblem
            if dual_status(m) == MOI.INFEASIBILITY_CERTIFICATE
                @warn "Falling back to slack feasibility subproblem (Farkas objective was ≤ 0)..."
            else
                @warn "Farkas duals unavailable (dual_status=$(dual_status(m))), falling back to slack feasibility subproblem..."
            end
            #### Feasibility cuts generation based on https://link.springer.com/chapter/10.1007/978-3-030-45771-6_7

            unfix.(m[:slack_max]);
            objfun = objective_function(m);
            @objective(m, Min, m[:slack_max])

            try; set_attribute(m, "Crossover", 0); catch; end
            optimize!(m)
            if !has_values(m)
                try; set_attribute(m, "Crossover", 1); catch; end
                optimize!(m)
            end
            if !has_values(m)
                @warn "Feasibility subproblem has no solution after barrier retries. Retrying with NumericFocus=3 and simplex."
                try; set_attribute(m, "NumericFocus", 3); catch; end
                try; set_attribute(m, "Method", 1); catch; end
                optimize!(m)
            end
            if !has_values(m)
                compute_conflict!(m)
                list_of_conflicting_constraints = ConstraintRef[];
                for (F, S) in list_of_constraint_types(m)
                    for con in all_constraints(m, F, S)
                        if get_attribute(con, MOI.ConstraintConflictStatus()) == MOI.IN_CONFLICT
                            push!(list_of_conflicting_constraints, con)
                        end
                    end
                end
                display(list_of_conflicting_constraints)
                error("Feasibility subproblem is infeasible even with slack variables and NumericFocus=3. Check conflicting constraints above.")
            end
            op_cost = objective_value(m);
            lambda = [dual(FixRef(variable_by_name(m,y))) for y in linking_variables_sub];
            theta_coeff = 0;
            @info "Slack feasibility cut: op_cost=$(round(op_cost, sigdigits=4)), lambda_norm=$(round(norm(lambda), sigdigits=4)), lambda_max=$(round(maximum(abs.(lambda)), sigdigits=4)), n_nonzero=$(sum(abs.(lambda) .> 1e-8))/$(length(lambda))"

            try; set_attribute(m, "Crossover", 1); catch; end
            fix.(m[:slack_max], 0.0);
            @objective(m, Min, objfun)
        end
	end

	return (op_cost=op_cost,lambda = lambda,theta_coeff=theta_coeff)

end


function solve_local_subproblems(subproblem_local::Vector{Dict{Any,Any}},planning_sol::NamedTuple, expect_feasible_subproblems::Bool)

    local_sol=Dict();
    for sp in subproblem_local
        m = sp[:model];
        linking_variables_sub = sp[:linking_variables_sub]
        w = sp[:subproblem_index];
		local_sol[w] = solve_subproblem(m,planning_sol,linking_variables_sub,expect_feasible_subproblems);
    end
    return local_sol
end

"""
    solve_subproblems(
        m_subproblems::DArray{Dict{Any, Any}, 1, Vector{Dict{Any, Any}}}, 
        planning_sol::NamedTuple
    )

Solves subproblems in parallel using distributed computing capabilities.

This function coordinates the parallel solution of operational subproblems across multiple workers,
using Julia's distributed computing framework. Each worker processes its local portion of the
distributed array of subproblems.

# Arguments
- `m_subproblems::DArray`: Distributed array containing the subproblems, where each element is a
   dictionary representing a subproblem
- `planning_sol::NamedTuple`: Current solution of the planning problem containing variable values
   needed for the subproblem solutions

# Returns
A merged dictionary containing results from all subproblems, where each entry contains:
- Optimal objective value
- Dual variables
- Other solution information from each subproblem

# Implementation Details
Uses `@sync` and `@async` for coordinated parallel execution, with results fetched from each worker
and merged into a single dictionary containing all subproblem solutions.
"""
function solve_subproblems(m_subproblems::DArray{Dict{Any, Any}, 1, Vector{Dict{Any, Any}}},planning_sol::NamedTuple,expect_feasible_subproblems::Bool)

    p_id = workers();
    np_id = length(p_id);

    sub_results = [Dict() for k in 1:np_id];

    @sync for k in 1:np_id
              @async sub_results[k]= @fetchfrom p_id[k] solve_local_subproblems(localpart(m_subproblems),planning_sol,expect_feasible_subproblems); ### This is equivalent to fetch(@spawnat p .....)
    end

	sub_results = merge(sub_results...);

    return sub_results
end


function solve_subproblems(m_subproblems::Vector{Dict{Any, Any}},planning_sol::NamedTuple,expect_feasible_subproblems::Bool)
    
    sub_results = solve_local_subproblems(m_subproblems,planning_sol,expect_feasible_subproblems); 

    return sub_results
end
