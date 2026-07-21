
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

    # Preserve the original structural rows before adding the Phase-I
    # epigraph constraints.  An opt-in audit can then attribute the relaxed
    # infeasibility to exact model rows instead of reporting the much less
    # informative IIS produced by the strict final solve.
    subproblem[:phase1_original_constraints] = Any[
        eq_cons...
        less_ineq_cons...
        greater_ineq_cons...
    ]


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

    # Add Big-M penalty to objective so elastic slack mode has a proper cost signal.
    # When slack_max is fixed to 0 (normal solves), this term = 0 and has no effect.
    # When unfixed (elastic mode), it makes unserved demand very expensive, driving
    # the master to build capacity. Use 100x max existing objective coefficient,
    # floored at 1.0 and capped at 1e8 to avoid numerical blow-up.
    objfun = objective_function(subproblem)
    abs_coeffs = filter(c -> c > 0.0, [abs(coefficient(objfun, v)) for v in all_variables(subproblem)])
    raw_bigm = isempty(abs_coeffs) ? 1.0 : 100.0 * maximum(abs_coeffs)
    big_m = clamp(raw_bigm, 1.0, 1e8)
    @info "ElasticSlack Big-M penalty: $(big_m) (raw=$(round(raw_bigm, sigdigits=3)), capped=$(raw_bigm != big_m))"
    set_objective_function(subproblem, objfun + big_m * slack_max)

    fix.(slack_max,0.0);

    return nothing
end

_is_phase1_slack_variable(variable::VariableRef) = begin
    variable_name = name(variable)
    variable_name == "slack_max" || startswith(variable_name, "slack_eq[")
end

function _phase1_hard_activity(
    func::AffExpr,
    variable_value::Function,
)
    return value(
        variable -> _is_phase1_slack_variable(variable) ? 0.0 : variable_value(variable),
        func,
    )
end

function _phase1_hard_violation(activity::Real, set::MOI.LessThan)
    return max(0.0, Float64(activity) - set.upper)
end


function _phase1_hard_violation(activity::Real, set::MOI.GreaterThan)
    return max(0.0, set.lower - Float64(activity))
end


function _phase1_hard_violation(activity::Real, set::MOI.EqualTo)
    return abs(Float64(activity) - set.value)
end


_phase1_set_description(set::MOI.LessThan) = "LessThan($(set.upper))"
_phase1_set_description(set::MOI.GreaterThan) = "GreaterThan($(set.lower))"
_phase1_set_description(set::MOI.EqualTo) = "EqualTo($(set.value))"

function _audit_phase1_constraint_attribution!(
    m::Model,
    subproblem_index,
    phase1_objective::Real,
)
    get(ENV, "BENDERS_PHASE1_CONSTRAINT_AUDIT", "false") == "true" || return

    top_n = tryparse(
        Int,
        get(ENV, "BENDERS_PHASE1_CONSTRAINT_AUDIT_TOP", "25"),
    )
    isnothing(top_n) && error(
        "BENDERS_PHASE1_CONSTRAINT_AUDIT_TOP must be an integer",
    )
    top_n > 0 || error(
        "BENDERS_PHASE1_CONSTRAINT_AUDIT_TOP must be positive",
    )

    constraints = try
        m[:phase1_original_constraints]
    catch
        error(
            "BENDERS_PHASE1_CONSTRAINT_AUDIT requires the original " *
            "constraint registry created by add_slacks_to_subproblem!",
        )
    end

    rows = NamedTuple[]
    for (constraint_index, constraint) in enumerate(constraints)
        constraint_data = constraint_object(constraint)
        activity = _phase1_hard_activity(
            constraint_data.func,
            variable -> value(variable),
        )
        hard_violation = _phase1_hard_violation(
            activity,
            constraint_data.set,
        )
        dual_value = dual(constraint)
        constraint_name = name(constraint)
        isempty(constraint_name) &&
            (constraint_name = "<anonymous_original_constraint_$(constraint_index)>")
        push!(rows, (
            index=constraint_index,
            name=constraint_name,
            set=_phase1_set_description(constraint_data.set),
            activity=Float64(activity),
            hard_violation=hard_violation,
            dual=Float64(dual_value),
            abs_dual=abs(Float64(dual_value)),
        ))
    end

    nonzero_dual_rows = filter(row -> row.abs_dual > 1e-8, rows)
    violated_rows = filter(row -> row.hard_violation > 1e-8, rows)
    total_abs_dual = sum(row.abs_dual for row in nonzero_dual_rows)
    max_hard_violation = isempty(violated_rows) ? 0.0 : maximum(
        row.hard_violation for row in violated_rows
    )
    @info "PHASE1_CONSTRAINT_AUDIT_SUMMARY: w=$(subproblem_index) phase1_objective=$(phase1_objective) original_constraints=$(length(rows)) nonzero_duals=$(length(nonzero_dual_rows)) hard_violations=$(length(violated_rows)) max_hard_violation=$(max_hard_violation) total_abs_dual=$(total_abs_dual)"

    by_dual = sort(rows; by=row -> (-row.abs_dual, -row.hard_violation))
    for (rank, row) in enumerate(first(by_dual, min(top_n, length(by_dual))))
        dual_share = total_abs_dual > 0.0 ? row.abs_dual / total_abs_dual : 0.0
        @info "PHASE1_CONSTRAINT_DUAL: w=$(subproblem_index) rank=$(rank) name=$(repr(row.name)) set=$(row.set) activity=$(row.activity) hard_violation=$(row.hard_violation) dual=$(row.dual) abs_dual_share=$(dual_share)"
    end

    by_violation = sort(rows; by=row -> (-row.hard_violation, -row.abs_dual))
    for (rank, row) in enumerate(first(by_violation, min(top_n, length(by_violation))))
        @info "PHASE1_CONSTRAINT_VIOLATION: w=$(subproblem_index) rank=$(rank) name=$(repr(row.name)) set=$(row.set) activity=$(row.activity) hard_violation=$(row.hard_violation) dual=$(row.dual)"
    end

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

const PHASE1_ORACLE_AUDIT_DONE = Ref(false)

function optimizer_attribute_or_missing(m::Model, attribute::String)
    try
        return get_attribute(m, attribute)
    catch
        return missing
    end
end

function audit_phase1_at_oracle!(
    m::Model,
    planning_sol::NamedTuple,
    linking_variables_sub::Vector{String},
    op_cost::Real,
    lambda::AbstractVector{<:Real},
    subproblem_index,
)
    get(ENV, "BENDERS_PHASE1_ORACLE_AUDIT", "false") == "true" || return
    PHASE1_ORACLE_AUDIT_DONE[] && return

    if nprocs() > 1
        @warn "PHASE1_ORACLE_AUDIT_SKIPPED: diagnostic must run serially (nprocs=$(nprocs())). Set Distributed=false."
        return
    end

    min_objective = tryparse(Float64, get(ENV, "BENDERS_PHASE1_ORACLE_AUDIT_MIN_OBJECTIVE", "0"))
    isnothing(min_objective) && error("BENDERS_PHASE1_ORACLE_AUDIT_MIN_OBJECTIVE must be numeric")
    op_cost >= min_objective || return

    oracle_path = get(ENV, "BENDERS_MONO_LINKING_VARS", "")
    isempty(oracle_path) && error("BENDERS_PHASE1_ORACLE_AUDIT requires BENDERS_MONO_LINKING_VARS")
    isfile(oracle_path) || error("BENDERS_MONO_LINKING_VARS does not exist: $oracle_path")

    requested = Set(linking_variables_sub)
    oracle_values = Dict{String,Float64}()
    conflicting = Set{String}()
    open(oracle_path) do io
        eof(io) || readline(io) # header
        for line in eachline(io)
            idx = findlast(',', line)
            isnothing(idx) && continue
            variable_name = strip(line[1:idx-1])
            variable_name in requested || continue
            parsed_value = tryparse(Float64, strip(line[idx+1:end]))
            isnothing(parsed_value) && continue
            if haskey(oracle_values, variable_name) && !isapprox(
                oracle_values[variable_name], parsed_value; rtol=1e-8, atol=1e-8
            )
                push!(conflicting, variable_name)
            else
                oracle_values[variable_name] = parsed_value
            end
        end
    end

    missing = filter(v -> !haskey(oracle_values, v), linking_variables_sub)
    if !isempty(missing) || !isempty(conflicting)
        missing_preview = join(first(missing, min(5, length(missing))), ", ")
        conflict_preview = join(first(collect(conflicting), min(5, length(conflicting))), ", ")
        @error "PHASE1_ORACLE_AUDIT_INDETERMINATE: w=$(subproblem_index) missing=$(length(missing)) conflicting=$(length(conflicting)) missing_preview=[$missing_preview] conflict_preview=[$conflict_preview]"
        return
    end

    variables = VariableRef[]
    for variable_name in linking_variables_sub
        variable = variable_by_name(m, variable_name)
        isnothing(variable) && error("PHASE1_ORACLE_AUDIT: linking variable $variable_name is absent from the subproblem")
        push!(variables, variable)
    end
    generating_values = [fix_value(variable) for variable in variables]
    planning_values = [planning_sol.values[name] for name in linking_variables_sub]
    oracle_vector = [oracle_values[name] for name in linking_variables_sub]
    max_generating_fix_difference = isempty(variables) ? 0.0 : maximum(abs.(generating_values .- planning_values))
    barrier_cut_residual_oracle = op_cost + dot(lambda, oracle_vector .- generating_values)
    barrier_cut_scale_oracle = max(
        1.0,
        abs(op_cost) + sum(abs(lambda[i] * (oracle_vector[i] - generating_values[i])) for i in eachindex(lambda)),
    )

    # Snapshot the barrier/no-crossover Phase-I result before any diagnostic
    # re-solve changes the model's result state.
    barrier_term = termination_status(m)
    barrier_primal = primal_status(m)
    barrier_dual_status = dual_status(m)
    barrier_raw = raw_status(m)
    barrier_result_count = result_count(m)
    barrier_dual_objective = try
        dual_objective_value(m)
    catch
        NaN
    end
    barrier_constr_vio = optimizer_attribute_or_missing(m, "ConstrVio")
    barrier_bound_vio = optimizer_attribute_or_missing(m, "BoundVio")
    barrier_dual_vio = optimizer_attribute_or_missing(m, "DualVio")
    barrier_compl_vio = optimizer_attribute_or_missing(m, "ComplVio")
    original_method = optimizer_attribute_or_missing(m, "Method")
    original_crossover = optimizer_attribute_or_missing(m, "Crossover")

    # Mark before optimizing so a failed audit cannot repeat indefinitely.
    PHASE1_ORACLE_AUDIT_DONE[] = true
    try
        # Re-solve the identical generating-point Phase-I model with dual
        # simplex.  This produces a directly comparable set of fixing duals
        # without changing the cut returned by solve_subproblem.
        set_attribute(m, "Method", 1)
        optimize!(m)
        simplex_term = termination_status(m)
        simplex_primal = primal_status(m)
        simplex_dual_status = dual_status(m)
        simplex_raw = raw_status(m)
        simplex_result_count = result_count(m)
        simplex_values_available = has_values(m)
        simplex_duals_available = has_duals(m)
        simplex_objective = simplex_values_available ? objective_value(m) : NaN
        simplex_dual_objective = simplex_duals_available ? dual_objective_value(m) : NaN
        simplex_lambda = simplex_duals_available ? [dual(FixRef(variable)) for variable in variables] : fill(NaN, length(variables))
        simplex_cut_residual_oracle = simplex_duals_available ? simplex_objective + dot(simplex_lambda, oracle_vector .- generating_values) : NaN
        simplex_cut_scale_oracle = simplex_duals_available ? max(
            1.0,
            abs(simplex_objective) + sum(abs(simplex_lambda[i] * (oracle_vector[i] - generating_values[i])) for i in eachindex(simplex_lambda)),
        ) : NaN
        lambda_max_difference = simplex_duals_available && !isempty(lambda) ? maximum(abs.(simplex_lambda .- lambda)) : NaN
        simplex_constr_vio = optimizer_attribute_or_missing(m, "ConstrVio")
        simplex_bound_vio = optimizer_attribute_or_missing(m, "BoundVio")
        simplex_dual_vio = optimizer_attribute_or_missing(m, "DualVio")
        simplex_compl_vio = optimizer_attribute_or_missing(m, "ComplVio")

        @info "PHASE1_DUAL_METHOD_BARRIER: w=$(subproblem_index) termination_status=$(barrier_term) primal_status=$(barrier_primal) dual_status=$(barrier_dual_status) raw_status=$(repr(barrier_raw)) result_count=$(barrier_result_count) primal_objective=$(op_cost) dual_objective=$(barrier_dual_objective) constr_vio=$(barrier_constr_vio) bound_vio=$(barrier_bound_vio) dual_vio=$(barrier_dual_vio) compl_vio=$(barrier_compl_vio) cut_residual_oracle=$(barrier_cut_residual_oracle) normalized_cut_residual=$(barrier_cut_residual_oracle / barrier_cut_scale_oracle)"
        @info "PHASE1_DUAL_METHOD_SIMPLEX: w=$(subproblem_index) termination_status=$(simplex_term) primal_status=$(simplex_primal) dual_status=$(simplex_dual_status) raw_status=$(repr(simplex_raw)) result_count=$(simplex_result_count) primal_objective=$(simplex_objective) dual_objective=$(simplex_dual_objective) constr_vio=$(simplex_constr_vio) bound_vio=$(simplex_bound_vio) dual_vio=$(simplex_dual_vio) compl_vio=$(simplex_compl_vio) cut_residual_oracle=$(simplex_cut_residual_oracle) normalized_cut_residual=$(simplex_cut_residual_oracle / simplex_cut_scale_oracle) lambda_max_difference=$(lambda_max_difference)"

        for i in eachindex(variables)
            fix(variables[i], oracle_vector[i]; force=true)
        end
        maximum_abs_oracle_fixed_difference = isempty(variables) ? 0.0 : maximum(
            abs(fix_value(variables[i]) - oracle_vector[i]) for i in eachindex(variables)
        )

        optimize!(m)
        term = termination_status(m)
        primal = primal_status(m)
        dual_stat = dual_status(m)
        raw = raw_status(m)
        results = result_count(m)
        values_available = has_values(m)
        slack_at_oracle = values_available ? value(m[:slack_max]) : NaN
        objective_at_oracle = values_available ? objective_value(m) : NaN
        oracle_constr_vio = optimizer_attribute_or_missing(m, "ConstrVio")
        oracle_bound_vio = optimizer_attribute_or_missing(m, "BoundVio")
        oracle_dual_vio = optimizer_attribute_or_missing(m, "DualVio")
        oracle_compl_vio = optimizer_attribute_or_missing(m, "ComplVio")
        barrier_supporting_violation = values_available ? barrier_cut_residual_oracle - objective_at_oracle : NaN
        simplex_supporting_violation = values_available ? simplex_cut_residual_oracle - objective_at_oracle : NaN
        barrier_normalized_supporting_violation = values_available ? barrier_supporting_violation / max(1.0, abs(barrier_cut_residual_oracle), abs(objective_at_oracle)) : NaN
        simplex_normalized_supporting_violation = values_available ? simplex_supporting_violation / max(1.0, abs(simplex_cut_residual_oracle), abs(objective_at_oracle)) : NaN

        @info "PHASE1_ORACLE_AUDIT: w=$(subproblem_index) termination_status=$(term) primal_status=$(primal) dual_status=$(dual_stat) raw_status=$(repr(raw)) result_count=$(results) slack_max=$(slack_at_oracle) objective=$(objective_at_oracle) constr_vio=$(oracle_constr_vio) bound_vio=$(oracle_bound_vio) dual_vio=$(oracle_dual_vio) compl_vio=$(oracle_compl_vio) maximum_abs_master_fixed_difference=$(max_generating_fix_difference) maximum_abs_oracle_fixed_difference=$(maximum_abs_oracle_fixed_difference) barrier_cut_residual=$(barrier_cut_residual_oracle) barrier_supporting_violation=$(barrier_supporting_violation) barrier_normalized_supporting_violation=$(barrier_normalized_supporting_violation) simplex_cut_residual=$(simplex_cut_residual_oracle) simplex_supporting_violation=$(simplex_supporting_violation) simplex_normalized_supporting_violation=$(simplex_normalized_supporting_violation)"
        if !values_available
            @warn "PHASE1_ORACLE_AUDIT_BARRIER_INDETERMINATE: oracle Phase-I solve has no primal result."
        elseif barrier_normalized_supporting_violation > 1e-8
            @error "PHASE1_ORACLE_AUDIT_BARRIER_INVALID: barrier-derived cut exceeds the actual Phase-I value at the oracle point."
        else
            @info "PHASE1_ORACLE_AUDIT_BARRIER_VALID: barrier-derived cut satisfies the Phase-I supporting inequality at the oracle point."
        end
        if !values_available || !simplex_duals_available
            @warn "PHASE1_ORACLE_AUDIT_SIMPLEX_INDETERMINATE: simplex generating solve lacks duals or oracle Phase-I solve lacks a primal result."
        elseif simplex_normalized_supporting_violation > 1e-8
            @error "PHASE1_ORACLE_AUDIT_SIMPLEX_INVALID: dual-simplex-derived cut exceeds the actual Phase-I value at the oracle point."
        else
            @info "PHASE1_ORACLE_AUDIT_SIMPLEX_VALID: dual-simplex-derived cut satisfies the Phase-I supporting inequality at the oracle point."
        end
    catch err
        @error "PHASE1_ORACLE_AUDIT_ERROR: w=$(subproblem_index) error=$(sprint(showerror, err))"
    finally
        for i in eachindex(variables)
            fix(variables[i], generating_values[i]; force=true)
        end
        !ismissing(original_method) && set_attribute(m, "Method", original_method)
        !ismissing(original_crossover) && set_attribute(m, "Crossover", original_crossover)
        maximum_abs_restore_difference = isempty(variables) ? 0.0 : maximum(
            abs(fix_value(variables[i]) - generating_values[i]) for i in eachindex(variables)
        )
        @info "PHASE1_ORACLE_AUDIT_RESTORE: w=$(subproblem_index) maximum_abs_restore_difference=$(maximum_abs_restore_difference) restored_method=$(original_method) restored_crossover=$(original_crossover)"
    end
end

function solve_subproblem(m::Model,planning_sol::NamedTuple,linking_variables_sub::Vector{String},expect_feasible_subproblems::Bool,elastic_slack::Bool=false,subproblem_index=nothing)

    ### Solve the operational subproblem. If it is infeasible, compute feasibility cuts.

	fix_linking_variables!(m,planning_sol,linking_variables_sub)

    if elastic_slack
        # Always unfix slack before solving: subproblem is always feasible via slack absorption.
        # Slack penalty (baked into eVariableCost at model construction) drives master to invest.
        unfix.(m[:slack_max])
    else
        # Enable Farkas dual extraction on infeasible solves
        try; set_attribute(m, "InfUnbdInfo", 1); catch; end
    end

	optimize!(m)

	if has_values(m)
		op_cost = objective_value(m);
		lambda = [dual(FixRef(variable_by_name(m,y))) for y in linking_variables_sub];
		theta_coeff = 1;
        cut_source = elastic_slack ? :elastic_optimality : :optimality
        lmax = isempty(lambda) ? 0.0 : maximum(abs.(lambda))
        if elastic_slack
            slack_val = value(m[:slack_max])
            if slack_val > 1e-6
                @info "Subproblem elastic (slack=$(round(slack_val, sigdigits=4))): op_cost=$(round(op_cost, sigdigits=4)), lambda_norm=$(round(norm(lambda), sigdigits=4)), lambda_max=$(round(lmax, sigdigits=4))"
            else
                @info "Subproblem feasible (slack=0): op_cost=$(round(op_cost, sigdigits=4)), lambda_norm=$(round(norm(lambda), sigdigits=4)), lambda_max=$(round(lmax, sigdigits=4))"
            end
            fix.(m[:slack_max], 0.0)  # re-fix for next iteration
        end
    elseif elastic_slack
        # ElasticSlack guarantees feasibility — !has_values here means the model is
        # unbounded or numerically broken. Re-fix slack and surface a clear error.
        @error "Subproblem infeasible/unbounded even with ElasticSlack active. status=$(termination_status(m)), primal=$(primal_status(m)), dual=$(dual_status(m)). Check model construction (e.g. unbounded variables, conflicting variable bounds)."
        fix.(m[:slack_max], 0.0; force=true)
        error("ElasticSlack subproblem failed (status=$(termination_status(m))). See @error above.")
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

        feasibility_cut_mode = _feasibility_cut_mode()
        @info "FEASIBILITY_CUT_MODE: w=$(subproblem_index) mode=$(feasibility_cut_mode)"

        # Attempt Farkas dual approach: extract dual ray directly without re-solving.
        # op_cost must be the full Farkas objective pi^T*b + lambda^T*x_bar (> 0 by certificate).
        # Using only lambda^T*x_bar (the old formula) makes the cut 0 >= lambda^T*x — a hyperplane
        # through the origin that trivially passes all x >= 0 and builds zero capacity pressure.
        farkas_available = dual_status(m) == MOI.INFEASIBILITY_CERTIFICATE
        feasibility_cut_mode == :farkas && !farkas_available && error(
            "BENDERS_FEASIBILITY_CUT_MODE=farkas requested, but no Farkas certificate " *
            "is available for subproblem $(subproblem_index) (dual_status=$(dual_status(m)))",
        )
        use_farkas = feasibility_cut_mode != :phase1 && farkas_available
        if use_farkas
            lambda = [dual(FixRef(variable_by_name(m,y))) for y in linking_variables_sub];
            # Use Gurobi's certified dual objective directly instead of manually
            # reconstructing pi'b from constraint duals.  The manual loop missed up to 97%
            # of the certificate (variable-bound and bridged-constraint terms not reachable
            # via list_of_constraint_types).  dual_objective_value(m) = pi'b for all
            # constraints including bounds — the complete Farkas proof.
            cert = dual_objective_value(m)
            # Normalize so max|lambda| <= 1.  Near-zero Budget linking variables produce
            # extreme dual multipliers (lambda_max ~ 1e5) that create ill-conditioned master
            # rows.  Dividing by a positive scalar preserves the halfspace and scales cert
            # proportionally so the separation margin cert/s stays >> FeasibilityTol.
            lambda_scale = isempty(lambda) ? 1.0 : maximum(abs.(lambda))
            if lambda_scale > 1.0
                lambda = lambda ./ lambda_scale
                cert   = cert   / lambda_scale
            end
            # Diagnostic: set BENDERS_FARKAS_DEBUG=true to validate cut pipeline.
            # cert = dual_objective_value(m)/lambda_scale is the complete Farkas proof.
            # separation_margin = cert = cut value at x_bar (must be >> FeasibilityTol=1e-6).
            if get(ENV, "BENDERS_FARKAS_DEBUG", "false") == "true"
                x_fixed    = [fix_value(variable_by_name(m, y)) for y in linking_variables_sub]
                x_planning = [planning_sol.values[y] for y in linking_variables_sub]
                max_diff   = isempty(x_fixed) ? 0.0 : maximum(abs.(x_fixed .- x_planning))
                linking_farkas_diag = isempty(x_fixed) ? 0.0 : dot(lambda, x_fixed)
                physical_farkas_diag = cert - linking_farkas_diag
                @info "FARKAS_DIAG: dual_status=$(dual_status(m)) cert_raw=$(round(cert*lambda_scale,sigdigits=4)) lambda_scale=$(round(lambda_scale,sigdigits=4)) cert_norm=$(round(cert,sigdigits=4)) physical_norm=$(round(physical_farkas_diag,sigdigits=4)) separation_margin=$(round(cert,sigdigits=4)) max_fix_vs_planning_diff=$(round(max_diff,sigdigits=4))"
                if cert < 1e-6
                    @warn "FARKAS_DIAG: separation_margin=$(round(cert,sigdigits=4)) < FeasibilityTol=1e-6 — normalized cut may not separate x_bar from master"
                end
                for i in eachindex(linking_variables_sub)
                    diff = abs(x_fixed[i] - x_planning[i])
                    if diff > 1e-6
                        @warn "FARKAS_DIAG MISMATCH: $(linking_variables_sub[i]) fixed=$(round(x_fixed[i],sigdigits=4)) planning=$(round(x_planning[i],sigdigits=4)) Δ=$(round(diff,sigdigits=4)) λ=$(round(lambda[i],sigdigits=4))"
                    end
                end
                if cert > 1e-4
                    @info "FARKAS_DIAG VERDICT: PIPELINE OK — separation_margin=$(round(cert,sigdigits=4)) >> FeasibilityTol."
                else
                    @warn "FARKAS_DIAG VERDICT: WEAK SEPARATION — separation_margin=$(round(cert,sigdigits=4)) ≤ 1e-4; cut may be ineffective at separating x_bar."
                end
                # Cut validity test at x* (monolithic solution).
                # Set BENDERS_MONO_RESULTS_DIR to a results/ directory containing capacity.csv.
                # Optionally set BENDERS_MONO_LINKING_VARS to linking_vars_mono.csv (from
                # compute_mono_linking_vars.jl) to cover non-capacity linking variables.
                # A valid Farkas cut must satisfy: physical_farkas + lambda^T * x* <= 0.
                mono_dir = get(ENV, "BENDERS_MONO_RESULTS_DIR", "")
                if !isempty(mono_dir) && cert > 0
                    cap_path = joinpath(mono_dir, "capacity.csv")
                    if isfile(cap_path)
                        mono_vals = Dict{String,Float64}()
                        # Load capacity variables from capacity.csv
                        lines = readlines(cap_path)
                        if length(lines) > 1
                            header = split(lines[1], ",")
                            cid_col = findfirst(==("component_id"), header)
                            cap_col = findfirst(==("capacity"), header)
                            if !isnothing(cid_col) && !isnothing(cap_col)
                                for line in lines[2:end]
                                    parts = split(line, ",")
                                    length(parts) < max(cid_col, cap_col) && continue
                                    cid = strip(parts[cid_col])
                                    cap_val = tryparse(Float64, strip(parts[cap_col]))
                                    (isnothing(cap_val) || isempty(cid)) && continue
                                    mono_vals["vCAP_$(cid)_period1"] = cap_val
                                end
                            end
                        end
                        # Load non-capacity linking variables (Budget, etc.) from linking_vars_mono.csv
                        # Generated by compute_mono_linking_vars.jl in the case directory.
                        lv_path = get(ENV, "BENDERS_MONO_LINKING_VARS", "")
                        n_extra = 0
                        if !isempty(lv_path) && isfile(lv_path)
                            lv_lines = readlines(lv_path)
                            for line in lv_lines[2:end]
                                # JuMP array-variable names may contain commas;
                                # dump_monolithic_variables.jl writes the value
                                # after the final comma.
                                idx = findlast(',', line)
                                isnothing(idx) && continue
                                vname = strip(line[1:idx-1])
                                val = tryparse(Float64, strip(line[idx+1:end]))
                                if !isnothing(val) && !isempty(vname)
                                    mono_vals[vname] = val
                                    n_extra += 1
                                end
                            end
                            @info "FARKAS_DIAG CUT@MONO: loaded $(n_extra) non-capacity linking vars from $(basename(lv_path))"
                        end
                        x_mono = [get(mono_vals, v, NaN) for v in linking_variables_sub]
                        n_missing = sum(isnan, x_mono)
                        x_mono_clean = [isnan(v) ? 0.0 : v for v in x_mono]
                        cut_lhs_mono = physical_farkas_diag + dot(lambda, x_mono_clean)
                        complete = n_missing == 0 ? "COMPLETE" : "PARTIAL($(n_missing) missing→0)"
                        @info "FARKAS_DIAG CUT@MONO: cut_lhs=$(round(cut_lhs_mono,sigdigits=4)) [$(complete)] (cut valid if ≤0)"
                        if cut_lhs_mono > 1e-4
                            @error "FARKAS_DIAG CUT@MONO INVALID: cut_lhs=$(round(cut_lhs_mono,sigdigits=4)) > 0 at monolithic x* — cut incorrectly excludes the feasible solution; check coefficient signs or index mapping"
                        elseif cut_lhs_mono > 0
                            @warn "FARKAS_DIAG CUT@MONO MARGINAL: cut_lhs=$(round(cut_lhs_mono,sigdigits=6)) barely > 0 — possible unit mismatch or numerical noise"
                        else
                            @info "FARKAS_DIAG CUT@MONO VALID: cut does not exclude monolithic x*"
                        end
                        # Report nonzero-lambda variables: found vs missing
                        for i in eachindex(linking_variables_sub)
                            if abs(lambda[i]) > 1e-8
                                found = !isnan(x_mono[i])
                                tag = found ? "found x*=$(round(x_mono[i],sigdigits=4))" : "MISSING (treated as 0)"
                                @info "FARKAS_DIAG CUT@MONO nonzero_lambda: $(linking_variables_sub[i]) λ=$(round(lambda[i],sigdigits=4)) $(tag) contrib=$(round(lambda[i]*(found ? x_mono[i] : 0.0),sigdigits=4))"
                            end
                        end
                        if n_missing > 0
                            @warn "FARKAS_DIAG CUT@MONO: $(n_missing) linking variables still missing — run compute_mono_linking_vars.jl and set BENDERS_MONO_LINKING_VARS"
                        end
                    else
                        @warn "FARKAS_DIAG CUT@MONO: capacity.csv not found at $(cap_path)"
                    end
                end
            end
            # Compute linking_farkas using fix_value (the actual fixed value Gurobi saw) to
            # detect any mismatch vs planning_sol.values, which would corrupt the alpha term.
            x_fixed_vals = [fix_value(variable_by_name(m, linking_variables_sub[i])) for i in 1:length(linking_variables_sub)]
            linking_farkas_fixed  = sum(lambda[i] * x_fixed_vals[i]                                     for i in 1:length(linking_variables_sub))
            linking_farkas_plan   = sum(lambda[i] * planning_sol.values[linking_variables_sub[i]]        for i in 1:length(linking_variables_sub))
            max_fix_vs_plan_diff  = maximum(abs(x_fixed_vals[i] - planning_sol.values[linking_variables_sub[i]]) for i in 1:length(linking_variables_sub))
            linking_farkas = linking_farkas_plan  # used in cut; must be consistent with x_bar in master cut formula
            physical_farkas = cert - linking_farkas  # for logging; cert = physical_farkas + linking_farkas
            # alpha = cert - lambda^T*x_bar = physical_farkas (relative to planning_sol)
            # The cut added to master expands to: 0 >= alpha + lambda^T*x, where alpha=physical_farkas.
            # alpha=0 => cut is 0 >= lambda^T*x (origin-passing halfspace); repeated identical (lambda,alpha)
            # tuples are duplicate rows. Log relative alpha to distinguish from rounding artifacts.
            alpha_rel = abs(physical_farkas) / max(1.0, abs(cert))
            op_cost = cert
            theta_coeff = 0;
            cut_source = :farkas
            n_nz = sum(abs.(lambda) .> 1e-8)
            if op_cost > 0
                @info "Farkas cut (dual ray): op_cost=$(round(op_cost, sigdigits=4)) [physical=$(round(physical_farkas, sigdigits=6)) rel=$(round(alpha_rel, sigdigits=3)), linking_plan=$(round(linking_farkas_plan, sigdigits=4)) linking_fixed=$(round(linking_farkas_fixed, sigdigits=4))], lambda_norm=$(round(norm(lambda), sigdigits=4)), lambda_max=$(round(maximum(abs.(lambda)), sigdigits=4)), n_nonzero=$(n_nz)/$(length(lambda)), max_fix_vs_plan=$(round(max_fix_vs_plan_diff, sigdigits=3))"
                # Log all nonzero-lambda variable names to identify repeated ray structure.
                # Threshold raised to 50 (was 10) so the 20-nonzero case is always visible.
                if n_nz <= 50
                    for i in eachindex(linking_variables_sub)
                        if abs(lambda[i]) > 1e-8
                            pval  = planning_sol.values[linking_variables_sub[i]]
                            fval  = x_fixed_vals[i]
                            @info "  FARKAS_VAR var=$(linking_variables_sub[i]) λ=$(round(lambda[i],sigdigits=4)) x_plan=$(round(pval,sigdigits=4)) x_fixed=$(round(fval,sigdigits=4)) contrib_plan=$(round(lambda[i]*pval,sigdigits=4))"
                        end
                    end
                end
            else
                @warn "Farkas objective = $(round(op_cost, sigdigits=4)) ≤ 0 — Farkas certificate may be degenerate or solver returned an invalid ray. Falling back to slack approach."
                use_farkas = false
            end
        end
        if !use_farkas
            # Farkas duals unavailable or invalid — fall back to slack-based feasibility subproblem
            if feasibility_cut_mode == :phase1
                @info "Using forced dual-simplex Phase-I feasibility cut for subproblem $(subproblem_index)."
            elseif dual_status(m) == MOI.INFEASIBILITY_CERTIFICATE
                @warn "Falling back to slack feasibility subproblem (Farkas objective was ≤ 0)..."
            else
                @warn "Farkas duals unavailable (dual_status=$(dual_status(m))), falling back to slack feasibility subproblem..."
            end
            #### Feasibility cuts generation based on https://link.springer.com/chapter/10.1007/978-3-030-45771-6_7

            is_fixed(m[:slack_max]) && unfix(m[:slack_max])
            objfun = objective_function(m)
            @objective(m, Min, m[:slack_max])
            original_method = optimizer_attribute_or_missing(m, "Method")
            original_crossover = optimizer_attribute_or_missing(m, "Crossover")
            original_numeric_focus = optimizer_attribute_or_missing(m, "NumericFocus")
            duality_tolerance = tryparse(Float64, get(ENV, "BENDERS_PHASE1_DUALITY_TOL", "1e-6"))
            isnothing(duality_tolerance) && error("BENDERS_PHASE1_DUALITY_TOL must be numeric")

            try
                try
                    set_attribute(m, "Method", 1)
                catch err
                    @warn "Phase-I solver does not accept Gurobi Method=1; using configured method. error=$(sprint(showerror, err))"
                end
                optimize!(m)

                term = termination_status(m)
                primal = primal_status(m)
                dual_stat = dual_status(m)
                raw = raw_status(m)
                results = result_count(m)
                if term != MOI.OPTIMAL || !has_values(m) || !has_duals(m)
                    error("Phase-I solve cannot generate a cut: termination=$(term) primal=$(primal) dual=$(dual_stat) raw=$(repr(raw)) result_count=$(results)")
                end

                op_cost = objective_value(m)
                dual_objective = dual_objective_value(m)
                relative_duality_gap = abs(op_cost - dual_objective) / max(1.0, abs(op_cost), abs(dual_objective))
                @info "PHASE1_CUT_SOLVE: w=$(subproblem_index) method=dual_simplex termination_status=$(term) primal_status=$(primal) dual_status=$(dual_stat) raw_status=$(repr(raw)) result_count=$(results) primal_objective=$(op_cost) dual_objective=$(dual_objective) relative_duality_gap=$(relative_duality_gap) tolerance=$(duality_tolerance)"
                if !isfinite(relative_duality_gap) || relative_duality_gap > duality_tolerance
                    error("Phase-I primal/dual objective mismatch: relative_gap=$(relative_duality_gap) > tolerance=$(duality_tolerance). Refusing to generate an invalid feasibility cut.")
                end

                _audit_phase1_constraint_attribution!(
                    m,
                    subproblem_index,
                    op_cost,
                )

                lambda = [dual(FixRef(variable_by_name(m,y))) for y in linking_variables_sub]
                theta_coeff = 0
                cut_source = :phase1
                lambda_max = isempty(lambda) ? 0.0 : maximum(abs.(lambda))
                @info "Slack feasibility cut: op_cost=$(round(op_cost, sigdigits=4)), lambda_norm=$(round(norm(lambda), sigdigits=4)), lambda_max=$(round(lambda_max, sigdigits=4)), n_nonzero=$(sum(abs.(lambda) .> 1e-8))/$(length(lambda))"

                audit_phase1_at_oracle!(m, planning_sol, linking_variables_sub, op_cost, lambda, subproblem_index)
            finally
                fix.(m[:slack_max], 0.0; force=true)
                set_objective_function(m, objfun)
                !ismissing(original_method) && try; set_attribute(m, "Method", original_method); catch; end
                !ismissing(original_crossover) && try; set_attribute(m, "Crossover", original_crossover); catch; end
                !ismissing(original_numeric_focus) && try; set_attribute(m, "NumericFocus", original_numeric_focus); catch; end
            end
        end
	end

	return (
        op_cost=op_cost,
        lambda=lambda,
        theta_coeff=theta_coeff,
        cut_source=cut_source,
    )

end


function solve_local_subproblems(subproblem_local::Vector{Dict{Any,Any}},planning_sol::NamedTuple, expect_feasible_subproblems::Bool, elastic_slack::Bool=false)

    local_sol=Dict();
    for sp in subproblem_local
        m = sp[:model];
        linking_variables_sub = sp[:linking_variables_sub]
        w = sp[:subproblem_index];
        t_sp = @elapsed begin
            local_sol[w] = solve_subproblem(m,planning_sol,linking_variables_sub,expect_feasible_subproblems,elastic_slack,w);
        end
        @info "Subproblem w=$(w): status=$(termination_status(m)) time=$(round(t_sp, digits=2))s theta_coeff=$(local_sol[w].theta_coeff)"
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
function solve_subproblems(m_subproblems::DArray{Dict{Any, Any}, 1, Vector{Dict{Any, Any}}},planning_sol::NamedTuple,expect_feasible_subproblems::Bool,elastic_slack::Bool=false)

    p_id = workers();
    np_id = length(p_id);

    sub_results = [Dict() for _ in 1:np_id];

    @sync for k in 1:np_id
              @async sub_results[k]= @fetchfrom p_id[k] solve_local_subproblems(localpart(m_subproblems),planning_sol,expect_feasible_subproblems,elastic_slack); ### This is equivalent to fetch(@spawnat p .....)
    end

	sub_results = merge(sub_results...);

    return sub_results
end


function solve_subproblems(m_subproblems::Vector{Dict{Any, Any}},planning_sol::NamedTuple,expect_feasible_subproblems::Bool,elastic_slack::Bool=false)

    sub_results = solve_local_subproblems(m_subproblems,planning_sol,expect_feasible_subproblems,elastic_slack);

    return sub_results
end
