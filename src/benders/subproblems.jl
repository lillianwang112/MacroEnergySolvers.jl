
const _TRUTHY_ENV_VALUES = Set(("1", "true", "yes", "on"))

function _benders_diagnostics_enabled()
    return lowercase(strip(get(ENV, "BENDERS_FARKAS_DEBUG", "false"))) in _TRUTHY_ENV_VALUES
end

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

    # Add Big-M penalty to objective so elastic slack mode has a proper cost signal.
    # When slack_max is fixed to 0 (normal solves), this term = 0 and has no effect.
    # When unfixed (elastic mode), it makes unserved demand very expensive, driving
    # the master to build capacity. Use 100x max existing objective coefficient,
    # floored at 1.0 and capped at 1e8 to avoid numerical blow-up.
    objfun = objective_function(subproblem)
    abs_coeffs = filter(c -> c > 0.0, [abs(coefficient(objfun, v)) for v in all_variables(subproblem)])
    raw_bigm = isempty(abs_coeffs) ? 1.0 : 100.0 * maximum(abs_coeffs)
    big_m = clamp(raw_bigm, 1.0, 1e8)
    @debug "ElasticSlack Big-M penalty: $(big_m) (raw=$(round(raw_bigm, sigdigits=3)), capped=$(raw_bigm != big_m))"
    set_objective_function(subproblem, objfun + big_m * slack_max)

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

function solve_subproblem(m::Model,planning_sol::NamedTuple,linking_variables_sub::Vector{String},expect_feasible_subproblems::Bool,elastic_slack::Bool=false)

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
        lmax = isempty(lambda) ? 0.0 : maximum(abs.(lambda))
        if elastic_slack
            slack_val = value(m[:slack_max])
            if slack_val > 1e-6
                @debug "Subproblem elastic (slack=$(round(slack_val, sigdigits=4))): op_cost=$(round(op_cost, sigdigits=4)), lambda_norm=$(round(norm(lambda), sigdigits=4)), lambda_max=$(round(lmax, sigdigits=4))"
            else
                @debug "Subproblem feasible (slack=0): op_cost=$(round(op_cost, sigdigits=4)), lambda_norm=$(round(norm(lambda), sigdigits=4)), lambda_max=$(round(lmax, sigdigits=4))"
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
        @debug "Subproblem is infeasible (status=$(termination_status(m)), primal=$(primal_status(m)), dual=$(dual_status(m))), attempting Farkas dual feasibility cut..."

        # Attempt Farkas dual approach: extract dual ray directly without re-solving.
        # op_cost must be the full Farkas objective pi^T*b + lambda^T*x_bar (> 0 by certificate).
        # Using only lambda^T*x_bar (the old formula) makes the cut 0 >= lambda^T*x — a hyperplane
        # through the origin that trivially passes all x >= 0 and builds zero capacity pressure.
        use_farkas = dual_status(m) == MOI.INFEASIBILITY_CERTIFICATE
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
            if _benders_diagnostics_enabled()
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
                                idx = findfirst(',', line)
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
            op_cost = cert
            theta_coeff = 0;
            if op_cost > 0
                if _benders_diagnostics_enabled()
                    # These quantities are useful for certificate audits, but expensive and
                    # extremely verbose for production cases with many linking variables.
                    x_fixed_vals = [fix_value(variable_by_name(m, y)) for y in linking_variables_sub]
                    linking_farkas_fixed = dot(lambda, x_fixed_vals)
                    linking_farkas_plan = sum(
                        lambda[i] * planning_sol.values[linking_variables_sub[i]]
                        for i in eachindex(linking_variables_sub)
                    )
                    max_fix_vs_plan_diff = isempty(x_fixed_vals) ? 0.0 : maximum(
                        abs(x_fixed_vals[i] - planning_sol.values[linking_variables_sub[i]])
                        for i in eachindex(linking_variables_sub)
                    )
                    physical_farkas = cert - linking_farkas_plan
                    alpha_rel = abs(physical_farkas) / max(1.0, abs(cert))
                    n_nz = count(x -> abs(x) > 1e-8, lambda)
                    @info "Farkas cut (dual ray): op_cost=$(round(op_cost, sigdigits=4)) [physical=$(round(physical_farkas, sigdigits=6)) rel=$(round(alpha_rel, sigdigits=3)), linking_plan=$(round(linking_farkas_plan, sigdigits=4)) linking_fixed=$(round(linking_farkas_fixed, sigdigits=4))], lambda_norm=$(round(norm(lambda), sigdigits=4)), lambda_max=$(round(maximum(abs.(lambda)), sigdigits=4)), n_nonzero=$(n_nz)/$(length(lambda)), max_fix_vs_plan=$(round(max_fix_vs_plan_diff, sigdigits=3))"
                    if n_nz <= 50
                        for i in eachindex(linking_variables_sub)
                            if abs(lambda[i]) > 1e-8
                                pval = planning_sol.values[linking_variables_sub[i]]
                                fval = x_fixed_vals[i]
                                @info "  FARKAS_VAR var=$(linking_variables_sub[i]) λ=$(round(lambda[i],sigdigits=4)) x_plan=$(round(pval,sigdigits=4)) x_fixed=$(round(fval,sigdigits=4)) contrib_plan=$(round(lambda[i]*pval,sigdigits=4))"
                            end
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
            if dual_status(m) == MOI.INFEASIBILITY_CERTIFICATE
                @warn "Falling back to slack feasibility subproblem (Farkas objective was ≤ 0)..."
            else
                @warn "Farkas duals unavailable (dual_status=$(dual_status(m))), falling back to slack feasibility subproblem..."
            end
            #### Feasibility cuts generation based on https://link.springer.com/chapter/10.1007/978-3-030-45771-6_7

            is_fixed(m[:slack_max]) && unfix(m[:slack_max])
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
            @debug "Slack feasibility cut: op_cost=$(round(op_cost, sigdigits=4)), lambda_norm=$(round(norm(lambda), sigdigits=4)), lambda_max=$(round(maximum(abs.(lambda)), sigdigits=4)), n_nonzero=$(sum(abs.(lambda) .> 1e-8))/$(length(lambda))"

            try; set_attribute(m, "Crossover", 1); catch; end
            fix.(m[:slack_max], 0.0);
            @objective(m, Min, objfun)
        end
	end

	return (op_cost=op_cost,lambda = lambda,theta_coeff=theta_coeff)

end


function solve_local_subproblems(subproblem_local::Vector{Dict{Any,Any}},planning_sol::NamedTuple, expect_feasible_subproblems::Bool, elastic_slack::Bool=false)

    local_sol=Dict();
    for sp in subproblem_local
        m = sp[:model];
        linking_variables_sub = sp[:linking_variables_sub]
        w = sp[:subproblem_index];
        t_sp = @elapsed begin
            local_sol[w] = solve_subproblem(m,planning_sol,linking_variables_sub,expect_feasible_subproblems,elastic_slack);
        end
        @debug "Subproblem w=$(w): status=$(termination_status(m)) time=$(round(t_sp, digits=2))s theta_coeff=$(local_sol[w].theta_coeff)"
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
