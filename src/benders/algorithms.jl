"""
	benders(planning_problem::Model, 
		subproblems::Union{Vector{Dict{Any, Any}}, DistributedArrays.DArray}, 
		linking_variables_sub::Dict, 
		setup::Dict
	)

Implements a regularized Benders decomposition algorithm for solving large-scale energy systems planning problems.

## Algorithm from:
F. Pecci and J. D. Jenkins (2025). “Regularized Benders Decomposition for High Performance Capacity Expansion Models”. doi: [10.1109/TPWRS.2025.3526413](https://ieeexplore.ieee.org/document/10829583)

It's a regularized version of the Benders decomposition algorithm in:

A. Jacobson, F. Pecci, N. Sepulveda, Q. Xu, and J. Jenkins (2024). “A computationally efficient Benders decomposition for energy systems planning problems with detailed operations and time-coupling constraints.” doi: [https://doi.org/10.1287/ijoo.2023.0005](https://doi.org/10.1287/ijoo.2023.0005)

# Arguments
- `planning_problem::Model`: The upper-level problem JuMP model representing the planning decisions (e.g., investment decisions, policy budgeting variables, multi-day storage states).
- `subproblems::Union{Vector{Dict{Any, Any}},DistributedArrays.DArray}`: Collection of lower-level subproblems (i.e., the operational problems for each sub-period)
- `linking_variables_sub::Dict`: Mapping between subproblems and their associated linking variables
- `setup::Dict`: Algorithm parameters including:
	- `MaxIter`: Maximum number of iterations
	- `ConvTol`: Convergence tolerance
	- `MaxCpuTime`: Maximum CPU time allowed
	- `StabParam`: Stabilization parameter γ
	- `StabDynamic`: Boolean for dynamic stabilization adjustment
	- `IntegerInvestment`: Boolean for integer investment variables
- `reformat_logging::Bool = false`: MacroEnergySolvers will reformat logging output if set to true

# Returns
@NamedTuple containing:
- `planning_problem`: Updated upper-level problem model
- `planning_sol`: Best planning solution found
- `subop_sol`: Subproblem solutions corresponding to the best planning solution
- `LB_hist`: History of lower bounds
- `UB_hist`: History of upper bounds
- `cpu_time`: CPU time history
- `planning_sol_hist`: Solution history for linking variables
"""
function benders(planning_problem::Model,subproblems::Union{Vector{Dict{Any, Any}},DistributedArrays.DArray},linking_variables_sub::Dict,setup::Dict, reformat_logging::Bool=false)
	
    #### Algorithm from:
    ### F. Pecci and J. D. Jenkins (2025). “Regularized Benders Decomposition for High Performance Capacity Expansion Models”. doi: https://doi.org/10.1109/TPWRS.2025.3526413

	### It's a regularized version of the Benders decomposition algorithm in:
	### A. Jacobson, F. Pecci, N. Sepulveda, Q. Xu, and J. Jenkins (2024). “A computationally efficient Benders decomposition for energy systems planning problems with detailed operations and time-coupling constraints.” doi: https://doi.org/10.1287/ijoo.2023.0005

	# Initialize the MacroEnergySolvers logger
	if reformat_logging
		old_logger = set_logger(Logging.Info)
		LOGS_REFORMATTED = true
	else
		LOGS_REFORMATTED = false
	end
	@info("Running Benders decomposition algorithm from `MacroEnergySolvers.jl`")
	
	expect_feasible_subproblems = setup[:ExpectFeasibleSubproblems];
	elastic_slack = get(setup, :ElasticSlack, false);
	elastic_slack && @info("ElasticSlack=true: subproblem slack variables always unfixed; subproblems always feasible (optimality cuts only).")

	if expect_feasible_subproblems == true
		@info("Feasibility cuts will not be computed because ExpectFeasibleSubproblems is set to true.")
	else
		add_slacks_to_subproblems!(subproblems);
	end

	# Scale objectives once before any solve to improve numerical condition number.
	# With obj~1e10 and unit-scale constraints, duals are corrupted. Scaling by 1/1e6
	# brings the objective to ~1e4, which Gurobi's barrier handles cleanly.
	# Everything stays in scaled space throughout (Benders + MGA); only divide back at write_outputs.
	obj_scale = Float64(get(setup, :ObjScaleFactor, 1.0))
	if obj_scale != 1.0
		@info("Applying objective scaling by 1/$(obj_scale). LB/UB reported values will be unscaled.")
		set_objective_function(planning_problem, objective_function(planning_problem) / obj_scale)
		scale_subproblem_objectives!(subproblems, obj_scale)
	end

	# The model builder owns variable domains. Policy Budgets and storage changes
	# may be signed; inferring bounds from variable names changes the feasible set.

	add_approximate_variable_cost!(planning_problem,length(linking_variables_sub));

	## Start solver time
	solver_start_time = time()
    
    #### Algorithm parameters:
	MaxIter = setup[:MaxIter];
    ConvTol = setup[:ConvTol];
	MaxCpuTime = setup[:MaxCpuTime];
	γ = setup[:StabParam];
	term_status = "NONE";

	stab_dynamic = setup[:StabDynamic];

	if γ ≈ 0.0
		stab_method = "off";
	else
		stab_method = "int_level_set";
	end

    integer_investment = setup[:IntegerInvestment];

	integer_routine_flag = false
	# Anonymous auxiliaries remain in the master but have no name-based transfer.
	planning_variables_ref = filter(v -> !isempty(name(v)), all_variables(planning_problem));
	planning_variables = name.(planning_variables_ref);

	if integer_investment == 1 && stab_method != "off"
		integer_variables = planning_variables_ref[is_integer.(planning_variables_ref)];
		binary_variables = planning_variables_ref[is_binary.(planning_variables_ref)];
		unset_integer.(integer_variables)
		unset_binary.(binary_variables)
		set_lower_bound.(binary_variables, 0)
		set_upper_bound.(binary_variables, 1)
		integer_routine_flag = true;
	end

    #### Initialize UB and LB
	planning_sol, LB = solve_planning_problem(planning_problem,planning_variables);

	# Send the solved master point to recourse without a manual Budget allocation.

    UB = Inf;

    LB_hist = Float64[];
    UB_hist = Float64[];
	gap_hist = Float64[];
    cpu_time = Float64[];

	planning_sol_best = deepcopy(planning_sol);
	subop_sol_best = Dict{Any,Any}()

	planning_sol_hist = [planning_sol.values[s] for s in planning_variables];

    #### Run Benders iterations
    diagnostics_enabled = _benders_diagnostics_enabled()
    causal_diagnostic_iter = let raw = strip(get(ENV, "BENDERS_CAUSAL_DIAGNOSTIC_ITER", ""))
        if isempty(raw)
            nothing
        else
            parsed = tryparse(Int, raw)
            isnothing(parsed) && throw(ArgumentError(
                "BENDERS_CAUSAL_DIAGNOSTIC_ITER must be a nonnegative integer; got '$raw'",
            ))
            parsed >= 0 || throw(ArgumentError(
                "BENDERS_CAUSAL_DIAGNOSTIC_ITER must be nonnegative; got $parsed",
            ))
            parsed
        end
    end
    causal_unst_infeasible = nothing
    if !isnothing(causal_diagnostic_iter)
        @info(
            "BENDERS_CAUSAL_DIAGNOSTIC enabled at k=$causal_diagnostic_iter: " *
            "one extra read-only subproblem sweep will evaluate unst_planning_sol; " *
            "its cuts will not be added to the master.",
        )
    end
    # Retaining every Farkas ray is only needed for an explicit certificate audit.
    historical_farkas_cuts = NamedTuple{(:w, :lambda, :linking_vars, :cert, :k_added), Tuple{Any,Vector{Float64},Vector{String},Float64,Int}}[]

    for k = 0:MaxIter

		start_subop_sol = time();

		planning_sol_hist = hcat(planning_sol_hist, [planning_sol.values[s] for s in planning_variables])

        subop_sol = solve_subproblems(subproblems,planning_sol,expect_feasible_subproblems,elastic_slack);

		cpu_subop_sol = time()-start_subop_sol;
		@info("Solving the subproblems required $(tidy_timing(cpu_subop_sol)) seconds")

        if !isnothing(causal_unst_infeasible) && k == causal_diagnostic_iter + 1
            causal_stab_infeasible = Set(
                w for (w, sol) in subop_sol if sol.theta_coeff == 0
            )
            only_unst = sort!(collect(setdiff(causal_unst_infeasible, causal_stab_infeasible)))
            only_stab = sort!(collect(setdiff(causal_stab_infeasible, causal_unst_infeasible)))
            both = intersect(causal_unst_infeasible, causal_stab_infeasible)
            sample_limit = 20
            @info(
                "BENDERS_CAUSAL_COMPARISON: source_k=$causal_diagnostic_iter " *
                "unst_infeasible=$(length(causal_unst_infeasible))/$(length(subop_sol)) " *
                "stab_infeasible=$(length(causal_stab_infeasible))/$(length(subop_sol)) " *
                "both=$(length(both)) only_unst=$(length(only_unst)) " *
                "only_stab=$(length(only_stab))",
            )
            !isempty(only_unst) && @info(
                "BENDERS_CAUSAL_ONLY_UNST sample=" *
                string(first(only_unst, min(sample_limit, length(only_unst)))),
            )
            !isempty(only_stab) && @info(
                "BENDERS_CAUSAL_ONLY_STAB sample=" *
                string(first(only_stab, min(sample_limit, length(only_stab)))),
            )
        end

		UBnew = compute_upper_bound(planning_problem,planning_sol,subop_sol);
		if UBnew < UB
			planning_sol_best = deepcopy(planning_sol);
			subop_sol_best = deepcopy(subop_sol);
			UB = UBnew;
		end

		@info("Updating the planning problem....")
		time_start_update = time()

		update_planning_problem_multi_cuts!(planning_problem,subop_sol,planning_sol,linking_variables_sub,k)

        if diagnostics_enabled
            # Record Farkas cuts for cross-iteration violation checking.
            for (w, sol) in subop_sol
                if sol.theta_coeff == 0
                    push!(historical_farkas_cuts, (w=w, lambda=copy(sol.lambda), linking_vars=copy(linking_variables_sub[w]), cert=sol.op_cost, k_added=k))
                end
            end
        end

		time_planning_update = time()-time_start_update
		@info("Done updating the planning problem. It took $(tidy_timing(time_planning_update)) seconds).")

		start_planning_sol = time()

		unst_planning_sol, LBnew = solve_planning_problem(planning_problem,planning_variables);

		cpu_planning_sol = time()-start_planning_sol;
		@info("Solving the planning problem required $(tidy_timing(cpu_planning_sol)) seconds")

		LB = max(LB,LBnew);
		@info("The optimal value of the planning problem is $(obj_scale * LBnew) (scaled: $LBnew)")
		if diagnostics_enabled
			n_nonzero = sum(abs(v) > 1e-6 for v in values(unst_planning_sol.values))
			cap_vals = collect(values(unst_planning_sol.values))
			@info "Planning solution summary: $(n_nonzero)/$(length(planning_variables)) variables non-zero, sum=$(round(sum(cap_vals), sigdigits=4)), max=$(round(maximum(cap_vals), sigdigits=4)), min=$(round(minimum(cap_vals), sigdigits=4))"
		end

		running_gap = (UB-LB)/abs(LB)

		append!(LB_hist,LB)
        append!(UB_hist,UB)
        append!(cpu_time,time()-solver_start_time)
		append!(gap_hist, running_gap)

		n_infeasible = count(sol -> sol.theta_coeff == 0, values(subop_sol))
		info_string = "k = $k      LB = $(round_from_tol(obj_scale * LB, ConvTol, 2))     UB = $(round_from_tol(obj_scale * UB, ConvTol, 2))       Gap = $(round_from_tol(running_gap, ConvTol, 2))       Infeasible = $n_infeasible/$(length(subop_sol))       CPU Time = $(tidy_timing(cpu_time[end]))"
		if n_infeasible > 0
			@info("*** $info_string")
		else
			@info("$info_string")
		end
		flush(stderr)

        if running_gap <= ConvTol
			if running_gap < 0
				@info("*** Warning: Negative gap detected, terminating (Gap= $(round_from_tol(running_gap, ConvTol, 2)))  ***")
				term_status = "NEGATIVE GAP"
				break
			else
				if integer_routine_flag
					@info("*** Switching on integer constraints *** ")
					UB = Inf;
					set_integer.(integer_variables)
					set_binary.(binary_variables)
					planning_sol, LB = solve_planning_problem(planning_problem,planning_variables);
					planning_sol_best = deepcopy(planning_sol);
					# UB was reset to Inf, so there is no evaluated incumbent yet. Clear the
					# relaxed-phase subproblem solutions rather than leave them paired with the
					# new integer planning solution they do not correspond to.
					subop_sol_best = Dict{Any,Any}();
					integer_routine_flag = false;
				else
					@info("*** Terminating because optimal solution found (Gap= $(round_from_tol(running_gap, ConvTol, 2)))  ***")
					term_status = "OPTIMAL"
					break
				end
			end
		elseif (cpu_time[end] >= MaxCpuTime)
			@info("*** Terminating because CPU time limit reached (MaxCpuTime=$MaxCpuTime)  ***")
			term_status = "TIMELIMIT"
			break
		elseif k == MaxIter
			@info("*** Terminating because maximum number of iterations reached (MaxIter=$MaxIter)  ***")
			term_status = "MAXITER"
			break
		elseif UB==Inf
			planning_sol = deepcopy(unst_planning_sol);
			# No finite UB yet — track most recent solution as best so the post-Benders
			# subproblem solve (in MacroEnergy's operations.jl) uses the most cuts-informed
			# planning solution rather than the initial pre-cut solution.
			planning_sol_best = deepcopy(planning_sol);
			subop_sol_best = deepcopy(subop_sol);
		else
			if stab_method == "int_level_set"
				if stab_dynamic == true && k >= 1
					γ = update_stab_param(γ,UB_hist[end],LB_hist[end],UB_hist[end-1],LB_hist[end-1]);
				end

				start_stab_method = time()
				if  integer_investment==1 && integer_routine_flag==false
					unset_integer.(integer_variables)
					unset_binary.(binary_variables)
					for v in integer_variables
						fix(v,unst_planning_sol.values[name(v)];force=true)
					end
					for v in binary_variables
						fix(v,unst_planning_sol.values[name(v)];force=true)
					end
                    @info("Solving the interior level set problem with γ = $γ")
					planning_sol = solve_int_level_set_problem(planning_problem,planning_variables,unst_planning_sol,LB,UB,γ);
					unfix.(integer_variables)
					unfix.(binary_variables)
					set_integer.(integer_variables)
					set_binary.(binary_variables)
					set_lower_bound.(integer_variables,0.0)
					set_lower_bound.(binary_variables,0.0)
				else
                    @info("Solving the interior level set problem with γ = $γ")
					planning_sol = solve_int_level_set_problem(planning_problem,planning_variables,unst_planning_sol,LB,UB,γ);
				end
				cpu_stab_method = time()-start_stab_method;
				@info("Solving the interior level set problem required $(tidy_timing(cpu_stab_method)) seconds")
			else
				planning_sol = deepcopy(unst_planning_sol);
			end

		end

        if diagnostics_enabled
            # Post-stabilization diagnostics: planning_sol is the point sent to subproblems.
            stab_max_diff = isempty(unst_planning_sol.values) ? 0.0 : maximum(abs(get(planning_sol.values, vk, 0.0) - get(unst_planning_sol.values, vk, 0.0)) for vk in keys(unst_planning_sol.values))
            @info "STAB_DIFF: max|planning_sol - unst_planning_sol| = $(round(stab_max_diff, sigdigits=4))"

            # Log aggregate vSTOR_CHANGE sums for hydro assets at both candidates.
            hydro_keys = filter(v -> contains(v, "vSTOR_CHANGE") && contains(v, "hydroelectric"), collect(keys(unst_planning_sol.values)))
            if !isempty(hydro_keys)
                period_groups = Dict{String,Vector{String}}()
                for v in hydro_keys
                    m_ps = match(r"(period\d+\[\d+\])$", v)
                    key_ps = isnothing(m_ps) ? "unknown" : m_ps.captures[1]
                    push!(get!(period_groups, key_ps, String[]), v)
                end
                for (ps, vars) in sort(collect(period_groups), by=first)
                    sum_unst = sum(get(unst_planning_sol.values, v, 0.0) for v in vars)
                    sum_stab = sum(get(planning_sol.values, v, 0.0) for v in vars)
                    @info "HYDRO_SUM: $(ps) unst=$(round(sum_unst,sigdigits=4)) stab=$(round(sum_stab,sigdigits=4)) Δ=$(round(sum_stab-sum_unst,sigdigits=4))"
                end
            end

            # Per-cut causal split: log λᵀx at both candidates.
            for cut in historical_farkas_cuts
                lhs_unst = sum(cut.lambda[i] * get(unst_planning_sol.values, cut.linking_vars[i], 0.0) for i in 1:length(cut.linking_vars))
                lhs_stab = sum(cut.lambda[i] * (haskey(planning_sol.values, cut.linking_vars[i]) ? planning_sol.values[cut.linking_vars[i]] : error("FARKAS_CAUSAL: missing $(cut.linking_vars[i])")) for i in 1:length(cut.linking_vars))
                @info "FARKAS_CAUSAL: w=$(cut.w) k_added=$(cut.k_added) λᵀx_unst=$(round(lhs_unst,sigdigits=4)) λᵀx_stab=$(round(lhs_stab,sigdigits=4)) Δ=$(round(lhs_stab-lhs_unst,sigdigits=4))"
            end

            # Cross-iteration Farkas cut violation check on stabilized planning_sol.
            n_cut_violations = 0
            for cut in historical_farkas_cuts
                lhs = sum(cut.lambda[i] * (haskey(planning_sol.values, cut.linking_vars[i]) ? planning_sol.values[cut.linking_vars[i]] : error("FARKAS_CUT_CHECK: variable $(cut.linking_vars[i]) missing from planning_sol.values")) for i in 1:length(cut.linking_vars))
                if lhs > 1e-4
                    n_cut_violations += 1
                    @warn "FARKAS_CUT_VIOLATION: w=$(cut.w) k_added=$(cut.k_added) lambda^T*planning_sol=$(round(lhs,sigdigits=4)) > 0 (cut requires ≤ 0); cert=$(round(cut.cert,sigdigits=4))"
                end
            end
            if n_cut_violations == 0 && !isempty(historical_farkas_cuts)
                @info "FARKAS_CUT_CHECK: all $(length(historical_farkas_cuts)) historical Farkas cuts satisfied at stabilized planning_sol"
            elseif n_cut_violations > 0
                @warn "FARKAS_CUT_CHECK: $(n_cut_violations)/$(length(historical_farkas_cuts)) historical Farkas cuts VIOLATED at stabilized planning_sol — stabilization moved solution outside feasible cone"
            end
        end

        if !isnothing(causal_diagnostic_iter) && k == causal_diagnostic_iter
            @info(
                "BENDERS_CAUSAL_UNST_SWEEP_BEGIN: k=$k; evaluating unst_planning_sol " *
                "without updating the master",
            )
            causal_start = time()
            causal_unst_sol = solve_subproblems(
                subproblems,
                unst_planning_sol,
                expect_feasible_subproblems,
                elastic_slack,
            )
            causal_unst_infeasible = Set(
                w for (w, sol) in causal_unst_sol if sol.theta_coeff == 0
            )
            causal_elapsed = time() - causal_start
            @info(
                "BENDERS_CAUSAL_UNST_SWEEP_END: k=$k " *
                "infeasible=$(length(causal_unst_infeasible))/$(length(causal_unst_sol)) " *
                "time=$(tidy_timing(causal_elapsed)) seconds; no cuts added",
            )
            causal_unst_sol = nothing
            GC.gc(false)
        end

    end

	if reformat_logging && LOGS_REFORMATTED
		# Restore the old logger
		set_logger(old_logger)
	end
	
	return (planning_problem=planning_problem,planning_sol = planning_sol_best, subop_sol = subop_sol_best,LB_hist = LB_hist,UB_hist = UB_hist,gap_hist = gap_hist, termination_status = term_status, cpu_time = cpu_time, planning_sol_hist = planning_sol_hist)
	
end

function update_planning_problem_multi_cuts!(m::Model,subop_sol::Dict,planning_sol::NamedTuple,linking_variables_sub::Dict,k::Int64)

	W = keys(subop_sol);

    @constraint(m,[w in W],subop_sol[w].theta_coeff*m[:vTHETA][w] >= subop_sol[w].op_cost + sum(subop_sol[w].lambda[i]*(variable_by_name(m,linking_variables_sub[w][i]) - planning_sol.values[linking_variables_sub[w][i]]) for i in 1:length(linking_variables_sub[w])), base_name="BendersCut_0_"*string(k));

end

function update_stab_param(γ,UB,LB,UB_old,LB_old)
	
	r=(UB_old-UB)/(UB_old-(LB+γ*(UB_old-LB)))
					
	ap=(UB_old-UB)
	pp=(UB_old-(LB+γ*(UB_old-LB)))
	@info(r, ap, pp)
	if ap>=0 && pp>=0
		if r<=0.2
			γ=0.9-0.5*(0.9-γ)
			@info("Increase γ: ", γ)
		elseif 0.2<r<0.8
			γ=γ
			@info("Keep γ: ", γ)
		else
			γ=0.5*γ
			@info("Decrease γ: ", γ)
		end
	end
	
	return γ
end

function compute_upper_bound(m::Model,planning_sol::NamedTuple,subop_sol::Dict)
	any(subop_sol[w].theta_coeff==0 for w in keys(subop_sol)) && return Inf;

	return  planning_sol.planning_cost + sum(subop_sol[w].op_cost for w in keys(subop_sol))
end
