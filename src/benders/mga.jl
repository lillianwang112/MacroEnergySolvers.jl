function setup_mga_master_problem!(planning_problem::Model, setup::Dict)
    if haskey(planning_problem.obj_dict, :cMGABudget)
        old_budget = planning_problem[:cMGABudget]
        if is_valid(planning_problem, old_budget)
            delete(planning_problem, old_budget)
        end
        unregister(planning_problem, :cMGABudget)
    end
    @constraint(
        planning_problem,
        cMGABudget,
        planning_problem[:ePlanningCost] + sum(planning_problem[:vTHETA]) <= setup[:MGABudget],
    )
    return planning_problem[:cMGABudget]
end

function forget_cuts_master!(planning_problem::Model, master_cons::AbstractVector{String})
    retained = Set(master_cons)
    for con in all_constraints(planning_problem, include_variable_in_set_constraints=false)
        if !(name(con) in retained)
            delete(planning_problem, con)
        end
    end
    return nothing
end

function _nonzero_random_column(rng::AbstractRNG, nvars::Int, distribution::Symbol)
    while true
        column = if distribution == :normal
            randn(rng, nvars)
        elseif distribution == :capmm
            Float64.(rand(rng, -1:1, nvars))
        else
            error("Unsupported MGA vector distribution: $distribution")
        end
        norm(column) > 0 && return column
    end
end

function _paired_vecs(rng::AbstractRNG, nvars::Int, iterations::Int, distribution::Symbol)
    nbase = cld(iterations, 2)
    base = hcat((_nonzero_random_column(rng, nvars, distribution) for _ in 1:nbase)...)
    return hcat(base, -base)[:, 1:iterations]
end

make_rand_vecs(nvars::Int, iterations::Int; rng::AbstractRNG=Random.default_rng()) =
    _paired_vecs(rng, nvars, iterations, :normal)

make_capmm_vecs(nvars::Int, iterations::Int; rng::AbstractRNG=Random.default_rng()) =
    _paired_vecs(rng, nvars, iterations, :capmm)

function find_ratio(setup::Dict)
    setup[:MGAMethod] == 0 || return 0.0
    ratio = Float64(get(setup, :MGAComboRatio, 0.25))
    0.0 <= ratio <= 1.0 || throw(ArgumentError("MGAComboRatio must be in [0, 1], got $ratio"))
    return ratio
end

function _normalize_columns(vecs::AbstractMatrix)
    normalized = Matrix{Float64}(vecs)
    for j in axes(normalized, 2)
        column_norm = norm(view(normalized, :, j))
        column_norm > 0 || throw(ArgumentError("MGA vector column $j is zero"))
        normalized[:, j] ./= column_norm
    end
    return normalized
end

function generate_vecs(setup::Dict, variables::Vector{String})
    iterations = Int(setup[:MGAIterations])
    iterations > 0 || throw(ArgumentError("MGAIterations must be positive, got $iterations"))
    nvars = length(variables)
    nvars > 0 || throw(ArgumentError("At least one MGA variable is required"))

    method = Int(setup[:MGAMethod])
    rng = MersenneTwister(Int(get(setup, :MGARandomSeed, 42)))

    vecs = if method == 1
        make_rand_vecs(nvars, iterations; rng)
    elseif method == 2
        make_capmm_vecs(nvars, iterations; rng)
    elseif method == 0
        n_random = round(Int, iterations * find_ratio(setup))
        n_capmm = iterations - n_random
        random_vecs = n_random == 0 ? zeros(nvars, 0) : make_rand_vecs(nvars, n_random; rng)
        capmm_vecs = n_capmm == 0 ? zeros(nvars, 0) : make_capmm_vecs(nvars, n_capmm; rng)
        hcat(random_vecs, capmm_vecs)
    elseif method == 3
        haskey(setup, :MGAUserVecs) || throw(ArgumentError("MGAMethod=3 requires MGAUserVecs"))
        Matrix{Float64}(setup[:MGAUserVecs])
    else
        throw(ArgumentError("Invalid MGAMethod=$method; use 0=combo, 1=random, 2=capMM, or 3=custom"))
    end

    size(vecs) == (nvars, iterations) || throw(DimensionMismatch(
        "MGA vectors have size $(size(vecs)); expected ($nvars, $iterations)",
    ))
    return reorder_vecs(_normalize_columns(vecs), setup)
end

function reorder_vecs(vecs::AbstractMatrix, setup::Dict)
    method = String(get(setup, :MGAVectorSortMethod, "none"))
    if method == "angle"
        reference = view(vecs, :, 1)
        similarities = [clamp(dot(view(vecs, :, i), reference), -1.0, 1.0) for i in axes(vecs, 2)]
        return vecs[:, sortperm(acos.(similarities))]
    elseif method == "nearest-neighbor"
        selected = [1]
        while length(selected) < size(vecs, 2)
            last = view(vecs, :, selected[end])
            candidates = setdiff(collect(axes(vecs, 2)), selected)
            distances = [norm(view(vecs, :, j) - last) for j in candidates]
            push!(selected, candidates[argmin(distances)])
        end
        return vecs[:, selected]
    elseif method == "none" || isempty(method)
        return vecs
    end
    throw(ArgumentError("Unknown MGAVectorSortMethod=$method"))
end

function _parse_benders_cut_name(n::String)
    m = match(r"^BendersCut_(\d+)_(\d+)", n)
    return isnothing(m) ? nothing : (parse(Int, m.captures[1]), parse(Int, m.captures[2]))
end

function _cut_groups(planning_problem::Model)
    structural = String[]
    groups = Dict{Tuple{Int,Int},Vector{String}}()
    for con in all_constraints(planning_problem, include_variable_in_set_constraints=false)
        n = name(con)
        key = _parse_benders_cut_name(n)
        if isnothing(key)
            push!(structural, n)
        else
            push!(get!(groups, key, String[]), n)
        end
    end
    return structural, groups
end

function retain_fixed_spcuts_early(planning_problem::Model, num_groups::Int, ::Int)
    num_groups >= 0 || throw(ArgumentError("MGAMaxCuts must be nonnegative"))
    structural, groups = _cut_groups(planning_problem)
    ordered = sort(collect(keys(groups)))
    retained = copy(structural)
    for key in ordered[1:min(num_groups, length(ordered))]
        append!(retained, groups[key])
    end
    return retained
end

function retain_early_cuts_latest_iterations(planning_problem::Model, num_groups::Int, iteration::Int)
    num_groups >= 0 || throw(ArgumentError("MGAMaxCuts must be nonnegative"))
    structural, groups = _cut_groups(planning_problem)
    num_groups == 0 && return structural

    baseline = sort([key for key in keys(groups) if key[1] == 0])
    early_count = min(cld(length(baseline), 4), num_groups)
    selected = baseline[1:early_count]

    recent = sort(
        [key for key in keys(groups) if 1 <= key[1] <= iteration && !(key in selected)];
        by=key -> (-key[1], key[2]),
    )
    for key in recent
        length(selected) >= num_groups && break
        push!(selected, key)
    end

    # If recent groups do not fill the budget, retain additional baseline groups.
    for key in baseline
        length(selected) >= num_groups && break
        key in selected || push!(selected, key)
    end

    retained = copy(structural)
    for key in selected
        append!(retained, groups[key])
    end
    return retained
end

function update_planning_problem_multi_cuts_mga!(
    planning_problem::Model,
    subop_sol::Dict,
    planning_sol::NamedTuple,
    linking_vars_sub::Dict,
    mga_it::Int,
    benders_it::Int,
)
    W = keys(subop_sol)
    cut_name = "BendersCut_$(mga_it)_$(benders_it)"
    @constraint(
        planning_problem,
        [w in W],
        subop_sol[w].theta_coeff * planning_problem[:vTHETA][w] >=
        subop_sol[w].op_cost + sum(
            subop_sol[w].lambda[i] * (
                variable_by_name(planning_problem, linking_vars_sub[w][i]) -
                planning_sol.values[linking_vars_sub[w][i]]
            ) for i in eachindex(linking_vars_sub[w])
        ),
        base_name=cut_name,
    )
    return nothing
end

function validate_mga_variables(planning_problem::Model, variables::Vector{String}, setup::Dict)
    isempty(variables) && throw(ArgumentError(
        "No Benders-MGA variables were selected. Pass explicit capacity variables or aggregated MGA variables.",
    ))
    length(unique(variables)) == length(variables) || throw(ArgumentError("MGA variable names must be unique"))
    missing = filter(name -> isnothing(variable_by_name(planning_problem, name)), variables)
    isempty(missing) || throw(ArgumentError("MGA variables not found in the planning model: $(join(missing, ", "))"))

    unsafe = filter(name -> name == "vREF" || startswith(name, "vTHETA"), variables)
    if !isempty(unsafe) && !Bool(get(setup, :MGAAllowUnsafeVariables, false))
        throw(ArgumentError(
            "MGA variables include cost/reference auxiliaries ($(join(unsafe, ", "))). " *
            "Set MGAAllowUnsafeVariables=true only if this is intentional.",
        ))
    end
    return variables
end

function _atomic_serialize(path::AbstractString, payload)
    mkpath(dirname(path))
    temporary = path * ".tmp.$(getpid())"
    try
        open(temporary, "w") do io
            serialize(io, payload)
            flush(io)
        end
        mv(temporary, path; force=true)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return path
end

function _write_text_atomic(path::AbstractString, contents::AbstractString)
    mkpath(dirname(path))
    temporary = path * ".tmp.$(getpid())"
    try
        open(temporary, "w") do io
            write(io, contents)
            flush(io)
        end
        mv(temporary, path; force=true)
    finally
        isfile(temporary) && rm(temporary; force=true)
    end
    return path
end

function write_mga_checkpoint(
    checkpoint_dir::AbstractString,
    iteration::Int,
    result::NamedTuple,
    vector::AbstractVector,
    variables::Vector{String},
    setup::Dict,
)
    payload = (
        schema_version=1,
        iteration=iteration,
        written_at=Dates.now(),
        variables=copy(variables),
        vector=collect(Float64, vector),
        mga_budget=Float64(setup[:MGABudget]),
        result=result,
    )
    stem = "mga_iteration_$(lpad(iteration, 4, '0'))"
    binary_path = _atomic_serialize(joinpath(checkpoint_dir, stem * ".jls"), payload)

    summary = join([
        "schema_version\t1",
        "iteration\t$iteration",
        "status\t$(result.status)",
        "converged\t$(result.converged)",
        "inner_iterations\t$(result.iterations)",
        "true_system_cost\t$(result.true_system_cost)",
        "best_true_system_cost\t$(result.best_true_system_cost)",
        "mga_budget\t$(setup[:MGABudget])",
        "budget_violation\t$(result.budget_violation)",
        "checkpoint\t$(basename(binary_path))",
    ], "\n") * "\n"
    _write_text_atomic(joinpath(checkpoint_dir, stem * "_summary.tsv"), summary)
    return binary_path
end

function _mga_result(
    status::Symbol,
    planning_sol_best,
    subop_sol_best,
    approximate_cost_hist,
    true_cost_hist,
    cpu_time,
    best_true_system_cost,
    selected_true_system_cost,
    budget,
    iteration_count,
)
    violation = isfinite(selected_true_system_cost) ?
        (selected_true_system_cost - budget) / max(abs(budget), eps(Float64)) : Inf
    return (
        status=status,
        converged=status == :converged,
        iterations=iteration_count,
        planning_sol=planning_sol_best,
        subop_sol=subop_sol_best,
        ApproxSystemCost_hist=approximate_cost_hist,
        TrueSystemCost_hist=true_cost_hist,
        cpu_time=cpu_time,
        true_system_cost=selected_true_system_cost,
        best_true_system_cost=best_true_system_cost,
        budget_violation=violation,
    )
end

function mga_cutting_plane(
    planning_problem::Model,
    subproblems,
    linking_variables_sub::Dict,
    setup::Dict,
    mga_it::Int,
)
    solver_start_time = time()
    indicator = 0

    max_iter = Int(get(setup, :MGAMaxIter, setup[:MaxIter]))
    max_cpu_time = Float64(get(setup, :MGAMaxCpuTime, setup[:MaxCpuTime]))
    max_iter > 0 || throw(ArgumentError("MGAMaxIter must be positive"))
    max_cpu_time > 0 || throw(ArgumentError("MGAMaxCpuTime must be positive"))

    expect_feasible = Bool(setup[:ExpectFeasibleSubproblems])
    elastic_slack = Bool(get(setup, :ElasticSlack, false))
    budget = Float64(setup[:MGABudget])
    relaxed_budget = Float64(get(setup, :MGARelaxBudget, get(setup, :RelaxBudget, 0.0)))
    relaxed_budget >= 0 || throw(ArgumentError("MGARelaxBudget must be nonnegative"))

    approximate_cost_hist = Float64[]
    true_cost_hist = Float64[]
    cpu_time = Float64[]
    planning_sol_best = nothing
    subop_sol_best = nothing
    best_true_system_cost = Inf

    planning_variables = name.(all_variables(planning_problem))

    for k in 1:max_iter
        start_planning = time()
        planning_sol, _ = solve_planning_problem(planning_problem, planning_variables)
        planning_time = time() - start_planning
        @info("Solving the planning problem required $(tidy_timing(planning_time)) seconds")

        approximate_cost = value(planning_problem[:ePlanningCost]) + sum(value, planning_problem[:vTHETA])

        start_subproblems = time()
        subop_sol = solve_subproblems(
            subproblems,
            planning_sol,
            expect_feasible,
            elastic_slack,
        )
        subproblem_time = time() - start_subproblems
        @info("Solving the subproblems required $(tidy_timing(subproblem_time)) seconds")

        operationally_feasible = all(sol.theta_coeff != 0 for sol in values(subop_sol))
        true_system_cost = operationally_feasible ?
            compute_upper_bound(planning_problem, planning_sol, subop_sol) : Inf

        if isfinite(true_system_cost) && true_system_cost < best_true_system_cost
            best_true_system_cost = true_system_cost
            planning_sol_best = deepcopy(planning_sol)
            subop_sol_best = deepcopy(subop_sol)
        end

        push!(approximate_cost_hist, approximate_cost)
        push!(true_cost_hist, true_system_cost)
        push!(cpu_time, time() - solver_start_time)

        budget_violation = isfinite(true_system_cost) ?
            (true_system_cost - budget) / max(abs(budget), eps(Float64)) : Inf
        @info(
            "MGA iteration=$mga_it inner_k=$k approximate_cost=$approximate_cost " *
            "true_cost=$true_system_cost best_true_cost=$best_true_system_cost " *
            "operationally_feasible=$operationally_feasible budget_violation=$budget_violation " *
            "elapsed=$(tidy_timing(cpu_time[end]))",
        )

        within_budget = operationally_feasible && budget_violation <= relaxed_budget
        if within_budget && indicator == 1
            @info("MGA iteration $mga_it converged after $k inner iterations")
            return _mga_result(
                :converged,
                deepcopy(planning_sol),
                deepcopy(subop_sol),
                approximate_cost_hist,
                true_cost_hist,
                cpu_time,
                best_true_system_cost,
                true_system_cost,
                budget,
                k,
            )
        elseif cpu_time[end] >= max_cpu_time
            @warn("MGA iteration $mga_it hit MGAMaxCpuTime=$max_cpu_time")
            return _mga_result(
                :time_limit,
                planning_sol_best,
                subop_sol_best,
                approximate_cost_hist,
                true_cost_hist,
                cpu_time,
                best_true_system_cost,
                best_true_system_cost,
                budget,
                k,
            )
        elseif within_budget
            @info("MGA iteration $mga_it found a budget-feasible point; confirming with crossover")
            try; set_attribute(planning_problem, "Crossover", 1); catch; end
            try; set_attribute(planning_problem, "run_crossover", "on"); catch; end
            indicator = 1
        else
            @info("Updating the MGA planning problem")
            update_planning_problem_multi_cuts_mga!(
                planning_problem,
                subop_sol,
                planning_sol,
                linking_variables_sub,
                mga_it,
                k,
            )
        end
    end

    @warn("MGA iteration $mga_it reached MGAMaxIter=$max_iter without converging")
    return _mga_result(
        :max_iter,
        planning_sol_best,
        subop_sol_best,
        approximate_cost_hist,
        true_cost_hist,
        cpu_time,
        best_true_system_cost,
        best_true_system_cost,
        budget,
        max_iter,
    )
end

function _validate_benders_result_for_mga(setup::Dict, benders_result)
    convergence = benders_result.convergence
    isnothing(convergence) && throw(ArgumentError("Benders-MGA requires completed Benders convergence data"))

    finite_ubs = filter(isfinite, convergence.UB_hist)
    isempty(finite_ubs) && throw(ArgumentError(
        "Benders-MGA requires at least one finite Benders upper bound; no feasible incumbent was found",
    ))

    status = uppercase(String(convergence.termination_status))
    final_gap = isempty(convergence.gap_hist) ? Inf : convergence.gap_hist[end]
    if Bool(get(setup, :MGARequireOptimalBenders, true)) && status != "OPTIMAL"
        throw(ArgumentError(
            "Benders terminated with status=$status and gap=$final_gap. " *
            "Set MGARequireOptimalBenders=false only to intentionally use a non-optimal incumbent.",
        ))
    elseif status != "OPTIMAL"
        @warn("Starting MGA from non-optimal Benders incumbent: status=$status gap=$final_gap")
    end
    return minimum(finite_ubs)
end

function benders_mga(
    planning_problem::Model,
    subproblems::Union{Vector{Dict{Any,Any}},DistributedArrays.DArray},
    linking_variables_sub::Dict,
    setup::Dict,
    benders_result,
    variables::Vector{String};
    checkpoint_dir::Union{Nothing,AbstractString}=nothing,
    iteration_callback::Union{Nothing,Function}=nothing,
)
    validate_mga_variables(planning_problem, variables, setup)

    original_objective = objective_function(planning_problem)
    original_sense = objective_sense(planning_problem)
    original_constraints = all_constraints(
        planning_problem;
        include_variable_in_set_constraints=false,
    )

    slack = Float64(setup[:MGASlack])
    slack >= 0 || throw(ArgumentError("MGASlack must be nonnegative"))
    best_ub = _validate_benders_result_for_mga(setup, benders_result)
    setup[:MGABudget] = best_ub + slack * abs(best_ub)

    setup_mga_master_problem!(planning_problem, setup)
    optimal_constraints = name.(all_constraints(
        planning_problem;
        include_variable_in_set_constraints=false,
    ))

    iterations = Int(setup[:MGAIterations])
    retain_master_cuts = Int(setup[:MGARetainBendersCuts])
    vectors = generate_vecs(setup, variables)
    results = Vector{Any}(undef, iterations)

    if !isnothing(checkpoint_dir)
        mkpath(checkpoint_dir)
        _atomic_serialize(joinpath(checkpoint_dir, "mga_run_metadata.jls"), (
            schema_version=1,
            written_at=Dates.now(),
            variables=copy(variables),
            vectors=copy(vectors),
            setup=copy(setup),
        ))
        variable_lines = ["index\tvariable"]
        append!(variable_lines, ["$i\t$(variables[i])" for i in eachindex(variables)])
        _write_text_atomic(
            joinpath(checkpoint_dir, "mga_variables.tsv"),
            join(variable_lines, "\n") * "\n",
        )

        vector_lines = ["iteration\t" * join(variables, '\t')]
        for iteration in axes(vectors, 2)
            push!(vector_lines, "$iteration\t" * join(view(vectors, :, iteration), '\t'))
        end
        _write_text_atomic(
            joinpath(checkpoint_dir, "mga_vectors.tsv"),
            join(vector_lines, "\n") * "\n",
        )
    end

    try
        for iteration in 1:iterations
            @info("Starting MGA iteration $iteration of $iterations")

            if retain_master_cuts == 1
                nothing
            elseif retain_master_cuts == 2
                forget_cuts_master!(planning_problem, optimal_constraints)
            elseif retain_master_cuts == 3
                retained = retain_fixed_spcuts_early(
                    planning_problem,
                    Int(setup[:MGAMaxCuts]),
                    iteration,
                )
                forget_cuts_master!(planning_problem, retained)
            elseif retain_master_cuts == 4
                retained = retain_early_cuts_latest_iterations(
                    planning_problem,
                    Int(setup[:MGAMaxCuts]),
                    iteration,
                )
                forget_cuts_master!(planning_problem, retained)
            else
                throw(ArgumentError("MGARetainBendersCuts must be 1, 2, 3, or 4"))
            end

            objective_terms = map(eachindex(variables)) do i
                variable_by_name(planning_problem, variables[i]) * vectors[i, iteration]
            end
            @objective(planning_problem, Min, sum(objective_terms))

            result = mga_cutting_plane(
                planning_problem,
                subproblems,
                linking_variables_sub,
                setup,
                iteration,
            )
            results[iteration] = result

            if !isnothing(checkpoint_dir)
                checkpoint = write_mga_checkpoint(
                    checkpoint_dir,
                    iteration,
                    result,
                    view(vectors, :, iteration),
                    variables,
                    setup,
                )
                @info("Wrote durable MGA checkpoint: $checkpoint")
            end
            if !isnothing(iteration_callback)
                iteration_callback(iteration, result, view(vectors, :, iteration), variables)
            end
            # Keep the converged master result valid through the callback: changing
            # optimizer attributes invalidates JuMP primal/dual values. Reset
            # crossover only after detailed output extraction has completed.
            try; set_attribute(planning_problem, "Crossover", 0); catch; end
            try; set_attribute(planning_problem, "run_crossover", "off"); catch; end
            if !result.converged && !Bool(get(setup, :MGAContinueOnFailure, false))
                @warn(
                    "Stopping Benders-MGA after direction $iteration ended with status=$(result.status). " *
                    "Set MGAContinueOnFailure=true only to retain the old continue-on-failure behavior.",
                )
                resize!(results, iteration)
                return results, vectors[:, 1:iteration], variables
            end
        end

        return results, vectors, variables
    finally
        try; set_attribute(planning_problem, "Crossover", 0); catch; end
        try; set_attribute(planning_problem, "run_crossover", "off"); catch; end
        set_objective_sense(planning_problem, original_sense)
        set_objective_function(planning_problem, original_objective)

        original_set = Set(original_constraints)
        for constraint in all_constraints(
            planning_problem;
            include_variable_in_set_constraints=false,
        )
            constraint in original_set || delete(planning_problem, constraint)
        end
        if haskey(planning_problem.obj_dict, :cMGABudget)
            unregister(planning_problem, :cMGABudget)
        end
        if retain_master_cuts in (3, 4) && any(!is_valid(planning_problem, constraint) for constraint in original_constraints)
            @warn(
                "MGARetainBendersCuts=$retain_master_cuts deleted baseline Benders cuts that cannot be " *
                "restored in-place; use the default value 2 when the returned master must remain reusable.",
            )
        end
    end
end
