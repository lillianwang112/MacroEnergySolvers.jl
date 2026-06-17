function setup_mga_master_problem!(planning_problem::Model, setup::Dict)
    @constraint(planning_problem, cMGABudget, planning_problem[:ePlanningCost] + sum(planning_problem[:vTHETA]) <= setup[:MGABudget])
end

function name_cuts!(planning_problem::Model, counter::Int64)
    for con in all_constraints(planning_problem, include_variable_in_set_constraints=false)
        if name(con) == ""
            set_name(con, "BendersCut_0_"*string(counter))
        end
        counter+=1
    end
    return counter
end

function forget_cuts_master!(planning_problem::Model, master_cons::Vector{String})
    for con in all_constraints(planning_problem, include_variable_in_set_constraints=false)
        if !(name(con) in master_cons)
            delete(planning_problem, con)
        end
    end
end

function make_rand_vecs(nvars::Int64, iterations::Int64)
    vecs = randn(nvars, ceil(Int64, iterations/2))
    vecs = hcat(vecs, -vecs)
    vecs = vecs[:, 1:iterations]
    return vecs
end

function make_capmm_vecs(nvars::Int64, iterations::Int64)
    # Potential GenX bug: if nvars is small, unique() may return fewer columns than
    # ceil(iterations/2) since there are only 3^nvars possible {-1,0,1} vectors.
    # Fix: clamp the column slice to however many unique columns actually exist.
    raw = unique(rand(-1:1, nvars, ceil(Int64, iterations)), dims=2)
    half = min(ceil(Int64, iterations/2), size(raw, 2))
    vecs = raw[:, 1:half]
    vecs = hcat(vecs, -vecs)
    vecs = vecs[:, 1:min(iterations, size(vecs, 2))]
    return vecs
end

function find_ratio(setup::Dict)
    if setup[:MGAMethod] == 0
        if setup[:MGAComboRatio] >= 0 && setup[:MGAComboRatio] <= 1
            return setup[:MGAComboRatio]
        else
            println("Invalid combo ratio specified. Defaulting to 0.25 (i.e. 25% random vectors and 75% capMM vectors).")
            return 0.25
        end
    else
        return 0
    end
end

function generate_vecs(setup::Dict, variables::Vector{String})
    iterations = setup[:MGAIterations]
    method = setup[:MGAMethod]
    combo_ratio = find_ratio(setup)
    seed = setup[:MGARandomSeed]

    nvars = length(variables)
    vecs = Array{Float64,2}(undef, nvars, iterations)
    Random.seed!(seed)

    if method == 1
        vecs = make_rand_vecs(nvars, iterations)
    elseif method == 2
        vecs = make_capmm_vecs(nvars, iterations)
    elseif method == 0
        vecs_a = make_rand_vecs(nvars, ceil(Int64, iterations*combo_ratio))
        vecs_b = make_capmm_vecs(nvars, ceil(Int64, iterations*(1-combo_ratio)))
        vecs = hcat(vecs_a, vecs_b)[:, 1:iterations]
    elseif method == 3
        vecs = setup[:MGAUserVecs]
    else
        error("Invalid MGA_Method specified. Please specify 0 for combination of random and capMM vectors, 1 for random vectors, 2 for capMM vectors, or 3 for custom vectors.")
    end
    vecs = reorder_vecs(vecs, setup)
    return vecs
end

function reorder_vecs(vecs::AbstractArray, setup::Dict)
    if setup[:MGAVectorSortMethod] == "angle"
        norms = [norm(vecs[:,i]) for i in 1:size(vecs,2)]
        dot_product = collect(dot(vecs[:,i], vecs[:,1]) for i in 1:size(vecs,2)) ./ (norms .* norm(vecs[:,1]))
        dot_product = clamp.(dot_product, -1.0, 1.0)
        angles = [acos(dot_product[i]) for i in 1:size(vecs,2)]
        sorted_indices = sortperm(angles)
        vecs = vecs[:,sorted_indices]
    elseif setup[:MGAVectorSortMethod] == "nearest-neighbor"
        sorted_indices = [1]
        for i in 2:size(vecs,2)
            last_vec = vecs[:,sorted_indices[end]]
            # Assign Inf to already-selected indices so argmin never picks them again (fixes GenX bug)
            distances = [j in sorted_indices ? Inf : norm(vecs[:,j] - last_vec) for j in 1:size(vecs,2)]
            sorted_indices = vcat(sorted_indices, argmin(distances))
        end
        vecs = vecs[:,sorted_indices]  # apply the reordering (fixes second GenX bug)
    end
    return vecs
end

function retain_fixed_spcuts_early(planning_problem::Model, num_cuts::Int64, iterations::Int64)
    cut_names = Vector{String}(undef, 0)
    cuts_by_iteration = Dict{Int64, Vector{String}}()
    for i in 0:iterations
        cuts_by_iteration[i] = Vector{String}(undef, 0)
    end

    struc_names = Vector{String}(undef, 0)
    for con in all_constraints(planning_problem, include_variable_in_set_constraints=false)
        if occursin("BendersCut", name(con))
            split_name = split(name(con), "_")
            push!(cuts_by_iteration[parse(Int, split_name[2])], name(con))
        else
            push!(struc_names, name(con))
        end
    end

    for i in 0:iterations
        cut_names = vcat(cut_names, cuts_by_iteration[i])
    end
    cut_names = cut_names[1:min(num_cuts, length(cut_names))]
    return vcat(struc_names, cut_names)
end

function retain_early_cuts_latest_iterations(planning_problem::Model, num_cuts::Int64, iteration::Int64)
    cuts_by_iteration = Dict{Int64, Vector{String}}()
    struc_names = Vector{String}(undef, 0)
    for i in 0:iteration
        cuts_by_iteration[i] = Vector{String}(undef, 0)
    end

    for con in all_constraints(planning_problem, include_variable_in_set_constraints=false)
        if occursin("BendersCut", name(con))
            split_name = split(name(con), "_")
            push!(cuts_by_iteration[parse(Int, split_name[2])], name(con))
        else
            push!(struc_names, name(con))
        end
    end

    cuts_saved_per_it = ceil(Int64, length(cuts_by_iteration[0])/4)

    if sum(length.(values(cuts_by_iteration))) >= num_cuts
        cut_names = cuts_by_iteration[0][1:min(cuts_saved_per_it, length(cuts_by_iteration[0]))]
    else
        cut_names = cuts_by_iteration[0]
    end

    for i in length(keys(cuts_by_iteration))-1:-1:1
        if length(cut_names) >= num_cuts
            cut_names = cut_names[1:num_cuts]
            break
        end
        if sum(length.(values(cuts_by_iteration))) >= num_cuts
            cut_names = vcat(cut_names, cuts_by_iteration[i][1:min(cuts_saved_per_it, length(cuts_by_iteration[i]))])
        else
            cut_names = vcat(cut_names, cuts_by_iteration[i])
        end
    end

    new_master_cons = struc_names
    append!(new_master_cons, cut_names)
    return new_master_cons
end

function update_planning_problem_multi_cuts_mga!(planning_problem::Model, subop_sol::Dict, planning_sol::NamedTuple, linking_vars_sub::Dict, mga_it::Int64, benders_it::Int64)
    W = keys(subop_sol)
    # Renamed from GenX's "name" variable which shadows JuMP's built-in name() function
    cut_name = "BendersCut_" * string(mga_it) * "_" * string(benders_it)
    @constraint(planning_problem, [w in W], subop_sol[w].theta_coeff * planning_problem[:vTHETA][w] >= subop_sol[w].op_cost + sum(subop_sol[w].lambda[i] * (variable_by_name(planning_problem, linking_vars_sub[w][i]) - planning_sol.values[linking_vars_sub[w][i]]) for i in 1:length(linking_vars_sub[w])), base_name = cut_name * "_" * string(w))
end

function mga_cutting_plane(planning_problem::Model, subproblems, linking_variables_sub::Dict, setup::Dict, mga_it::Int64)
    # Inner Benders loop for one MGA iteration. Convergence criterion: true system cost <= MGABudget
    # (unlike main benders() which converges on the optimality gap)
    cpu_time = [0.0]
    solver_start_time = time()
    # indicator: two-stage crossover — first hit triggers Crossover=1 for a clean vertex, second confirms done
    indicator = 0

    MaxIter = setup[:MaxIter]
    MaxCpuTime = setup[:MaxCpuTime]
    expect_feasible_subproblems = setup[:ExpectFeasibleSubproblems]

    TrueSystemCost = Inf
    ApproxSystemCost = setup[:MGABudget]

    ApproxSystemCost_hist = [ApproxSystemCost]
    TrueSystemCost_hist = [TrueSystemCost]
    planning_sol_final = (planning_cost = 0.0, values = Dict())
    subop_sol = Dict()

    master_times = Vector{Float64}(undef, 0)
    sub_times = Vector{Float64}(undef, 0)

    planning_variables = name.(all_variables(planning_problem))

    for k in 1:MaxIter
        start_planning_sol = time()
        planning_sol, _ = solve_planning_problem(planning_problem, planning_variables)
        cpu_planning_sol = time() - start_planning_sol
        @info("Solving the planning problem required $(tidy_timing(cpu_planning_sol)) seconds")

        start_subop_sol = time()
        subop_sol = solve_subproblems(subproblems, planning_sol, expect_feasible_subproblems)
        cpu_subop_sol = time() - start_subop_sol
        push!(sub_times, cpu_subop_sol)
        @info("Solving the subproblems required $(tidy_timing(cpu_subop_sol)) seconds")

        TrueSystemCost_new = sum(subop_sol[w].op_cost for w in keys(subop_sol)) + planning_sol.planning_cost

        if TrueSystemCost_new <= TrueSystemCost
            TrueSystemCost = copy(TrueSystemCost_new)
            planning_sol_final = deepcopy(planning_sol)
        end

        append!(ApproxSystemCost_hist, ApproxSystemCost)
        append!(TrueSystemCost_hist, TrueSystemCost)
        append!(cpu_time, time() - solver_start_time)

        budget_violation = (TrueSystemCost_new - setup[:MGABudget]) / abs(setup[:MGABudget])
        @info("k = $k      ApproxSystemCost = $ApproxSystemCost     TrueSystemCost = $TrueSystemCost     TrueSystemCost_new = $TrueSystemCost_new       MGABudget Violation = $budget_violation       CPU Time = $(tidy_timing(cpu_time[end]))")

        within_budget = TrueSystemCost_new <= setup[:MGABudget]
        within_relaxed_budget = haskey(setup, :RelaxBudget) && setup[:RelaxBudget] > 0 && isapprox(TrueSystemCost_new, setup[:MGABudget], rtol=setup[:RelaxBudget])

        if within_budget || within_relaxed_budget
            if indicator == 0
                @info("Rerunning with crossover on")
                # Try both Gurobi and HiGHS attribute names — one will succeed, other silently fails
                try; set_attribute(planning_problem, "Crossover", 1);      catch; end
                try; set_attribute(planning_problem, "run_crossover", "on"); catch; end
                TrueSystemCost = Inf
                TrueSystemCost_new = Inf
                indicator = 1
            else
                try; set_attribute(planning_problem, "Crossover", 0);       catch; end
                try; set_attribute(planning_problem, "run_crossover", "off"); catch; end
                total_master_time = sum(master_times)
                total_sub_time = sum(sub_times)
                @info("MGA iteration $mga_it finished. Total planning time = $(tidy_timing(total_master_time)), Total subproblem time = $(tidy_timing(total_sub_time))")
                return (planning_problem=planning_problem, planning_sol=planning_sol_final, subop_sol=subop_sol, ApproxSystemCost_hist=ApproxSystemCost_hist, TrueSystemCost_hist=TrueSystemCost_hist, cpu_time=cpu_time)
            end
        elseif cpu_time[end] >= MaxCpuTime
            @info("*** MGA iteration $mga_it hit CPU time limit (MaxCpuTime=$MaxCpuTime) ***")
            return (planning_problem=planning_problem, planning_sol=planning_sol_final, subop_sol=subop_sol, ApproxSystemCost_hist=ApproxSystemCost_hist, TrueSystemCost_hist=TrueSystemCost_hist, cpu_time=cpu_time)
        else
            @info("Updating the planning problem...")
            time_start_update = time()
            update_planning_problem_multi_cuts_mga!(planning_problem, subop_sol, planning_sol, linking_variables_sub, mga_it, k)
            time_master_update = time() - time_start_update
            @info("Done updating the planning problem (it took $(tidy_timing(time_master_update)) seconds).")
            push!(master_times, cpu_planning_sol + time_master_update)
        end
    end

    @info("*** MGA iteration $mga_it reached MaxIter=$MaxIter without converging within budget ***")
    return (planning_problem=planning_problem, planning_sol=planning_sol_final, subop_sol=subop_sol, ApproxSystemCost_hist=ApproxSystemCost_hist, TrueSystemCost_hist=TrueSystemCost_hist, cpu_time=cpu_time)
end

function benders_mga(planning_problem::Model, subproblems::Union{Vector{Dict{Any,Any}}, DistributedArrays.DArray}, linking_variables_sub::Dict, setup::Dict, benders_result, variables::Vector{String})
    # Required setup keys (in addition to existing Benders keys):
    #   :MGASlack              — budget slack fraction e.g. 0.05 for 5% above optimal
    #   :MGAIterations         — number of MGA iterations to run
    #   :MGARetainBendersCuts  — cut retention method: 1=keep all, 2=keep opt only, 3=fixed early, 4=early+recent
    #   :MGAMaxCuts            — max cuts to retain (used by methods 3 and 4)
    #   :MGAMethod             — vector generation method (1=random, 2=capMM, 0=combo, 3=custom)
    #   :MGARandomSeed         — random seed for reproducibility
    #   :MGAVectorSortMethod   — "angle", "nearest-neighbor", or omit for no sorting
    #   :RelaxBudget           — optional tolerance for budget convergence check (e.g. 1e-4)

    setup[:MGABudget] = benders_result.UB_hist[end] * (1 + setup[:MGASlack])

    setup_mga_master_problem!(planning_problem, setup)
    name_cuts!(planning_problem, 0)

    opt_cuts = name.(all_constraints(planning_problem, include_variable_in_set_constraints=false))

    Iterations = setup[:MGAIterations]
    retain_master_cuts = setup[:MGARetainBendersCuts]

    vectors = generate_vecs(setup, variables)
    results = Vector{Any}(undef, Iterations)

    for iteration in 1:Iterations
        @info("Starting MGA iteration $iteration of $Iterations")

        if retain_master_cuts == 1
            # keep all cuts
        elseif retain_master_cuts == 2
            forget_cuts_master!(planning_problem, opt_cuts)
        elseif retain_master_cuts == 3
            cuts = retain_fixed_spcuts_early(planning_problem, setup[:MGAMaxCuts], iteration)
            forget_cuts_master!(planning_problem, cuts)
        elseif retain_master_cuts == 4
            cuts = retain_early_cuts_latest_iterations(planning_problem, setup[:MGAMaxCuts], iteration)
            forget_cuts_master!(planning_problem, cuts)
        else
            @info("No valid cut-retention method specified (MGARetainBendersCuts=$(retain_master_cuts)), defaulting to keeping original optimal solve cuts only")
            forget_cuts_master!(planning_problem, opt_cuts)
        end

        # Minimizing a negative weight effectively maximizes that variable,
        # letting one objective explore opposite directions
        @objective(planning_problem, Min, sum(variable_by_name(planning_problem, variables[i]) * vectors[i, iteration] for i in eachindex(variables)))

        mga_result = mga_cutting_plane(planning_problem, subproblems, linking_variables_sub, setup, iteration)
        results[iteration] = (planning_sol=mga_result.planning_sol, subop_sol=mga_result.subop_sol)
    end

    return results, vectors, variables
end
