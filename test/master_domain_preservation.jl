@testset "Benders preserves master domains and solved policy allocations" begin
    for allocations in ([-1.0, 5.0], [1.0, 3.0])
        master = Model(HiGHS.Optimizer)
        set_silent(master)
        @variable(master, capacity >= 0)
        @variable(master, vPolicy_Budget_electricity[1:2])
        @constraint(master, sum(vPolicy_Budget_electricity) == 4)
        @constraint(master, vPolicy_Budget_electricity[1] == allocations[1])
        # A new upstream model may contain unnamed helper variables.
        auxiliary = @variable(master, lower_bound = 0)
        @constraint(master, auxiliary >= capacity)
        master[:eAvailableCapacity] = [capacity]
        @objective(master, Min, 1 + capacity)
        names = name.(vPolicy_Budget_electricity)
        subproblems = Dict{Any,Any}[]
        linking = Dict(w => [names[w]] for w in 1:2)
        for w in 1:2
            sub = Model(HiGHS.Optimizer)
            set_silent(sub)
            budget = @variable(sub, base_name = names[w])
            @variable(sub, operation >= 0)
            @constraint(sub, operation >= 3 + budget)
            @objective(sub, Min, operation)
            push!(subproblems, Dict{Any,Any}(
                :model => sub, :linking_variables_sub => linking[w], :subproblem_index => w,
            ))
        end
        setup = Dict{Symbol,Any}(
            :ExpectFeasibleSubproblems => true, :MaxIter => 2, :ConvTol => 1e-8,
            :MaxCpuTime => 60, :StabParam => 0.0, :StabDynamic => false,
            :IntegerInvestment => false,
        )
        result = MES.benders(master, subproblems, linking, setup)
        @test result.termination_status == "OPTIMAL"
        @test result.UB_hist[end] ≈ 11.0
        @test result.LB_hist[end] ≈ 11.0
        @test [result.planning_sol.values[n] for n in names] ≈ allocations
        @test all(!has_lower_bound(v) for v in vPolicy_Budget_electricity)
        @test all(!has_upper_bound(v) for v in vPolicy_Budget_electricity)
        @test !haskey(result.planning_sol.values, "")
        @test value(auxiliary) >= 0

        mga_setup = merge(setup, Dict{Symbol,Any}(
            :MGASlack => 0.05, :MGAIterations => 1, :MGAMethod => 3,
            :MGAUserVecs => -ones(1, 1), :MGARetainBendersCuts => 2,
            :MGAMaxIter => 5, :MGARelaxBudget => 1e-7,
        ))
        mktempdir() do directory
            directions, _, selected = MES.benders_mga(
                master, subproblems, linking, mga_setup, (convergence=result,),
                ["capacity"]; checkpoint_dir=directory,
            )
            @test only(directions).converged
            @test only(directions).planning_sol.values["capacity"] ≈ 0.55
            @test only(directions).true_system_cost ≈ 11.55
            @test only(directions).budget_violation <= 1e-7
            @test selected == ["capacity"]
            @test !haskey(master, :cMGABudget)
            payload = deserialize(joinpath(directory, "mga_iteration_0001.jls"))
            @test payload.schema_version == 2
            @test payload.result.subop_sol === nothing
        end
        optimize!(master)
        @test objective_value(master) ≈ 11.0
    end
end
