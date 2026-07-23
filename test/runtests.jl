using MacroEnergySolvers
using Test
using Aqua
using JuMP
using LinearAlgebra
using Serialization

@testset "MacroEnergySolvers.jl" begin
    @testset "MacroEnergySolvers.jl" begin
        @testset "MGA vectors and validation" begin
            setup = Dict(
                :MGAIterations => 7,
                :MGAMethod => 2,
                :MGAComboRatio => 0.25,
                :MGARandomSeed => 19,
                :MGAVectorSortMethod => "none",
            )
            variables = ["vMGA[1]", "vMGA[2]"]
            vecs = MacroEnergySolvers.generate_vecs(setup, variables)
            @test size(vecs) == (2, 7)
            @test all(isapprox(norm(vecs[:, i]), 1.0; atol=1e-12) for i in axes(vecs, 2))
            @test vecs[:, 5] ≈ -vecs[:, 1]

            custom = copy(setup)
            custom[:MGAMethod] = 3
            custom[:MGAUserVecs] = zeros(2, 7)
            @test_throws ArgumentError MacroEnergySolvers.generate_vecs(custom, variables)

            model = Model()
            @variable(model, vREF)
            @variable(model, vTHETA[1:2])
            @variable(model, vMGA[1:2])
            @test_throws ArgumentError MacroEnergySolvers.validate_mga_variables(
                model,
                ["vREF", "vTHETA[1]"],
                Dict{Symbol,Any}(),
            )
            @test MacroEnergySolvers.validate_mga_variables(
                model,
                ["vMGA[1]", "vMGA[2]"],
                Dict{Symbol,Any}(),
            ) == ["vMGA[1]", "vMGA[2]"]
        end

        @testset "MGA budget and durable checkpoint" begin
            model = Model()
            @variable(model, x >= 0)
            @variable(model, vTHETA[1:2] >= 0)
            @expression(model, ePlanningCost, x)
            setup = Dict{Symbol,Any}(:MGABudget => 10.0)

            first_budget = MacroEnergySolvers.setup_mga_master_problem!(model, setup)
            setup[:MGABudget] = 12.0
            second_budget = MacroEnergySolvers.setup_mga_master_problem!(model, setup)
            @test !is_valid(model, first_budget)
            @test is_valid(model, second_budget)
            @test length(all_constraints(model; include_variable_in_set_constraints=false)) == 1

            result = (
                status=:converged,
                converged=true,
                iterations=2,
                planning_sol=(planning_cost=1.0, values=Dict("x" => 1.0)),
                subop_sol=Dict(1 => (op_cost=2.0, lambda=[0.0], theta_coeff=1)),
                ApproxSystemCost_hist=[3.0],
                TrueSystemCost_hist=[3.0],
                cpu_time=[0.1],
                true_system_cost=3.0,
                best_true_system_cost=3.0,
                budget_violation=-0.75,
            )
            mktempdir() do checkpoint_dir
                path = MacroEnergySolvers.write_mga_checkpoint(
                    checkpoint_dir,
                    1,
                    result,
                    [1.0, 0.0],
                    ["vMGA[1]", "vMGA[2]"],
                    setup,
                )
                @test isfile(path)
                @test isfile(joinpath(checkpoint_dir, "mga_iteration_0001_summary.tsv"))
                payload = open(deserialize, path)
                @test payload.schema_version == 1
                @test payload.result.status == :converged
                @test payload.variables == ["vMGA[1]", "vMGA[2]"]
            end
        end
    end
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MacroEnergySolvers)
    end
end
