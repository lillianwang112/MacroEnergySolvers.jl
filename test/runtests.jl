using MacroEnergySolvers
using Test
using Aqua
using JuMP

@testset "MacroEnergySolvers.jl" begin
    @testset "MacroEnergySolvers.jl" begin
		@test MacroEnergySolvers._level_set_proximal_scale(0.0, 0.0) == 1.0
		@test MacroEnergySolvers._level_set_proximal_scale(10.0, 2.0) == 10.0
		@test MacroEnergySolvers._level_set_proximal_scale(-3.0, -12.0) == 12.0
    end
    @testset "Oracle seed respects the planning domain" begin
        model = Model()
        @variable(model, lower_bounded >= 0.0)
        @variable(model, upper_bounded <= 2.0)
        @variable(model, free_variable)
        @variable(model, fixed_variable)
        fix(fixed_variable, 1.5; force=true)

        variables = [lower_bounded, upper_bounded, free_variable, fixed_variable]
        names = name.(variables)
        invalid_values = Dict(
            name(lower_bounded) => -0.1,
            name(upper_bounded) => 2.1,
            name(free_variable) => -10.0,
            name(fixed_variable) => 1.6,
        )

        violations = MacroEnergySolvers._oracle_seed_bound_violations(
            variables,
            names,
            invalid_values,
        )
        @test Set((violation.name, violation.kind) for violation in violations) == Set([
            (name(lower_bounded), :lower),
            (name(upper_bounded), :upper),
            (name(fixed_variable), :fixed),
        ])

        valid_values = Dict(
            name(lower_bounded) => 0.0,
            name(upper_bounded) => 2.0,
            name(free_variable) => -10.0,
            name(fixed_variable) => 1.5,
        )
        @test isempty(MacroEnergySolvers._oracle_seed_bound_violations(
            variables,
            names,
            valid_values,
        ))
    end
    @testset "Signed linking variables keep their model domains" begin
        @test !MacroEnergySolvers._infer_nonnegative_linking_bound(
            "vCO2CapConstraint_Budget_CO2_period1[7]",
        )
        @test MacroEnergySolvers._infer_nonnegative_linking_bound(
            "vCO2StorageConstraint_Budget_co2_storage_SE_1_period1[18]",
        )
        @test !MacroEnergySolvers._infer_nonnegative_linking_bound(
            "vSTOR_CHANGE_SE_Above_ground_storage_period1[7]",
        )
        @test MacroEnergySolvers._infer_nonnegative_linking_bound(
            "vCAP_SE_solar_photovoltaic_1_period1",
        )
    end
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MacroEnergySolvers)
    end
end
