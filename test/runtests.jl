using MacroEnergySolvers
using Test
using Aqua

@testset "MacroEnergySolvers.jl" begin
    @testset "MacroEnergySolvers.jl" begin
		@test MacroEnergySolvers._level_set_proximal_scale(0.0, 0.0) == 1.0
		@test MacroEnergySolvers._level_set_proximal_scale(10.0, 2.0) == 10.0
		@test MacroEnergySolvers._level_set_proximal_scale(-3.0, -12.0) == 12.0
    end
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MacroEnergySolvers)
    end
end
