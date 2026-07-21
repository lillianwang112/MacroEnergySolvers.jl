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
    @testset "Multisector biomass master strengthening" begin
        model = Model()
        variables = Dict{String,VariableRef}()
        for group in MacroEnergySolvers._MULTISECTOR_BIOMASS_SUPPLY_GROUPS
            for (variable_name, _) in group.coefficients
                variables[variable_name] = @variable(model, base_name=variable_name)
            end
        end

        constraints = MacroEnergySolvers._add_multisector_biomass_master_strengthening!(
            model,
        )
        @test length(constraints) == 6
        for (constraint, group) in zip(
            constraints,
            MacroEnergySolvers._MULTISECTOR_BIOMASS_SUPPLY_GROUPS,
        )
            @test name(constraint) ==
                "BendersMasterStrengthening_$(group.node)_period1"
            constraint_data = constraint_object(constraint)
            @test constraint_data.set.upper == group.limit
            @test JuMP.constant(constraint_data.func) == 0.0
            for (variable_name, coefficient) in group.coefficients
                @test JuMP.coefficient(
                    constraint_data.func,
                    variables[variable_name],
                ) == coefficient
            end
        end

        # The exact single-technology ray from the Della diagnostic is cut
        # off, while the known monolithic capacity remains feasible.
        ne_ft = variables["vCAP_NE_BECCS_FT_Herb_biomass_edge_period1"]
        ne_herb_constraint = constraint_object(constraint_by_name(
            model,
            "BendersMasterStrengthening_bioherb_NE_period1",
        ))
        @test 0.85 * 697.788922721467 > 297.76
        @test 0.85 * 330.84444 < 297.76
        @test JuMP.coefficient(ne_herb_constraint.func, ne_ft) == 0.85

        # The two wood-capacity combinations reported by the strict final IIS
        # in job 11429292 also violate their corresponding aggregate limits.
        @test 11357.589421229883 > 11160.76
        @test 2015.0486799642754 > 1905.74

        @test_throws ErrorException MacroEnergySolvers._add_multisector_biomass_master_strengthening!(
            model,
        )

        incomplete_model = Model()
        first_name = first(first(
            MacroEnergySolvers._MULTISECTOR_BIOMASS_SUPPLY_GROUPS,
        ).coefficients)[1]
        @variable(incomplete_model, base_name=first_name)
        @test_throws ErrorException MacroEnergySolvers._add_multisector_biomass_master_strengthening!(
            incomplete_model,
        )
    end
    @testset "Feasibility-cut checkpoints round-trip and replay" begin
        checkpoint_directory = mktempdir()
        cut = (
            w=2,
            lambda=[2.0, -1.0],
            linking_vars=["x", "y"],
            op_cost=5.0,
            alpha=3.0,
            x_generated=[3.0, 4.0],
            generating_residual=5.0,
            k_added=7,
            cut_source=:phase1,
        )

        checkpoint_path = MacroEnergySolvers._write_feasibility_cut_checkpoint(
            checkpoint_directory,
            1,
            cut,
        )
        @test isfile(checkpoint_path)
        loaded_cuts = MacroEnergySolvers._load_feasibility_cut_checkpoints(
            checkpoint_directory,
        )
        @test length(loaded_cuts) == 1
        loaded = only(loaded_cuts)
        @test loaded.checkpoint_id == 1
        @test loaded.w == "2"
        @test loaded.k_added == 7
        @test loaded.cut_source == :phase1
        @test loaded.alpha == 3.0
        @test loaded.op_cost == 5.0
        @test loaded.generating_residual == 5.0
        @test loaded.linking_vars == ["x", "y"]
        @test loaded.lambda == [2.0, -1.0]
        @test loaded.x_generated == [3.0, 4.0]

        # Incomplete temporary files are intentionally invisible to replay.
        write(joinpath(checkpoint_directory, ".feasibility_cut_00000002.tsv.tmp"), "partial")
        @test length(MacroEnergySolvers._load_feasibility_cut_checkpoints(
            checkpoint_directory,
        )) == 1

        model = Model()
        @variable(model, x, base_name="x")
        @variable(model, y, base_name="y")
        constraints = MacroEnergySolvers._add_replayed_feasibility_cuts!(
            model,
            loaded_cuts,
        )
        @test length(constraints) == 1
        constraint = only(constraints)
        @test name(constraint) == "BendersReplayFeasibilityCut_00000001"
        constraint_data = constraint_object(constraint)
        # JuMP normalizes the affine constant into the LessThan upper bound.
        @test JuMP.constant(constraint_data.func) == 0.0
        @test JuMP.coefficient(constraint_data.func, x) == 2.0
        @test JuMP.coefficient(constraint_data.func, y) == -1.0
        @test constraint_data.set.upper == -3.0

        state_path = MacroEnergySolvers._write_feasibility_checkpoint_state(
            checkpoint_directory,
            1,
            12.5,
        )
        @test isfile(state_path)
        @test MacroEnergySolvers._read_feasibility_checkpoint_state(
            checkpoint_directory,
        ) == (cut_count=1, master_objective=12.5)
        MacroEnergySolvers._write_feasibility_checkpoint_state(
            checkpoint_directory,
            2,
            13.5,
        )
        @test MacroEnergySolvers._read_feasibility_checkpoint_state(
            checkpoint_directory,
        ) == (cut_count=2, master_objective=13.5)

        center_sol = (
            planning_cost=99.0,
            values=Dict(
                "x" => 1.25,
                "y" => -2.5,
                "vTHETA[1]" => 0.0,
            ),
        )
        center_path = MacroEnergySolvers._write_feasibility_proximal_center(
            checkpoint_directory,
            2,
            center_sol,
            ["x", "y", "vTHETA[1]"],
        )
        @test isfile(center_path)
        loaded_center = MacroEnergySolvers._read_feasibility_proximal_center(
            checkpoint_directory,
        )
        @test loaded_center.cut_count == 2
        @test loaded_center.planning_cost == 99.0
        @test loaded_center.values == Dict("x" => 1.25, "y" => -2.5)

        raw_center = (
            planning_cost=12.0,
            values=Dict("x" => 0.0, "y" => 0.0, "z" => 9.0),
        )
        older_high_k_cut = merge(loaded, (checkpoint_id=0, k_added=99))
        recovered_center = MacroEnergySolvers._recover_latest_feasibility_generating_point(
            [older_high_k_cut, loaded],
            raw_center,
        )
        @test recovered_center.latest_iteration == 7
        @test recovered_center.recovered_variables == 2
        @test recovered_center.values == Dict("x" => 3.0, "y" => 4.0, "z" => 9.0)
    end
    @testset "Feasibility-cut modes and duplicate suppression" begin
        original_mode = get(ENV, "BENDERS_FEASIBILITY_CUT_MODE", nothing)
        try
            delete!(ENV, "BENDERS_FEASIBILITY_CUT_MODE")
            @test MacroEnergySolvers._feasibility_cut_mode() == :auto
            for mode in ("auto", "farkas", "phase1")
                ENV["BENDERS_FEASIBILITY_CUT_MODE"] = mode
                @test MacroEnergySolvers._feasibility_cut_mode() == Symbol(mode)
            end
            ENV["BENDERS_FEASIBILITY_CUT_MODE"] = "invalid"
            @test_throws ErrorException MacroEnergySolvers._feasibility_cut_mode()
        finally
            if isnothing(original_mode)
                delete!(ENV, "BENDERS_FEASIBILITY_CUT_MODE")
            else
                ENV["BENDERS_FEASIBILITY_CUT_MODE"] = original_mode
            end
        end

        signature = MacroEnergySolvers._canonical_feasibility_cut_signature
        @test signature(-1.0, [1.0, 0.0], ["x", "y"]) ==
            signature(-2.0, [2.0, 0.0], ["x", "y"])
        @test signature(-1.0, [1.0, 0.0], ["x", "y"]) !=
            signature(-1.0, [0.0, 1.0], ["x", "y"])
        @test_throws ErrorException signature(0.0, [0.0], ["x"])

        planning_sol = (
            planning_cost=0.0,
            values=Dict("x" => 2.0, "y" => 3.0),
        )
        subop_sol = Dict(
            1 => (op_cost=1.0, lambda=[1.0], theta_coeff=0, cut_source=:farkas),
            2 => (op_cost=1.0, lambda=[1.0], theta_coeff=0, cut_source=:farkas),
            3 => (op_cost=2.0, lambda=[1.0], theta_coeff=0, cut_source=:phase1),
        )
        linking_variables_sub = Dict(1 => ["x"], 2 => ["x"], 3 => ["y"])
        seen = Set{String}()
        selection = MacroEnergySolvers._select_new_master_cuts!(
            seen,
            subop_sol,
            planning_sol,
            linking_variables_sub,
            4,
        )
        @test Set(keys(selection.selected_subop_sol)) == Set([1, 3])
        @test length(selection.accepted_feasibility_cuts) == 2
        @test length(selection.duplicate_feasibility_cuts) == 1
        @test only(selection.duplicate_feasibility_cuts).w == 2
        @test length(seen) == 2
    end
    @testset "Checkpoint configuration is explicit" begin
        environment_names = (
            "BENDERS_FEASIBILITY_CUT_CHECKPOINT_DIR",
            "BENDERS_FEASIBILITY_CUT_REPLAY",
            "BENDERS_FEASIBILITY_CUT_WRITE",
        )
        original_values = Dict(name => get(ENV, name, nothing) for name in environment_names)
        try
            delete!(ENV, "BENDERS_FEASIBILITY_CUT_CHECKPOINT_DIR")
            ENV["BENDERS_FEASIBILITY_CUT_REPLAY"] = "true"
            ENV["BENDERS_FEASIBILITY_CUT_WRITE"] = "false"
            @test_throws ErrorException MacroEnergySolvers._feasibility_cut_checkpoint_config()

            ENV["BENDERS_FEASIBILITY_CUT_CHECKPOINT_DIR"] = "/tmp/checkpoints"
            ENV["BENDERS_FEASIBILITY_CUT_REPLAY"] = "false"
            ENV["BENDERS_FEASIBILITY_CUT_WRITE"] = "true"
            @test MacroEnergySolvers._feasibility_cut_checkpoint_config() == (
                directory="/tmp/checkpoints",
                replay=false,
                write=true,
            )
        finally
            for (name, value) in original_values
                if isnothing(value)
                    delete!(ENV, name)
                else
                    ENV[name] = value
                end
            end
        end
    end
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(MacroEnergySolvers)
    end
end
