"""
    solve_int_level_set_problem(m::Model, 
		planning_variables::Vector{String}, 
		planning_sol::NamedTuple, 
		LB, 
		UB, 
		γ;
		incumbent_sol::Union{Nothing,NamedTuple}=nothing,
		proximal::Bool=false
	)

Solves the interior level set stabilization problem for the regularized Benders decomposition algorithm.

This stabilization technique helps improve convergence by restricting the upper-level problem solution
to lie within a level set defined by the current lower and upper bounds, controlled by the
stabilization parameter γ.

# Arguments
- `m::Model`: The JuMP model representing the planning problem
- `planning_variables::Vector{String}`: Names of the variables of the planning problem
- `planning_sol::NamedTuple`: Current solution of the planning problem
- `LB`: Current lower bound
- `UB`: Current upper bound
- `γ`: Stabilization parameter controlling the size of the level set (0 ≤ γ ≤ 1)
- `incumbent_sol`: Optional feasible incumbent used as the center of the proximal diagnostic
- `proximal`: If true, minimize scaled squared distance from `incumbent_sol` instead of
  using the legacy zero objective

# Returns
A NamedTuple containing the solution of the stabilized problem with the same structure as the input `planning_sol`

"""
_level_set_proximal_scale(incumbent_value, raw_value) =
	max(1.0, abs(Float64(incumbent_value)), abs(Float64(raw_value)))

function _feasibility_proximal_metric(
	m::Model,
	planning_variables::Vector{String},
	raw_planning_sol::NamedTuple,
	center_sol::NamedTuple;
	include_raw_in_scale::Bool=true,
	fixed_proximal_scales::Union{Nothing,AbstractDict}=nothing,
)
	proximal_variables = String[]
	proximal_scales = Dict{String,Float64}()

	for variable_name in planning_variables
		startswith(variable_name, "vTHETA") && continue
		haskey(raw_planning_sol.values, variable_name) || error(
			"FEASIBILITY_PROXIMAL: raw planning solution is missing $variable_name",
		)
		haskey(center_sol.values, variable_name) || error(
			"FEASIBILITY_PROXIMAL: center solution is missing $variable_name",
		)
		isnothing(variable_by_name(m, variable_name)) && error(
			"FEASIBILITY_PROXIMAL: planning model is missing $variable_name",
		)
		push!(proximal_variables, variable_name)
		if isnothing(fixed_proximal_scales)
			proximal_scales[variable_name] = include_raw_in_scale ?
				_level_set_proximal_scale(
					center_sol.values[variable_name],
					raw_planning_sol.values[variable_name],
				) : max(1.0, abs(Float64(center_sol.values[variable_name])))
		else
			haskey(fixed_proximal_scales, variable_name) || error(
				"FEASIBILITY_PROXIMAL: fixed scale is missing $variable_name",
			)
			scale = Float64(fixed_proximal_scales[variable_name])
			isfinite(scale) && scale > 0.0 || error(
				"FEASIBILITY_PROXIMAL: fixed scale for $variable_name must be finite and positive",
			)
			proximal_scales[variable_name] = scale
		end
	end
	isempty(proximal_variables) && error(
		"FEASIBILITY_PROXIMAL: no non-vTHETA planning variables were found",
	)

	return (
		variables=proximal_variables,
		scales=proximal_scales,
	)
end

"""
    solve_feasibility_proximal_problem(
        m,
        planning_variables,
        raw_planning_sol,
        center_sol,
    )

Projects an infeasible Benders evaluation point onto the planning master after
new feasibility cuts have been added. The projection minimizes a scaled
squared distance to `center_sol` and intentionally does not replace the
cost-optimal master solve used to compute the lower bound.

This is useful before a finite upper bound exists, when the ordinary level-set
stabilization is undefined and the unregularized master otherwise jumps among
cost-optimal extreme points.
"""
function solve_feasibility_proximal_problem(
	m::Model,
	planning_variables::Vector{String},
	raw_planning_sol::NamedTuple,
	center_sol::NamedTuple,
	;
	master_objective_level::Union{Nothing,Float64}=nothing,
	include_raw_in_scale::Bool=true,
	fixed_proximal_scales::Union{Nothing,AbstractDict}=nothing,
)
	original_objective = objective_function(m)
	level_constraint = nothing
	proximal_metric = _feasibility_proximal_metric(
		m,
		planning_variables,
		raw_planning_sol,
		center_sol;
		include_raw_in_scale=include_raw_in_scale,
		fixed_proximal_scales=fixed_proximal_scales,
	)
	proximal_variables = proximal_metric.variables
	proximal_scales = proximal_metric.scales

	projected_sol = nothing
	try
		if !isnothing(master_objective_level)
			level_constraint = @constraint(
				m,
				original_objective <= master_objective_level,
			)
		end
		@objective(
			m,
			Min,
			sum(
				((variable_by_name(m, variable_name) - center_sol.values[variable_name]) /
				 proximal_scales[variable_name])^2
				for variable_name in proximal_variables
			),
		)
		optimize!(m)
		has_values(m) || error(
			"FEASIBILITY_PROXIMAL solve failed: termination=$(termination_status(m)) " *
			"primal=$(primal_status(m)) raw=$(repr(raw_status(m)))",
		)

		planning_cost, variable_values = process_planning_sol(m, planning_variables)
		projected_sol = (
			planning_cost=planning_cost,
			values=variable_values,
		)
		normalized_squared_distance = sum(
			((projected_sol.values[variable_name] - center_sol.values[variable_name]) /
			 proximal_scales[variable_name])^2
			for variable_name in proximal_variables
		)
		max_abs_center_difference = maximum(
			abs(projected_sol.values[variable_name] - center_sol.values[variable_name])
			for variable_name in proximal_variables
		)
		max_abs_raw_difference = maximum(
			abs(projected_sol.values[variable_name] - raw_planning_sol.values[variable_name])
			for variable_name in proximal_variables
		)
		projected_master_objective = value(original_objective)
		@info "FEASIBILITY_PROXIMAL_SOLVED: variables=$(length(proximal_variables)) normalized_squared_distance=$(normalized_squared_distance) max_abs_center_difference=$(max_abs_center_difference) max_abs_raw_difference=$(max_abs_raw_difference) planning_cost=$(planning_cost) master_objective=$(projected_master_objective) master_objective_level=$(master_objective_level) include_raw_in_scale=$(include_raw_in_scale) fixed_metric=$(!isnothing(fixed_proximal_scales))"
		flush(stdout)
		flush(stderr)
	finally
		@objective(m, Min, original_objective)
		!isnothing(level_constraint) && delete(m, level_constraint)
	end

	return projected_sol
end

function solve_int_level_set_problem(
	m::Model,
	planning_variables::Vector{String},
	planning_sol::NamedTuple,
	LB,
	UB,
	γ;
	incumbent_sol::Union{Nothing,NamedTuple}=nothing,
	proximal::Bool=false,
)
	
	### Interior point regularization based on https://ieeexplore.ieee.org/document/10829583

	objfun = objective_function(m)

	@constraint(m,cLevel_set, objfun <=LB+γ*(UB-LB))

	proximal_variables = String[]
	proximal_scales = Dict{String,Float64}()
	if proximal
		isnothing(incumbent_sol) && error("BENDERS_LEVELSET_PROXIMAL=true requires a finite incumbent solution")
		for variable_name in planning_variables
			startswith(variable_name, "vTHETA") && continue
			haskey(planning_sol.values, variable_name) || error("LEVELSET_PROXIMAL: raw planning solution is missing $variable_name")
			haskey(incumbent_sol.values, variable_name) || error("LEVELSET_PROXIMAL: incumbent solution is missing $variable_name")
			isnothing(variable_by_name(m, variable_name)) && error("LEVELSET_PROXIMAL: planning model is missing $variable_name")
			push!(proximal_variables, variable_name)
			proximal_scales[variable_name] = _level_set_proximal_scale(
				incumbent_sol.values[variable_name],
				planning_sol.values[variable_name],
			)
		end
		isempty(proximal_variables) && error("LEVELSET_PROXIMAL: no non-vTHETA planning variables were found")
		@objective(
			m,
			Min,
			sum(
				((variable_by_name(m, variable_name) - incumbent_sol.values[variable_name]) /
				 proximal_scales[variable_name])^2
				for variable_name in proximal_variables
			),
		)
	else
		@objective(m, Min, 0*sum(m[:vTHETA][1]))
	end

    optimize!(m)

	if has_values(m)

		planning_cost,variable_values = process_planning_sol(m,planning_variables)

		planning_sol = (;planning_sol..., planning_cost = planning_cost, values = variable_values)

		if proximal
			normalized_squared_distance = sum(
				((planning_sol.values[variable_name] - incumbent_sol.values[variable_name]) /
				 proximal_scales[variable_name])^2
				for variable_name in proximal_variables
			)
			max_abs_incumbent_difference = maximum(
				abs(planning_sol.values[variable_name] - incumbent_sol.values[variable_name])
				for variable_name in proximal_variables
			)
			@info "LEVELSET_PROXIMAL_SOLVED: variables=$(length(proximal_variables)) normalized_squared_distance=$(normalized_squared_distance) max_abs_incumbent_difference=$(max_abs_incumbent_difference)"
		end
		
	else

		@warn "the interior level set problem solution failed" termination_status=termination_status(m) primal_status=primal_status(m) raw_status=raw_status(m)

	end

	delete(m,m[:cLevel_set])
	unregister(m,:cLevel_set)
	@objective(m,Min, objfun)
	
	return planning_sol

end
