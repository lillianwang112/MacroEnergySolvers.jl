const _FEASIBILITY_CUT_CHECKPOINT_VERSION = 1
const _FEASIBILITY_CUT_CHECKPOINT_HEADER =
    "MacroEnergySolvers feasibility cut checkpoint\t$(_FEASIBILITY_CUT_CHECKPOINT_VERSION)"

function _checkpoint_env_flag(name::String, default::Bool=false)
    raw_value = lowercase(strip(get(ENV, name, string(default))))
    raw_value in ("1", "true", "yes", "on") && return true
    raw_value in ("0", "false", "no", "off") && return false
    error("$name must be a boolean value; received $(repr(raw_value))")
end

function _feasibility_cut_mode()
    raw_value = lowercase(strip(get(ENV, "BENDERS_FEASIBILITY_CUT_MODE", "auto")))
    raw_value in ("auto", "farkas", "phase1") || error(
        "BENDERS_FEASIBILITY_CUT_MODE must be auto, farkas, or phase1; " *
        "received $(repr(raw_value))",
    )
    return Symbol(raw_value)
end

function _canonical_feasibility_cut_signature(
    alpha::Real,
    lambda::AbstractVector{<:Real},
    linking_vars::AbstractVector{<:AbstractString};
    coefficient_tolerance::Real=1e-10,
    digits::Integer=10,
)
    length(lambda) == length(linking_vars) || error(
        "Cannot canonicalize a feasibility cut with mismatched coefficients and variables",
    )
    coefficients = Float64[Float64(alpha)]
    append!(coefficients, Float64.(lambda))
    all(isfinite, coefficients) || error(
        "Cannot canonicalize a feasibility cut containing non-finite coefficients",
    )
    scale = maximum(abs, coefficients; init=0.0)
    scale > coefficient_tolerance || error(
        "Cannot canonicalize a feasibility cut whose coefficients are all numerically zero",
    )

    normalized_alpha = round(Float64(alpha) / scale; digits=digits)
    normalized_terms = Tuple{String,Float64}[]
    for (variable_name, coefficient) in zip(linking_vars, lambda)
        abs(coefficient) <= coefficient_tolerance && continue
        push!(
            normalized_terms,
            (String(variable_name), round(Float64(coefficient) / scale; digits=digits)),
        )
    end
    sort!(normalized_terms; by=first)
    return string(normalized_alpha, '|', join(("$(name)=$(value)" for (name, value) in normalized_terms), '|'))
end

function _select_new_master_cuts!(
    seen_feasibility_cut_signatures::Set{String},
    subop_sol::AbstractDict,
    planning_sol::NamedTuple,
    linking_variables_sub::AbstractDict,
    k::Integer,
)
    selected_subop_sol = Dict{Any,Any}()
    accepted_feasibility_cuts = NamedTuple[]
    duplicate_feasibility_cuts = NamedTuple[]

    for w in sort!(collect(keys(subop_sol)); by=string)
        sol = subop_sol[w]
        if sol.theta_coeff != 0
            selected_subop_sol[w] = sol
            continue
        end

        linking_vars = copy(linking_variables_sub[w])
        x_generated = [planning_sol.values[v] for v in linking_vars]
        alpha = sol.op_cost - dot(sol.lambda, x_generated)
        generating_residual = alpha + dot(sol.lambda, x_generated)
        cut_source = hasproperty(sol, :cut_source) ? sol.cut_source : :unknown
        cut = (
            w=w,
            lambda=copy(sol.lambda),
            linking_vars=linking_vars,
            op_cost=sol.op_cost,
            alpha=alpha,
            x_generated=x_generated,
            generating_residual=generating_residual,
            k_added=Int(k),
            cut_source=cut_source,
        )
        signature = _canonical_feasibility_cut_signature(
            cut.alpha,
            cut.lambda,
            cut.linking_vars,
        )
        if signature in seen_feasibility_cut_signatures
            push!(duplicate_feasibility_cuts, merge(cut, (signature=signature,)))
            continue
        end

        push!(seen_feasibility_cut_signatures, signature)
        selected_subop_sol[w] = sol
        push!(accepted_feasibility_cuts, merge(cut, (signature=signature,)))
    end

    return (
        selected_subop_sol=selected_subop_sol,
        accepted_feasibility_cuts=accepted_feasibility_cuts,
        duplicate_feasibility_cuts=duplicate_feasibility_cuts,
    )
end

function _feasibility_cut_checkpoint_config()
    directory = strip(get(ENV, "BENDERS_FEASIBILITY_CUT_CHECKPOINT_DIR", ""))
    replay = _checkpoint_env_flag("BENDERS_FEASIBILITY_CUT_REPLAY")
    write_enabled = _checkpoint_env_flag("BENDERS_FEASIBILITY_CUT_WRITE")
    if (replay || write_enabled) && isempty(directory)
        error(
            "BENDERS_FEASIBILITY_CUT_REPLAY/WRITE requires " *
            "BENDERS_FEASIBILITY_CUT_CHECKPOINT_DIR",
        )
    end
    return (directory=directory, replay=replay, write=write_enabled)
end

_feasibility_cut_checkpoint_filename(id::Integer) =
    "feasibility_cut_$(lpad(id, 8, '0')).tsv"

function _feasibility_cut_checkpoint_files(directory::AbstractString)
    isdir(directory) || return Pair{Int,String}[]
    files = Pair{Int,String}[]
    for filename in readdir(directory)
        matched = match(r"^feasibility_cut_(\d{8})\.tsv$", filename)
        isnothing(matched) && continue
        push!(files, parse(Int, only(matched.captures)) => joinpath(directory, filename))
    end
    sort!(files; by=first)
    return files
end

function _validate_checkpoint_text(value::AbstractString, label::AbstractString)
    (occursin('\t', value) || occursin('\n', value) || occursin('\r', value)) &&
        error("Checkpoint $label contains a tab or newline: $(repr(value))")
    return String(value)
end

function _atomic_write(writer::Function, path::AbstractString)
    directory = dirname(path)
    mkpath(directory)
    temporary_path = joinpath(
        directory,
        ".$(basename(path)).tmp.$(getpid()).$(rand(UInt))",
    )
    try
        open(temporary_path, "w") do io
            writer(io)
            flush(io)
        end
        mv(temporary_path, path; force=false)
    finally
        isfile(temporary_path) && rm(temporary_path; force=true)
    end
    return path
end

function _write_feasibility_cut_checkpoint(
    directory::AbstractString,
    checkpoint_id::Integer,
    cut,
)
    checkpoint_id > 0 || error("Checkpoint id must be positive")
    n_terms = length(cut.linking_vars)
    length(cut.lambda) == n_terms || error("Checkpoint cut has mismatched lambda length")
    length(cut.x_generated) == n_terms ||
        error("Checkpoint cut has mismatched generating-point length")

    scalar_values = Float64[
        cut.alpha,
        cut.op_cost,
        cut.generating_residual,
    ]
    append!(scalar_values, Float64.(cut.lambda))
    append!(scalar_values, Float64.(cut.x_generated))
    all(isfinite, scalar_values) || error("Checkpoint cut contains a non-finite value")

    expected_alpha = cut.op_cost - dot(cut.lambda, cut.x_generated)
    isapprox(cut.alpha, expected_alpha; atol=1e-8, rtol=1e-10) || error(
        "Checkpoint cut alpha is inconsistent with op_cost - lambda' * x_generated",
    )

    path = joinpath(directory, _feasibility_cut_checkpoint_filename(checkpoint_id))
    ispath(path) && error("Refusing to overwrite existing feasibility-cut checkpoint: $path")
    _atomic_write(path) do io
        println(io, _FEASIBILITY_CUT_CHECKPOINT_HEADER)
        println(io, "checkpoint_id\t", checkpoint_id)
        println(io, "w\t", _validate_checkpoint_text(string(cut.w), "subproblem id"))
        println(io, "k_added\t", Int(cut.k_added))
        cut_source = hasproperty(cut, :cut_source) ? cut.cut_source : :unknown
        println(
            io,
            "cut_source\t",
            _validate_checkpoint_text(string(cut_source), "cut source"),
        )
        println(io, "alpha\t", repr(Float64(cut.alpha)))
        println(io, "op_cost\t", repr(Float64(cut.op_cost)))
        println(io, "generating_residual\t", repr(Float64(cut.generating_residual)))
        println(io, "n_terms\t", n_terms)
        println(io, "variable_name\tlambda\tx_generated")
        for i in eachindex(cut.linking_vars)
            variable_name = _validate_checkpoint_text(
                string(cut.linking_vars[i]),
                "variable name",
            )
            println(
                io,
                variable_name,
                '\t',
                repr(Float64(cut.lambda[i])),
                '\t',
                repr(Float64(cut.x_generated[i])),
            )
        end
    end
    return path
end

function _parse_checkpoint_metadata(lines::Vector{String}, path::AbstractString)
    length(lines) >= 9 || error("Incomplete feasibility-cut checkpoint: $path")
    lines[1] == _FEASIBILITY_CUT_CHECKPOINT_HEADER ||
        error("Unsupported feasibility-cut checkpoint header in $path")
    metadata = Dict{String,String}()
    term_header_index = findfirst(==("variable_name\tlambda\tx_generated"), lines)
    isnothing(term_header_index) && error("Missing term header in $path")
    for line in lines[2:term_header_index-1]
        fields = split(line, '\t'; limit=2)
        length(fields) == 2 || error("Malformed checkpoint metadata in $path: $line")
        haskey(metadata, fields[1]) && error("Duplicate metadata key $(fields[1]) in $path")
        metadata[fields[1]] = fields[2]
    end
    required = (
        "checkpoint_id",
        "w",
        "k_added",
        "alpha",
        "op_cost",
        "generating_residual",
        "n_terms",
    )
    missing = filter(key -> !haskey(metadata, key), required)
    isempty(missing) || error("Missing checkpoint metadata $(join(missing, ", ")) in $path")
    return metadata, term_header_index
end

function _parse_checkpoint_number(::Type{T}, raw::AbstractString, label, path) where {T<:Real}
    value = tryparse(T, raw)
    isnothing(value) && error("Invalid $label in $path: $(repr(raw))")
    isfinite(value) || error("Non-finite $label in $path")
    return value
end

function _read_feasibility_cut_checkpoint(path::AbstractString)
    lines = readlines(path)
    metadata, term_header_index = _parse_checkpoint_metadata(lines, path)
    checkpoint_id = _parse_checkpoint_number(Int, metadata["checkpoint_id"], "id", path)
    k_added = _parse_checkpoint_number(Int, metadata["k_added"], "iteration", path)
    n_terms = _parse_checkpoint_number(Int, metadata["n_terms"], "term count", path)
    alpha = _parse_checkpoint_number(Float64, metadata["alpha"], "alpha", path)
    op_cost = _parse_checkpoint_number(Float64, metadata["op_cost"], "op_cost", path)
    generating_residual = _parse_checkpoint_number(
        Float64,
        metadata["generating_residual"],
        "generating residual",
        path,
    )
    term_lines = lines[term_header_index+1:end]
    length(term_lines) == n_terms || error(
        "Checkpoint $path declares $n_terms terms but contains $(length(term_lines))",
    )

    linking_vars = String[]
    lambda = Float64[]
    x_generated = Float64[]
    for line in term_lines
        fields = split(line, '\t'; keepempty=true)
        length(fields) == 3 || error("Malformed checkpoint term in $path: $line")
        isempty(fields[1]) && error("Empty variable name in $path")
        push!(linking_vars, String(fields[1]))
        push!(lambda, _parse_checkpoint_number(Float64, fields[2], "lambda", path))
        push!(x_generated, _parse_checkpoint_number(Float64, fields[3], "x_generated", path))
    end

    expected_alpha = op_cost - dot(lambda, x_generated)
    isapprox(alpha, expected_alpha; atol=1e-8, rtol=1e-10) ||
        error("Checkpoint alpha consistency check failed in $path")
    return (
        checkpoint_id=checkpoint_id,
        w=metadata["w"],
        lambda=lambda,
        linking_vars=linking_vars,
        op_cost=op_cost,
        alpha=alpha,
        x_generated=x_generated,
        generating_residual=generating_residual,
        k_added=k_added,
        cut_source=Symbol(get(metadata, "cut_source", "unknown")),
    )
end

function _load_feasibility_cut_checkpoints(directory::AbstractString)
    isdir(directory) || error("Feasibility-cut checkpoint directory does not exist: $directory")
    checkpoint_files = _feasibility_cut_checkpoint_files(directory)
    cuts = [_read_feasibility_cut_checkpoint(path) for (_, path) in checkpoint_files]
    file_ids = first.(checkpoint_files)
    cut_ids = [cut.checkpoint_id for cut in cuts]
    file_ids == cut_ids || error("Checkpoint filename/id mismatch in $directory")
    length(unique(cut_ids)) == length(cut_ids) ||
        error("Duplicate feasibility-cut checkpoint ids in $directory")
    return cuts
end

function _add_replayed_feasibility_cuts!(model::Model, cuts)
    constraints = ConstraintRef[]
    for cut in cuts
        variables = VariableRef[]
        missing_variables = String[]
        for variable_name in cut.linking_vars
            variable = variable_by_name(model, variable_name)
            if isnothing(variable)
                push!(missing_variables, variable_name)
            else
                push!(variables, variable)
            end
        end
        isempty(missing_variables) || error(
            "Cannot replay checkpoint $(cut.checkpoint_id): missing planning variables " *
            join(first(missing_variables, min(5, length(missing_variables))), ", "),
        )
        constraint_name =
            "BendersReplayFeasibilityCut_$(lpad(cut.checkpoint_id, 8, '0'))"
        existing = constraint_by_name(model, constraint_name)
        isnothing(existing) || error("Replay constraint already exists: $constraint_name")
        constraint = @constraint(
            model,
            cut.alpha + sum(cut.lambda[i] * variables[i] for i in eachindex(variables)) <= 0,
            base_name=constraint_name,
        )
        push!(constraints, constraint)
    end
    return constraints
end

function _write_feasibility_checkpoint_state(
    directory::AbstractString,
    cut_count::Integer,
    master_objective::Real,
)
    cut_count >= 0 || error("Checkpoint cut count cannot be negative")
    isfinite(master_objective) || error("Checkpoint master objective must be finite")
    path = joinpath(directory, "master_state.tsv")
    temporary_target = path * ".new"
    ispath(temporary_target) && rm(temporary_target; force=true)
    _atomic_write(temporary_target) do io
        println(io, "MacroEnergySolvers feasibility cut master state\t1")
        println(io, "cut_count\t", cut_count)
        println(io, "master_objective\t", repr(Float64(master_objective)))
    end
    mv(temporary_target, path; force=true)
    return path
end

function _read_feasibility_checkpoint_state(directory::AbstractString)
    path = joinpath(directory, "master_state.tsv")
    isfile(path) || return nothing
    lines = readlines(path)
    length(lines) == 3 || error("Malformed feasibility-cut master state: $path")
    lines[1] == "MacroEnergySolvers feasibility cut master state\t1" ||
        error("Unsupported feasibility-cut master-state header: $path")
    count_fields = split(lines[2], '\t'; limit=2)
    objective_fields = split(lines[3], '\t'; limit=2)
    count_fields[1] == "cut_count" || error("Missing cut_count in $path")
    objective_fields[1] == "master_objective" || error("Missing master_objective in $path")
    return (
        cut_count=_parse_checkpoint_number(Int, count_fields[2], "cut_count", path),
        master_objective=_parse_checkpoint_number(
            Float64,
            objective_fields[2],
            "master_objective",
            path,
        ),
    )
end

const _FEASIBILITY_PROXIMAL_CENTER_HEADER =
    "MacroEnergySolvers feasibility proximal center\t1"

function _write_feasibility_proximal_center(
    directory::AbstractString,
    cut_count::Integer,
    planning_sol::NamedTuple,
    planning_variables::AbstractVector{<:AbstractString},
)
    cut_count >= 0 || error("Proximal-center cut count cannot be negative")
    variable_names = sort!(
        String[
            name for name in planning_variables
            if !startswith(name, "vTHETA")
        ],
    )
    isempty(variable_names) && error("Cannot checkpoint an empty proximal center")
    isfinite(planning_sol.planning_cost) ||
        error("Proximal center contains a non-finite planning cost")
    for variable_name in variable_names
        haskey(planning_sol.values, variable_name) || error(
            "Proximal center is missing planning variable $variable_name",
        )
        isfinite(planning_sol.values[variable_name]) || error(
            "Proximal center contains a non-finite value for $variable_name",
        )
    end

    path = joinpath(directory, "feasibility_proximal_center.tsv")
    temporary_target = path * ".new"
    ispath(temporary_target) && rm(temporary_target; force=true)
    _atomic_write(temporary_target) do io
        println(io, _FEASIBILITY_PROXIMAL_CENTER_HEADER)
        println(io, "cut_count\t", cut_count)
        println(io, "n_variables\t", length(variable_names))
        println(io, "planning_cost\t", repr(Float64(planning_sol.planning_cost)))
        println(io, "variable_name\tvalue")
        for variable_name in variable_names
            println(
                io,
                _validate_checkpoint_text(variable_name, "variable name"),
                '\t',
                repr(Float64(planning_sol.values[variable_name])),
            )
        end
    end
    mv(temporary_target, path; force=true)
    return path
end

function _read_feasibility_proximal_center(directory::AbstractString)
    path = joinpath(directory, "feasibility_proximal_center.tsv")
    isfile(path) || return nothing
    lines = readlines(path)
    length(lines) >= 5 || error("Malformed feasibility proximal center: $path")
    lines[1] == _FEASIBILITY_PROXIMAL_CENTER_HEADER ||
        error("Unsupported feasibility proximal-center header: $path")

    cut_fields = split(lines[2], '\t'; limit=2)
    count_fields = split(lines[3], '\t'; limit=2)
    cost_fields = split(lines[4], '\t'; limit=2)
    cut_fields[1] == "cut_count" || error("Missing cut_count in $path")
    count_fields[1] == "n_variables" || error("Missing n_variables in $path")
    cost_fields[1] == "planning_cost" || error("Missing planning_cost in $path")
    lines[5] == "variable_name\tvalue" || error("Missing variable header in $path")
    cut_count = _parse_checkpoint_number(Int, cut_fields[2], "cut_count", path)
    n_variables = _parse_checkpoint_number(Int, count_fields[2], "n_variables", path)
    planning_cost = _parse_checkpoint_number(
        Float64,
        cost_fields[2],
        "planning_cost",
        path,
    )
    variable_lines = lines[6:end]
    length(variable_lines) == n_variables || error(
        "Proximal center $path declares $n_variables variables but contains " *
        "$(length(variable_lines))",
    )

    values = Dict{String,Float64}()
    for line in variable_lines
        fields = split(line, '\t'; limit=2)
        length(fields) == 2 || error("Malformed proximal-center variable in $path: $line")
        isempty(fields[1]) && error("Empty proximal-center variable name in $path")
        haskey(values, fields[1]) && error(
            "Duplicate proximal-center variable $(fields[1]) in $path",
        )
        values[String(fields[1])] = _parse_checkpoint_number(
            Float64,
            fields[2],
            "proximal-center value",
            path,
        )
    end
    return (
        cut_count=cut_count,
        planning_cost=planning_cost,
        values=values,
        path=path,
    )
end

function _recover_latest_feasibility_generating_point(
    cuts,
    raw_planning_sol::NamedTuple,
)
    isempty(cuts) && return nothing
    # Iteration numbers restart at zero for each replay job, so the numerically
    # largest k is not necessarily the newest point. Checkpoints are loaded in
    # increasing id order; recover the final contiguous group instead.
    latest_iteration = last(cuts).k_added
    first_latest_index = length(cuts)
    while first_latest_index > 1 &&
            cuts[first_latest_index - 1].k_added == latest_iteration
        first_latest_index -= 1
    end
    latest_cuts = cuts[first_latest_index:end]
    recovered_values = copy(raw_planning_sol.values)
    recovered_names = Set{String}()

    for cut in latest_cuts
        for (variable_name, value) in zip(cut.linking_vars, cut.x_generated)
            if variable_name in recovered_names
                isapprox(
                    recovered_values[variable_name],
                    value;
                    atol=1e-8,
                    rtol=1e-9,
                ) || error(
                    "Conflicting generating-point values for $variable_name in " *
                    "replayed iteration $latest_iteration",
                )
            else
                recovered_values[variable_name] = value
                push!(recovered_names, variable_name)
            end
        end
    end
    return (
        planning_cost=raw_planning_sol.planning_cost,
        values=recovered_values,
        latest_iteration=latest_iteration,
        recovered_variables=length(recovered_names),
        cuts=length(latest_cuts),
    )
end
