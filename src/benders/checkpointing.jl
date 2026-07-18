const _FEASIBILITY_CUT_CHECKPOINT_VERSION = 1
const _FEASIBILITY_CUT_CHECKPOINT_HEADER =
    "MacroEnergySolvers feasibility cut checkpoint\t$(_FEASIBILITY_CUT_CHECKPOINT_VERSION)"

function _checkpoint_env_flag(name::String, default::Bool=false)
    raw_value = lowercase(strip(get(ENV, name, string(default))))
    raw_value in ("1", "true", "yes", "on") && return true
    raw_value in ("0", "false", "no", "off") && return false
    error("$name must be a boolean value; received $(repr(raw_value))")
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
