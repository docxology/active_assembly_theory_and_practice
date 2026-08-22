module ActiveAssemblyRxInfer

using Dates
using Distributions
using JSON3
using RxInfer
using SHA
using Statistics

export run_rxinfer, run_rxinfer_categorical, main

const DEFAULT_MODEL_NAME = "gaussian_event_code_model"
const CATEGORICAL_MODEL_NAME = "categorical_phase_hmm"

@model function gaussian_event_code_model(y)
    μ ~ NormalMeanVariance(0.0, 100.0)
    τ ~ GammaShapeRate(1.0, 1.0)
    for i in eachindex(y)
        y[i] ~ NormalMeanPrecision(μ, τ)
    end
end

@model function categorical_phase_hmm(x, initial_probs, transition_matrix, emission_matrix)
    z0 ~ Categorical(initial_probs)
    z_prev = z0
    for t in eachindex(x)
        z[t] ~ DiscreteTransition(z_prev, transition_matrix)
        x[t] ~ DiscreteTransition(z[t], emission_matrix)
        z_prev = z[t]
    end
end

function _read_json(path::AbstractString)
    return JSON3.read(read(path, String))
end

function _write_json(path::AbstractString, payload)
    open(path, "w") do io
        write(io, JSON3.write(payload))
        write(io, "\n")
    end
    return path
end

function _sha256_file(path::AbstractString)::String
    return bytes2hex(sha256(read(path)))
end

function _sha256_text(text::AbstractString)::String
    return bytes2hex(sha256(Vector{UInt8}(codeunits(text))))
end

function _package_version(package)::String
    version = Base.pkgversion(package)
    return isnothing(version) ? "unknown" : string(version)
end

function _normalized_columns(matrix::AbstractMatrix{<:Real})
    normalized = Matrix{Float64}(matrix)
    normalized ./= sum(normalized; dims = 1)
    return normalized
end

function _extract_observations(bundle)
    observations = collect(bundle.observations)
    if isempty(observations)
        error("trace bundle contains no observations")
    end
    values = Float64[]
    categories = String[]
    mnemonics = String[]
    for observation in observations
        push!(values, Float64(observation.observation_id) + 1.0)
        push!(categories, String(observation.category))
        push!(mnemonics, String(observation.mnemonic))
    end
    return values, categories, mnemonics
end

function _run_static_model(values::Vector{Float64}; iterations::Int)
    if iterations <= 0
        error("iterations must be positive")
    end
    initial_mean = mean(values)
    constraints = @constraints begin
        q(μ, τ) = q(μ)q(τ)
    end
    initialization = @initialization begin
        q(μ) = NormalMeanVariance(initial_mean, 100.0)
        q(τ) = GammaShapeRate(1.0, 1.0)
    end
    return infer(
        model = gaussian_event_code_model(),
        data = (y = values,),
        constraints = constraints,
        initialization = initialization,
        iterations = iterations,
        free_energy = true,
    )
end

function _categorical_phase_matrices()
    emission_matrix = [
        0.72 0.06 0.05 0.05 0.06
        0.08 0.74 0.08 0.06 0.08
        0.06 0.10 0.76 0.05 0.08
        0.05 0.04 0.04 0.76 0.10
        0.04 0.03 0.04 0.08 0.68
        0.05 0.03 0.03 0.00 0.00
    ]
    emission_matrix = _normalized_columns(max.(emission_matrix, 1.0e-6))

    self_loop = [0.70, 0.65, 0.65, 0.60, 0.99]
    transition_matrix = zeros(Float64, 5, 5)
    for state in eachindex(self_loop)
        transition_matrix[state, state] = self_loop[state]
        if state < length(self_loop)
            transition_matrix[state + 1, state] = 1.0 - self_loop[state]
        end
    end
    transition_matrix = _normalized_columns(max.(transition_matrix, 1.0e-6))

    initial_probs = [0.34, 0.24, 0.18, 0.14, 0.10]
    initial_probs = max.(initial_probs, 1.0e-6)
    initial_probs ./= sum(initial_probs)

    return initial_probs, transition_matrix, emission_matrix
end

function _observation_one_hot(value::Float64, category_count::Int)
    if !isfinite(value) || !isinteger(value)
        error("observation value $(repr(value)) must be a finite 1-based category index")
    end
    category_index = Int(value)
    if !(1 <= category_index <= category_count)
        error("observation value $(category_index) is out of range for $(category_count) categories")
    end
    encoded = zeros(Float64, category_count)
    encoded[category_index] = 1.0
    return encoded
end

function _categorical_observations(values::Vector{Float64}; category_count::Int = 6)
    return [_observation_one_hot(value, category_count) for value in values]
end

function _run_categorical_model(
    observations::Vector{Vector{Float64}},
    initial_probs::Vector{Float64},
    transition_matrix::Matrix{Float64},
    emission_matrix::Matrix{Float64};
    iterations::Int,
)
    if iterations <= 0
        error("iterations must be positive")
    end
    constraints = @constraints begin
        q(z, z0) = q(z, z0)
    end
    initialization = @initialization begin
        q(z) = vague(Categorical, 5)
    end
    return infer(
        model = categorical_phase_hmm(
            initial_probs = initial_probs,
            transition_matrix = transition_matrix,
            emission_matrix = emission_matrix,
        ),
        data = (x = observations,),
        constraints = constraints,
        initialization = initialization,
        iterations = iterations,
        free_energy = true,
        options = (limit_stack_depth = 500,),
        returnvars = (z = KeepLast(),),
    )
end

function _distribution_summary(distribution)
    summary = Dict{String, Any}("repr" => repr(distribution))
    try
        summary["mean"] = Float64(mean(distribution))
    catch
        summary["mean"] = nothing
    end
    try
        summary["variance"] = Float64(var(distribution))
    catch
        summary["variance"] = nothing
    end
    return summary
end

function _posterior_summary(posterior)
    if posterior isa AbstractVector
        return [_distribution_summary(item) for item in posterior]
    end
    return _distribution_summary(posterior)
end

function _free_energy_records(
    values::Vector{Float64};
    run_id::String,
    sample_id::String,
    data_hash::String,
    config_hash::String,
    model_name::String = DEFAULT_MODEL_NAME,
    mode::String = "static",
)
    rxinfer_version = _package_version(RxInfer)
    records = Vector{Dict{String, Any}}()
    previous = nothing
    for (index, value) in enumerate(values)
        delta = isnothing(previous) ? nothing : value - previous
        push!(
            records,
            Dict{String, Any}(
                "run_id" => run_id,
                "sample_id" => sample_id,
                "model_name" => model_name,
                "mode" => mode,
                "iteration" => index,
                "observation_index" => nothing,
                "value" => value,
                "source_field" => "result.free_energy",
                "is_finite" => isfinite(value),
                "delta" => delta,
                "rxinfer_version" => rxinfer_version,
                "julia_version" => string(Base.VERSION),
                "data_hash" => data_hash,
                "config_hash" => config_hash,
            ),
        )
        previous = value
    end
    return records
end

function _write_jsonl(path::AbstractString, records)
    open(path, "w") do io
        for record in records
            write(io, JSON3.write(record))
            write(io, "\n")
        end
    end
    return path
end

function run_rxinfer(trace_path::AbstractString, output_dir::AbstractString; iterations::Int = 8)
    mkpath(output_dir)
    bundle = _read_json(trace_path)
    observation_values, categories, mnemonics = _extract_observations(bundle)
    sample_id = String(bundle.sample_id)
    model_config = Dict{String, Any}(
        "model_name" => DEFAULT_MODEL_NAME,
        "mode" => "static",
        "iterations" => iterations,
        "observation_count" => length(observation_values),
        "observation_categories" => categories,
        "mnemonics" => mnemonics,
        "observation_values" => observation_values,
        "note" => "Numeric event-code Gaussian model; categorical RxInfer HMM is emitted as a namespaced companion.",
    )
    config_text = JSON3.write(model_config)
    data_hash = _sha256_file(trace_path)
    config_hash = _sha256_text(config_text)
    run_id = "$(sample_id)-$(Dates.format(now(UTC), dateformat"yyyymmddTHHMMSS"))"

    result = _run_static_model(observation_values; iterations = iterations)
    free_energy_values = Float64.(collect(result.free_energy))
    free_energy_records = _free_energy_records(
        free_energy_values;
        run_id = run_id,
        sample_id = sample_id,
        data_hash = data_hash,
        config_hash = config_hash,
    )

    posteriors = Dict{String, Any}()
    for (name, posterior) in result.posteriors
        posteriors[string(name)] = _posterior_summary(posterior)
    end

    finite_values = all(isfinite, free_energy_values)
    deltas = [nothing; [free_energy_values[i] - free_energy_values[i - 1] for i in 2:length(free_energy_values)]]
    diagnostics = Dict{String, Any}(
        "run_id" => run_id,
        "sample_id" => sample_id,
        "model_name" => DEFAULT_MODEL_NAME,
        "mode" => "static",
        "iteration_count" => length(free_energy_values),
        "all_free_energy_finite" => finite_values,
        "free_energy_initial" => first(free_energy_values),
        "free_energy_final" => last(free_energy_values),
        "free_energy_min" => minimum(free_energy_values),
        "free_energy_max" => maximum(free_energy_values),
        "free_energy_deltas" => deltas,
        "rxinfer_version" => _package_version(RxInfer),
        "julia_version" => string(Base.VERSION),
        "data_hash" => data_hash,
        "config_hash" => config_hash,
    )

    outputs = Dict{String, String}(
        "free_energy_trace" => joinpath(output_dir, "free_energy_trace.jsonl"),
        "posteriors" => joinpath(output_dir, "posteriors.json"),
        "diagnostics" => joinpath(output_dir, "diagnostics.json"),
        "model_config" => joinpath(output_dir, "model_config.json"),
    )
    _write_jsonl(outputs["free_energy_trace"], free_energy_records)
    _write_json(outputs["posteriors"], Dict("run_id" => run_id, "posteriors" => posteriors))
    _write_json(outputs["diagnostics"], diagnostics)
    _write_json(outputs["model_config"], model_config)
    return Dict{String, Any}("outputs" => outputs, "diagnostics" => diagnostics)
end

"""Run the categorical phase HMM and write namespaced inference artifacts."""
function run_rxinfer_categorical(
    trace_path::AbstractString,
    output_dir::AbstractString;
    iterations::Int = 8,
)
    mkpath(output_dir)
    bundle = _read_json(trace_path)
    observation_values, categories, mnemonics = _extract_observations(bundle)
    observations = _categorical_observations(observation_values)
    initial_probs, transition_matrix, emission_matrix = _categorical_phase_matrices()
    sample_id = String(bundle.sample_id)
    model_config = Dict{String, Any}(
        "model_name" => CATEGORICAL_MODEL_NAME,
        "mode" => "categorical",
        "iterations" => iterations,
        "observation_count" => length(observation_values),
        "observation_categories" => categories,
        "mnemonics" => mnemonics,
        "observation_values" => observation_values,
        "initial_probs" => initial_probs,
        "transition_matrix" => [collect(row) for row in eachrow(transition_matrix)],
        "emission_matrix" => [collect(row) for row in eachrow(emission_matrix)],
        "note" => "Categorical RxInfer HMM with fixed left-to-right latent phase dynamics.",
    )
    config_text = JSON3.write(model_config)
    data_hash = _sha256_file(trace_path)
    config_hash = _sha256_text(config_text)
    run_id = "$(sample_id)-$(Dates.format(now(UTC), dateformat"yyyymmddTHHMMSS"))"

    result = _run_categorical_model(
        observations,
        initial_probs,
        transition_matrix,
        emission_matrix;
        iterations = iterations,
    )
    free_energy_values = Float64.(collect(result.free_energy))
    finite_values = all(isfinite, free_energy_values)
    if !finite_values
        error("categorical RxInfer free energy contains non-finite values")
    end
    free_energy_records = _free_energy_records(
        free_energy_values;
        run_id = run_id,
        sample_id = sample_id,
        data_hash = data_hash,
        config_hash = config_hash,
        model_name = CATEGORICAL_MODEL_NAME,
        mode = "categorical",
    )

    posteriors = Dict{String, Any}()
    for (name, posterior) in result.posteriors
        posteriors[string(name)] = _posterior_summary(posterior)
    end

    deltas = [nothing; [free_energy_values[i] - free_energy_values[i - 1] for i in 2:length(free_energy_values)]]
    diagnostics = Dict{String, Any}(
        "run_id" => run_id,
        "sample_id" => sample_id,
        "model_name" => CATEGORICAL_MODEL_NAME,
        "mode" => "categorical",
        "iteration_count" => length(free_energy_values),
        "all_free_energy_finite" => finite_values,
        "free_energy_initial" => first(free_energy_values),
        "free_energy_final" => last(free_energy_values),
        "free_energy_min" => minimum(free_energy_values),
        "free_energy_max" => maximum(free_energy_values),
        "free_energy_deltas" => deltas,
        "rxinfer_version" => _package_version(RxInfer),
        "julia_version" => string(Base.VERSION),
        "data_hash" => data_hash,
        "config_hash" => config_hash,
    )

    outputs = Dict{String, String}(
        "free_energy_trace" => joinpath(output_dir, "categorical_free_energy_trace.jsonl"),
        "posteriors" => joinpath(output_dir, "categorical_posteriors.json"),
        "diagnostics" => joinpath(output_dir, "categorical_diagnostics.json"),
        "model_config" => joinpath(output_dir, "categorical_model_config.json"),
    )
    _write_jsonl(outputs["free_energy_trace"], free_energy_records)
    _write_json(outputs["posteriors"], Dict("run_id" => run_id, "posteriors" => posteriors))
    _write_json(outputs["diagnostics"], diagnostics)
    _write_json(outputs["model_config"], model_config)
    return Dict{String, Any}("outputs" => outputs, "diagnostics" => diagnostics)
end

function _parse_args(args::Vector{String})
    trace_path = nothing
    output_dir = nothing
    iterations = 8
    model = "gaussian"
    index = 1
    while index <= length(args)
        arg = args[index]
        if arg == "--trace"
            index += 1
            trace_path = args[index]
        elseif arg == "--output"
            index += 1
            output_dir = args[index]
        elseif arg == "--iterations"
            index += 1
            iterations = parse(Int, args[index])
        elseif arg == "--model"
            index += 1
            model = args[index]
        else
            error("unknown argument: $(arg)")
        end
        index += 1
    end
    if isnothing(trace_path)
        error("--trace is required")
    end
    if isnothing(output_dir)
        error("--output is required")
    end
    if !(model in ("gaussian", "categorical"))
        error("--model must be gaussian or categorical")
    end
    return String(trace_path), String(output_dir), iterations, model
end

function main(args::Vector{String})
    trace_path, output_dir, iterations, model = _parse_args(args)
    result = if model == "categorical"
        run_rxinfer_categorical(trace_path, output_dir; iterations = iterations)
    else
        run_rxinfer(trace_path, output_dir; iterations = iterations)
    end
    println(JSON3.write(result["diagnostics"]))
    return 0
end

end
