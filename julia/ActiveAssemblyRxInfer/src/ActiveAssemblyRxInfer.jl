module ActiveAssemblyRxInfer

using Dates
using Distributions
using JSON3
using RxInfer
using SHA
using Statistics

export run_rxinfer, main

const DEFAULT_MODEL_NAME = "gaussian_event_code_model"

@model function gaussian_event_code_model(y)
    μ ~ NormalMeanVariance(0.0, 100.0)
    τ ~ GammaShapeRate(1.0, 1.0)
    for i in eachindex(y)
        y[i] ~ NormalMeanPrecision(μ, τ)
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
                "model_name" => DEFAULT_MODEL_NAME,
                "mode" => "static",
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
        "note" => "Numeric event-code Gaussian model; categorical RxInfer HMM is a planned extension.",
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

function _parse_args(args::Vector{String})
    trace_path = nothing
    output_dir = nothing
    iterations = 8
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
    return String(trace_path), String(output_dir), iterations
end

function main(args::Vector{String})
    trace_path, output_dir, iterations = _parse_args(args)
    result = run_rxinfer(trace_path, output_dir; iterations = iterations)
    println(JSON3.write(result["diagnostics"]))
    return 0
end

end
