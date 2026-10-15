module ActiveAssemblyRxInfer

using Dates
using Distributions
using JSON3
using RxInfer
using SHA
using Statistics

export run_rxinfer, run_rxinfer_categorical, run_shared_categorical, main

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
    z[1] ~ Categorical(initial_probs)
    x[1] ~ DiscreteTransition(z[1], emission_matrix)
    for t in 2:length(x)
        z[t] ~ DiscreteTransition(z[t-1], transition_matrix)
        x[t] ~ DiscreteTransition(z[t], emission_matrix)
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

function _categorical_phase_matrices(model_path::AbstractString)
    spec = _read_json(model_path)
    if spec.schema_version != "1.0" || spec.dialect != "active-assembly-gnn-1" || length(spec.factors) != 1 || length(spec.modalities) != 1 || length(spec.actions[1]) != 1
        error("GNN_NATIVE_UNSUPPORTED_MODEL")
    end
    if length(spec.A) != 1 || length(spec.B) != 1 || length(spec.C) != 1 || length(spec.D) != 1 || length(spec.state_labels) != 1 || length(spec.outcomes) != 1 || length(spec.actions) != 1 || spec.A_dependencies != [[0]] || spec.B_dependencies != [[0]] || spec.B_action_dependencies != [[0]]
        error("GNN_NATIVE_UNSUPPORTED_MODEL")
    end
    spec.horizon isa Integer && !(spec.horizon isa Bool) && 1 <= spec.horizon <= 3 || error("GNN-E103: native horizon")
    isfinite(spec.gamma) && spec.gamma > 0 || error("GNN-E102: native gamma")
    length(spec.E) == 1 && isfinite(spec.E[1]) && abs(spec.E[1]-1) <= 1e-10 || error("GNN-E101: native E")
    all(isfinite, spec.C[1]) || error("GNN-E101: native utilities")
    length(spec.policies) == 1 && length(spec.policies[1]) == spec.horizon && all(step -> step == [0], spec.policies[1]) || error("GNN-E102: native policies")
    for names in (spec.factors, spec.modalities, spec.state_labels[1], spec.actions[1], spec.outcomes[1])
        all(n -> n isa AbstractString && !isempty(strip(n)), names) && length(unique(names)) == length(names) || error("GNN-E102: native names")
    end
    occursin(r"^[0-9a-f]{64}$", String(spec.source_sha256)) || error("GNN-E102: native source digest")
    n = length(spec.D[1]); o = length(spec.C[1])
    if !(1 <= n <= 64 && 1 <= o <= 64) || length(spec.state_labels[1]) != n || length(spec.outcomes[1]) != o || length(spec.A[1]) != o || length(spec.B[1]) != n
        error("GNN-E102: native axes mismatch")
    end
    initial_probs = Float64.(spec.D[1])
    emission_matrix = zeros(Float64, o, n)
    transition_matrix = zeros(Float64, n, n)
    for i in 1:o
        length(spec.A[1][i]) == n || error("GNN-E102: native A axes")
        emission_matrix[i,:] = Float64.(spec.A[1][i])
    end
    for i in 1:n
        length(spec.B[1][i]) == n || error("GNN-E102: native B axes")
        for j in 1:n
            length(spec.B[1][i][j]) == 1 || error("GNN_NATIVE_UNSUPPORTED_MODEL")
            transition_matrix[i,j] = Float64(spec.B[1][i][j][1])
        end
    end
    for matrix in (emission_matrix, transition_matrix)
        all(isfinite, matrix) && all(matrix .>= 0) && all(abs.(vec(sum(matrix; dims=1)) .- 1) .<= 1e-10) || error("GNN-E101: native invalid distribution")
    end
    all(isfinite, initial_probs) && all(initial_probs .>= 0) && abs(sum(initial_probs)-1) <= 1e-10 || error("GNN-E101: native invalid prior")
    occursin(r"^[0-9a-f]{64}$", String(spec.model_digest)) || error("GNN-E102: native invalid digest")
    return initial_probs, transition_matrix, emission_matrix
end

function run_shared_categorical(model_path::AbstractString, observations_path::AbstractString, output_path::AbstractString)
    initial, transition, emission = _categorical_phase_matrices(model_path)
    spec = _read_json(model_path)
    input = _read_json(observations_path)
    input.model_digest == spec.model_digest || error("GNN-E102: model digest mismatch")
    input.model_file_sha256 == _sha256_file(model_path) || error("GNN-E102: model bytes mismatch")
    contract_text = String(input.tensor_contract_json)
    _sha256_text(contract_text) == String(spec.model_digest) || error("GNN-E102: tensor digest mismatch")
    contract = JSON3.read(contract_text)
    expected_keys = Set(["schema_version", "dialect", "factors", "state_labels", "modalities", "actions", "outcomes", "A", "B", "C", "D", "E", "A_dependencies", "B_dependencies", "B_action_dependencies", "policies", "horizon", "gamma", "seed"])
    Set(String.(keys(contract))) == expected_keys || error("GNN-E102: tensor contract fields mismatch")
    Set(String.(keys(spec))) == union(expected_keys, Set(["name", "source_sha256", "annotations", "model_digest"])) || error("GNN-E102: model fields mismatch")
    for key in keys(contract)
        contract[key] == spec[key] || error("GNN-E102: tensor contract content mismatch")
    end
    isempty(input.observations) && error("GNN_EMPTY_OBSERVATIONS")
    values = Float64.(input.observations) .+ 1
    observations = _categorical_observations(values; category_count=size(emission,1))
    result = _run_categorical_model(observations, initial, transition, emission; iterations=8)
    rows = [Float64.(probvec(q)) for q in result.posteriors[:z]]
    payload = Dict("model_digest" => String(spec.model_digest), "model_source_sha256" => String(spec.source_sha256), "model_file_sha256" => _sha256_file(model_path), "trace_sha256" => _sha256_file(observations_path), "time_convention" => "first_observation_D", "backend" => "RxInfer", "julia_version" => string(VERSION), "rxinfer_version" => _package_version(RxInfer), "native_smoothed" => rows, "free_energy" => Float64.(result.free_energy))
    _write_json(output_path, payload)
    return payload
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
        q(z) = q(z)
    end
    initialization = @initialization begin
        q(z) = vague(Categorical, length(initial_probs))
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
    model_path::AbstractString,
)
    mkpath(output_dir)
    bundle = _read_json(trace_path)
    observation_values, categories, mnemonics = _extract_observations(bundle)
    initial_probs, transition_matrix, emission_matrix = _categorical_phase_matrices(model_path)
    observations = _categorical_observations(observation_values; category_count=size(emission_matrix,1))
    spec = _read_json(model_path)
    sample_id = String(bundle.sample_id)
    model_config = Dict{String, Any}(
        "model_name" => CATEGORICAL_MODEL_NAME,
        "model_digest" => String(spec.model_digest),
        "time_convention" => "first_observation_D",
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
        "model_digest" => String(spec.model_digest),
        "time_convention" => "first_observation_D",
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
    model_path = nothing
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
        elseif arg == "--model-spec"
            index += 1
            model_path = args[index]
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
    return String(trace_path), String(output_dir), iterations, model, model_path
end

function main(args::Vector{String})
    trace_path, output_dir, iterations, model, model_path = _parse_args(args)
    result = if model == "categorical"
        run_rxinfer_categorical(trace_path, output_dir; iterations = iterations, model_path = (isnothing(model_path) ? error("--model-spec is required") : String(model_path)))
    else
        run_rxinfer(trace_path, output_dir; iterations = iterations)
    end
    println(JSON3.write(result["diagnostics"]))
    return 0
end

end
