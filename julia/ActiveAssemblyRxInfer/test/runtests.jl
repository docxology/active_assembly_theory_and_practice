using ActiveAssemblyRxInfer
using JSON3
using Test
using SHA

@testset "ActiveAssemblyRxInfer" begin
    tmp = mktempdir()
    trace_path = joinpath(tmp, "trace_bundle.json")
    output_dir = joinpath(tmp, "rxinfer")
    trace = Dict(
        "schema_version" => "1.0.0",
        "sample_id" => "julia_fixture",
        "architecture" => "x86_64",
        "binary_sha256" => repeat("0", 64),
        "observations" => [
            Dict("observation_id" => 0, "category" => "load_immediate", "mnemonic" => "mov"),
            Dict("observation_id" => 1, "category" => "arithmetic_update", "mnemonic" => "add"),
            Dict("observation_id" => 2, "category" => "bitwise_transform", "mnemonic" => "shl"),
            Dict("observation_id" => 1, "category" => "arithmetic_update", "mnemonic" => "sub"),
        ],
    )
    open(trace_path, "w") do io
        write(io, JSON3.write(trace))
    end

    root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    model_path = joinpath(tmp, "gaussian_gnn_model.json")
    code = "import json; from src.gnn_numeric import bind_numeric; print(json.dumps(bind_numeric('gaussian_event_code').to_dict()))"
    model_json = cd(root) do
        read(`uv run --locked python -c $code`, String)
    end
    write(model_path, model_json)
    result = ActiveAssemblyRxInfer.run_rxinfer(trace_path, output_dir; iterations = 4, model_path=model_path)
    diagnostics = result["diagnostics"]
    @test diagnostics["iteration_count"] == 4
    @test diagnostics["all_free_energy_finite"] == true
    @test isfinite(diagnostics["free_energy_final"])

    free_energy_path = joinpath(output_dir, "free_energy_trace.jsonl")
    @test isfile(free_energy_path)
    lines = readlines(free_energy_path)
    @test length(lines) == 4
    records = JSON3.read.(lines)
    @test all(record -> record.is_finite == true, records)
    @test all(record -> record.source_field == "result.free_energy", records)
    @test isfile(joinpath(output_dir, "posteriors.json"))
    @test isfile(joinpath(output_dir, "diagnostics.json"))
    @test isfile(joinpath(output_dir, "model_config.json"))
    changed_code = "import json; from src.gnn_numeric import bind_numeric; print(json.dumps(bind_numeric('gaussian_event_code', {'mean': 20.0, 'variance': 0.1, 'shape': 2.0, 'rate': 3.0}).to_dict()))"
    changed_json = cd(root) do
        read(`uv run --locked python -c $changed_code`, String)
    end
    changed_path = joinpath(tmp, "changed_prior.json")
    write(changed_path, changed_json)
    changed = ActiveAssemblyRxInfer.run_rxinfer(trace_path, joinpath(tmp, "changed"); iterations=4, model_path=changed_path)
    @test !isapprox(changed["diagnostics"]["free_energy_final"], diagnostics["free_energy_final"]; atol=1e-3)
    invalid = JSON3.read(model_json, Dict{String,Any})
    invalid["parameters"]["variance"] = [-1.0]
    invalid_path = joinpath(tmp, "invalid_prior.json")
    write(invalid_path, JSON3.write(invalid))
    @test_throws ErrorException ActiveAssemblyRxInfer._gaussian_gnn_parameters(invalid_path)
    invalid = JSON3.read(model_json, Dict{String,Any})
    invalid["source_sha256"] = repeat("0", 64)
    write(invalid_path, JSON3.write(invalid))
    @test_throws ErrorException ActiveAssemblyRxInfer._gaussian_gnn_parameters(invalid_path)
end

@testset "ActiveAssemblyRxInfer categorical" begin
    tmp = mktempdir()
    trace_path = joinpath(tmp, "categorical_trace_bundle.json")
    output_dir = joinpath(tmp, "categorical_rxinfer")
    trace = Dict(
        "schema_version" => "1.0.0",
        "sample_id" => "julia_categorical_fixture",
        "architecture" => "x86_64",
        "binary_sha256" => repeat("1", 64),
        "observations" => [
            Dict("observation_id" => 0, "category" => "load_immediate", "mnemonic" => "mov"),
            Dict("observation_id" => 1, "category" => "arithmetic_update", "mnemonic" => "add"),
            Dict("observation_id" => 2, "category" => "bitwise_transform", "mnemonic" => "xor"),
            Dict("observation_id" => 3, "category" => "comparison", "mnemonic" => "cmp"),
            Dict("observation_id" => 4, "category" => "control_transfer", "mnemonic" => "jne"),
        ],
    )
    open(trace_path, "w") do io
        write(io, JSON3.write(trace))
    end

    root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    model_path = joinpath(tmp, "gnn_model.json")
    code = "from pathlib import Path; import json; from src.gnn_parser import compile_gnn; print(json.dumps(compile_gnn(Path('configs/gnn/passive_phase.md')).to_dict()))"
    model_json = cd(root) do
        read(`uv run python -c $code`, String)
    end
    write(model_path, model_json)
    result = ActiveAssemblyRxInfer.run_rxinfer_categorical(trace_path, output_dir; iterations = 4, model_path = model_path)
    diagnostics = result["diagnostics"]
    @test diagnostics["iteration_count"] == 4
    @test diagnostics["all_free_energy_finite"] == true
    @test isfinite(diagnostics["free_energy_final"])

    free_energy_path = joinpath(output_dir, "categorical_free_energy_trace.jsonl")
    @test isfile(free_energy_path)
    lines = readlines(free_energy_path)
    @test length(lines) == 4
    records = JSON3.read.(lines)
    @test all(record -> record.is_finite == true, records)
    @test all(record -> record.model_name == "categorical_phase_hmm", records)
    @test all(record -> record.mode == "categorical", records)
    @test all(record -> record.source_field == "result.free_energy", records)
    @test isfile(joinpath(output_dir, "categorical_posteriors.json"))
    @test isfile(joinpath(output_dir, "categorical_diagnostics.json"))
    @test isfile(joinpath(output_dir, "categorical_model_config.json"))
    prefixes = JSON3.read(read(joinpath(output_dir, "categorical_evidence_by_prefix.json"), String))
    @test length(prefixes) == length(trace["observations"])
    @test all(row -> row.backend_iterations == 1 && row.conditioning == "trace_prefix", prefixes)
    @test prefixes[end].value ≈ diagnostics["free_energy_final"] atol=1e-8
end


@testset "Shared GNN parameter rejection" begin
    tmp = mktempdir()
    root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    code = "from pathlib import Path; import json; from src.gnn_parser import compile_gnn; print(json.dumps(compile_gnn(Path('configs/gnn/passive_phase.md')).to_dict()))"
    model_json = cd(root) do
        read(`uv run python -c $code`, String)
    end
    payload = JSON3.read(model_json, Dict{String,Any})
    payload["A"][1][1][1] = -0.2
    invalid = joinpath(tmp, "invalid.json")
    write(invalid, JSON3.write(payload))
    @test_throws ErrorException ActiveAssemblyRxInfer._categorical_phase_matrices(invalid)
    sensor_code = replace(code, "passive_phase.md" => "fixture_sensor.md")
    sensor_json = cd(root) do
        read(`uv run python -c $sensor_code`, String)
    end
    unsupported = joinpath(tmp, "sensor.json")
    write(unsupported, sensor_json)
    @test_throws ErrorException ActiveAssemblyRxInfer._categorical_phase_matrices(unsupported)
end


@testset "Shared GNN digest rejection" begin
    tmp = mktempdir()
    root = normpath(joinpath(@__DIR__, "..", "..", ".."))
    code = "from pathlib import Path; import json; from src.gnn_parser import compile_gnn; print(json.dumps(compile_gnn(Path('configs/gnn/passive_phase.md')).to_dict()))"
    model_json = cd(root) do
        read(`uv run python -c $code`, String)
    end
    model_path = joinpath(tmp, "model.json")
    write(model_path, model_json)
    spec = JSON3.read(model_json)
    request = Dict("observations" => [0], "model_digest" => String(spec.model_digest), "model_file_sha256" => bytes2hex(sha256(read(model_path))), "tensor_contract_json" => "{}")
    request_path = joinpath(tmp, "request.json")
    write(request_path, JSON3.write(request))
    @test_throws ErrorException ActiveAssemblyRxInfer.run_shared_categorical(model_path, request_path, joinpath(tmp,"output.json"))
    @test !isfile(joinpath(tmp,"output.json"))
end
