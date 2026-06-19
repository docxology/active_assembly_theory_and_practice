using ActiveAssemblyRxInfer
using JSON3
using Test

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

    result = ActiveAssemblyRxInfer.run_rxinfer(trace_path, output_dir; iterations = 4)
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

    result = ActiveAssemblyRxInfer.run_rxinfer_categorical(trace_path, output_dir; iterations = 4)
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
end
