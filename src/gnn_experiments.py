"""Canonical bounded GNN study and source-bound outputs."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from .corpus import build_corpus
from .disassembly import disassemble_bytes
from .features import categorize_instruction
from .fixture_sensor_environment import FixtureSensorEnvironment
from .gnn_upstream import compile_upstream
from .gnn_reference import forward_backward, policy_reference
from .provenance import sha256_path, toolchain_versions
from .pymdp_runtime import PymdpRuntime, beliefs, run_pymdp
from .receipt import build_receipt_envelope

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = {
    "protocol_id": "active-assembly-gnn-finite-v1",
    "models": ["passive_phase", "fixture_sensor"],
    "time_convention": "first_observation_D",
    "objective": "predictive-policy utility plus state information gain",
}
PROTOCOL["protocol_sha256"] = hashlib.sha256(
    json.dumps(PROTOCOL, sort_keys=True).encode()
).hexdigest()


def source_hashes():
    paths = [
        ROOT / "pyproject.toml",
        ROOT / "uv.lock",
        *sorted((ROOT / "configs/gnn").rglob("*.md")),
        *(
            ROOT / "src" / n
            for n in (
                "gnn_model.py",
                "gnn_parser.py",
                "fixture_sensor_environment.py",
                "pymdp_runtime.py",
                "gnn_reference.py",
                "gnn_experiments.py",
                "gnn_upstream.py",
                "gnn_upstream_worker.py",
                "gnn_cli.py",
                "gnn_validation.py",
                "receipt.py",
                "provenance.py",
                "corpus.py",
                "disassembly.py",
                "emulation.py",
                "features.py",
            )
        ),
    ]
    return {str(p.relative_to(ROOT)): sha256_path(p) for p in paths}


def envelope(payload):
    return {
        **build_receipt_envelope(
            protocol=PROTOCOL,
            generator_revision=sha256_path(Path(__file__)),
            input_hashes=source_hashes(),
            toolchain_versions=toolchain_versions(
                ["inferactively-pymdp", "jax", "numpy", "capstone", "unicorn"]
            ),
            status="validated_bounded",
            limitations=[
                "Two transparent fixtures only; authored phase reference; dimensionless illustrative utilities; no learned semantics or generalization claim."
            ],
        ),
        "generated_at": datetime.now(UTC).isoformat(),
        **payload,
    }


def write_payload(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def model_payload(model):
    return {**model.to_dict(), "model_digest": model.tensor_digest}


def compare_oracle(model, run):
    maximum = 0.0
    for row in run["trajectory"]:
        ref = policy_reference(model, [np.array(q) for q in row["posterior"]])
        maximum = max(
            maximum,
            float(np.max(np.abs(np.array(row["q_pi"]) - ref["q_pi"]))),
            float(np.max(np.abs(np.array(row["negative_efe"]) - ref["negative_efe"]))),
        )
        if (
            run.get("action_selection", "deterministic") == "deterministic"
            and row["action"] != ref["action"]
        ):
            raise RuntimeError("GNN_ORACLE_ACTION_MISMATCH")
    if maximum > 1e-5:
        raise RuntimeError("GNN_ORACLE_POLICY_MISMATCH")
    return maximum


def run_passive(
    model,
    fixture_name: str = "x86_accumulator",
    *,
    steps: int | None = None,
    seed: int | None = None,
):
    seed = model.seed if seed is None else seed
    if fixture_name not in ("x86_accumulator", "x86_bitwise"):
        raise ValueError("SENSOR_UNSUPPORTED_FIXTURE")
    if (
        len(model.factors) != 1
        or len(model.modalities) != 1
        or model.actions != (("noop",),)
    ):
        raise ValueError("GNN_PASSIVE_CONTRACT_MISMATCH")
    fixture = next(f for f in build_corpus() if f.name == fixture_name)
    observations = [
        model.outcomes[0].index(categorize_instruction(i))
        for i in disassemble_bytes(fixture.code)
    ]
    if steps is not None:
        if type(steps) is not int or not 1 <= steps <= len(observations):
            raise ValueError("GNN_INVALID_RUN_BOUND")
        observations = observations[:steps]
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("GNN_INVALID_RUN_BOUND")
    runtime = PymdpRuntime(model)
    rows = []
    for t, o in enumerate(observations):
        q = runtime.infer((o,))
        pi, neg = runtime.agent.infer_policies(q)
        action = runtime.agent.sample_action(pi)
        rows.append(
            {
                "prior": [
                    np.asarray(v)[0].astype(float).tolist() for v in runtime.prior
                ],
                "observation": [o],
                "posterior": beliefs(q),
                "q_pi": np.asarray(pi)[0].astype(float).tolist(),
                "negative_efe": np.asarray(neg)[0].astype(float).tolist(),
                "efe": (-np.asarray(neg)[0]).astype(float).tolist(),
                "action": np.asarray(action)[0].astype(int).tolist(),
            }
        )
        runtime.prior = runtime.agent.update_empirical_prior(action, q)
        runtime.assert_unchanged()
    exact = forward_backward(model, observations)
    error = float(
        np.max(np.abs(np.array([r["posterior"][0] for r in rows]) - exact["filtered"]))
    )
    if error > 1e-5:
        raise RuntimeError("GNN_ORACLE_POSTERIOR_MISMATCH")
    return {
        "model_digest": model.tensor_digest,
        "model_source_sha256": model.source_sha256,
        "backend": "pymdp.Agent",
        "backend_version": importlib.metadata.version("inferactively-pymdp"),
        "seed": seed,
        "final_posterior": rows[-1]["posterior"],
        "time_convention": "first_observation_D",
        "steps": len(rows),
        "trajectory": rows,
        "exact_reference": exact,
        "oracle_max_error": error,
        "tensors_unchanged": True,
        "fixture_sha256": hashlib.sha256(fixture.code).hexdigest(),
    }


def control_variants(model):
    a = [v.copy() for v in model.A]
    a[0][2, :, 2] = [0.1, 0.9]
    a[0][3, :, 2] = [0.9, 0.1]
    b = [v.copy() for v in model.B]
    b[0][:, :, 0] = [[0.9, 0.9], [0.1, 0.1]]
    c = [v.copy() for v in model.C]
    c[1][2] = -2.0
    d = [v.copy() for v in model.D]
    d[0] = np.array([0.9, 0.1])
    return {
        "A": replace(model, A=tuple(a)),
        "B": replace(model, B=tuple(b)),
        "C": replace(model, C=tuple(c)),
        "D": replace(model, D=tuple(d)),
        "E": replace(model, E=np.array([0.98, 0.01, 0.01])),
    }


def run_gnn_study(output_dir: Path, gnn_root: Path):
    output_dir = Path(output_dir)
    passive_qualified = compile_upstream(
        ROOT / "configs/gnn/passive_phase.md", gnn_root
    )
    sensor_qualified = compile_upstream(
        ROOT / "configs/gnn/fixture_sensor.md", gnn_root
    )
    passive = passive_qualified.model
    sensor = sensor_qualified.model
    passive_run = run_passive(passive)
    sensor_run = run_pymdp(
        sensor, FixtureSensorEnvironment("x86_accumulator"), steps=2, seed=sensor.seed
    )
    errors = [compare_oracle(sensor, sensor_run)]
    alternate = run_pymdp(
        sensor, FixtureSensorEnvironment("x86_bitwise"), steps=2, seed=sensor.seed
    )
    errors.append(compare_oracle(sensor, alternate))
    controls = {}
    for key, variant in control_variants(sensor).items():
        run = run_pymdp(
            variant,
            FixtureSensorEnvironment("x86_accumulator"),
            steps=1,
            seed=variant.seed,
        )
        error = compare_oracle(variant, run)
        errors.append(error)
        delta = float(
            np.max(
                np.abs(
                    np.array(run["trajectory"][0]["q_pi"])
                    - sensor_run["trajectory"][0]["q_pi"]
                )
            )
        )
        if delta <= 1e-4:
            raise RuntimeError("GNN_INEFFECTIVE_INTERVENTION")
        controls[key] = {
            "model": model_payload(variant),
            "run": run,
            "oracle": policy_reference(
                variant, [np.array(v) for v in run["trajectory"][0]["posterior"]]
            ),
            "policy_delta": delta,
            "oracle_max_error": error,
            "passed": True,
        }
    horizon = replace(sensor, horizon=2, E=np.full(9, 1 / 9))
    hrun = run_pymdp(
        horizon, FixtureSensorEnvironment("x86_accumulator"), steps=1, seed=horizon.seed
    )
    errors.append(compare_oracle(horizon, hrun))
    invariant = (
        sensor_run["trajectory"][0]["q_pi"] == alternate["trajectory"][0]["q_pi"]
    )
    if not invariant:
        raise RuntimeError("GNN_WITHHELD_INFORMATION_LEAK")
    digests = {"passive": passive.tensor_digest, "sensor": sensor.tensor_digest}
    initial_reference = policy_reference(
        sensor, [np.array([0.5, 0.5]), np.array([1.0, 0.0, 0.0])]
    )
    log_e = -sensor.gamma * np.array(initial_reference["negative_efe"])
    random_e = np.exp(log_e - log_e.max())
    random_e /= random_e.sum()
    random_model = replace(sensor, E=random_e)
    random_run = run_pymdp(
        random_model,
        FixtureSensorEnvironment("x86_accumulator"),
        steps=1,
        seed=11,
        action_selection="stochastic",
    )
    random_reference = policy_reference(
        random_model, [np.array(v) for v in random_run["trajectory"][0]["posterior"]]
    )
    random_error = compare_oracle(random_model, random_run)
    if random_error > 1e-5:
        raise RuntimeError("GNN_RANDOM_BASELINE_MISMATCH")
    errors.append(random_error)
    baseline_payloads = {}
    for label, e in (
        ("fixed_idle", [1.0 - 2e-6, 1e-6, 1e-6]),
        ("fixed_static", [1e-6, 1.0 - 2e-6, 1e-6]),
    ):
        baseline_model = replace(sensor, E=np.array(e))
        baseline_run = run_pymdp(
            baseline_model, FixtureSensorEnvironment("x86_accumulator"), steps=1, seed=0
        )
        baseline_oracle = policy_reference(
            baseline_model,
            [np.array(v) for v in baseline_run["trajectory"][0]["posterior"]],
        )
        baseline_error = compare_oracle(baseline_model, baseline_run)
        errors.append(baseline_error)
        baseline_payloads[label] = {
            "model": model_payload(baseline_model),
            "run": baseline_run,
            "oracle": baseline_oracle,
            "oracle_max_error": baseline_error,
        }
    control_payload = {
        "model_digests": digests,
        "controls": controls,
        "withheld_invariance": {
            "passed": invariant,
            "accumulator": sensor_run["trajectory"][0],
            "bitwise": alternate["trajectory"][0],
            "bitwise_run": alternate,
            "revealed_posterior_diverges": sensor_run["final_posterior"][0]
            != alternate["final_posterior"][0],
        },
        "horizon_sensitivity": {
            "horizon": 2,
            "model": model_payload(horizon),
            "run": hrun,
        },
        "oracle_max_error": max(errors),
        "tolerance": 1e-5,
        "baselines": {
            "random_uniform_policy": {
                "model": model_payload(random_model),
                "run": random_run,
                "oracle": random_reference,
                "oracle_max_error": random_error,
            },
            **baseline_payloads,
        },
    }
    figures = {
        "model_digests": digests,
        "passive_posterior": [r["posterior"][0] for r in passive_run["trajectory"]],
        "passive_smoothed": passive_run["exact_reference"]["smoothed"],
        "sensor_policies": sensor_run["trajectory"][0]["q_pi"],
        "sensor_actions": [r["action"] for r in sensor_run["trajectory"]],
        "sensor_posteriors": [r["posterior"][0] for r in sensor_run["trajectory"]],
        "negative_efe": sensor_run["trajectory"][0]["negative_efe"],
        "intervention_policies": {
            k: v["run"]["trajectory"][0]["q_pi"] for k, v in controls.items()
        },
    }
    payloads = {
        "upstream": envelope(
            {
                "passive": passive_qualified.receipt,
                "sensor": sensor_qualified.receipt,
            }
        ),
        "passive_model": model_payload(passive),
        "sensor_model": model_payload(sensor),
        "passive_run": envelope(passive_run),
        "sensor_run": envelope(sensor_run),
        "controls": envelope(control_payload),
        "figure_inputs": envelope(figures),
    }
    for name, run in [("passive", passive_run), ("sensor", sensor_run)]:
        payloads[name + "_trajectory"] = envelope(
            {
                k: run[k]
                for k in (
                    "model_digest",
                    "model_source_sha256",
                    "backend",
                    "time_convention",
                    "steps",
                    "trajectory",
                )
            }
        )
    for name, payload in payloads.items():
        write_payload(output_dir / f"{name}.json", payload)
    return payloads
