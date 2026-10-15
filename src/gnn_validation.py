from __future__ import annotations

import hashlib
import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path
import subprocess

import numpy as np

from .corpus import build_corpus
from .disassembly import disassemble_bytes
from .features import categorize_instruction
from .fixture_sensor_environment import FixtureSensorEnvironment
from .gnn_experiments import PROTOCOL, control_variants, source_hashes
from .gnn_model import CompiledModel
from .gnn_reference import forward_backward, policy_reference, posterior_reference
from .gnn_upstream import compile_upstream
from .provenance import toolchain_versions
from .receipt import validate_receipt_envelope

ARTIFACTS = (
    "upstream",
    "passive_model",
    "sensor_model",
    "passive_run",
    "sensor_run",
    "passive_trajectory",
    "sensor_trajectory",
    "controls",
    "figure_inputs",
)
ROOT = Path(__file__).resolve().parents[1]


def _require(condition, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _close(actual, expected) -> None:
    _require(np.asarray(actual).dtype.kind in "fiu", "GNN_AUDIT_NONNUMERIC")
    observed = np.asarray(actual, dtype=float)
    reference = np.asarray(expected, dtype=float)
    _require(
        observed.shape == reference.shape
        and np.isfinite(observed).all()
        and np.allclose(observed, reference, atol=1e-5, rtol=0),
        "GNN_AUDIT_NUMERICAL_MISMATCH",
    )


def _error_claim(actual, expected: float) -> None:
    _require(
        type(actual) in (int, float)
        and np.isfinite(actual)
        and abs(actual - expected) < 1e-12
        and expected <= 1e-5,
        "GNN_AUDIT_ERROR_CLAIM",
    )


def _oracle_error(model: CompiledModel, run) -> float:
    error = 0.0
    for row in run["trajectory"]:
        reference = policy_reference(model, [np.array(q) for q in row["posterior"]])
        for field in ("q_pi", "negative_efe"):
            error = max(
                error, float(np.max(np.abs(np.array(row[field]) - reference[field])))
            )
    return error


def _check_oracle_summary(model: CompiledModel, item) -> float:
    row = item["run"]["trajectory"][0]
    reference = policy_reference(model, [np.array(q) for q in row["posterior"]])
    _require(item["oracle"] == reference, "GNN_AUDIT_ORACLE_EXPORT")
    error = _oracle_error(model, item["run"])
    _error_claim(item["oracle_max_error"], error)
    return error


def _check_run(model: CompiledModel, run, fixture: str) -> None:
    _require(
        run["model_digest"] == model.tensor_digest
        and run["model_source_sha256"] == model.source_sha256
        and run["backend"] == "pymdp.Agent"
        and run["backend_version"] == importlib.metadata.version("inferactively-pymdp")
        and run["time_convention"] == "first_observation_D"
        and run["tensors_unchanged"] is True
        and type(run["steps"]) is int
        and 1 <= run["steps"] <= 100
        and len(run["trajectory"]) == run["steps"],
        "GNN_AUDIT_RUN_CONTRACT",
    )
    _require(
        type(run["seed"]) is int and 0 <= run["seed"] < 2**32,
        "GNN_AUDIT_SEED",
    )
    _require(
        run.get("action_selection", "deterministic") in ("deterministic", "stochastic"),
        "GNN_AUDIT_ACTION_SELECTION",
    )
    prior = model.D
    sensor = len(model.factors) == 2
    if sensor:
        environment = FixtureSensorEnvironment(fixture)
        observation = list(environment.reset())
    else:
        owned = next(f for f in build_corpus() if f.name == fixture)
        _require(
            run["fixture_sha256"] == hashlib.sha256(owned.code).hexdigest(),
            "GNN_AUDIT_FIXTURE_HASH",
        )
        observations = [
            [model.outcomes[0].index(categorize_instruction(i))]
            for i in disassemble_bytes(owned.code)
        ]
        _require(len(run["trajectory"]) <= len(observations), "GNN_AUDIT_RUN_BOUND")
    for index, row in enumerate(run["trajectory"]):
        if not sensor:
            observation = observations[index]
        _require(row["observation"] == observation, "GNN_AUDIT_OBSERVATION")
        _require(len(row["prior"]) == len(prior), "GNN_AUDIT_PRIOR")
        for actual, expected in zip(row["prior"], prior, strict=True):
            _close(actual, expected)
        posterior = posterior_reference(model, prior, observation)
        _require(len(row["posterior"]) == len(posterior), "GNN_AUDIT_POSTERIOR")
        for actual, expected in zip(row["posterior"], posterior, strict=True):
            _close(actual, expected)
        reference = policy_reference(model, posterior)
        _close(row["q_pi"], reference["q_pi"])
        _close(row["negative_efe"], reference["negative_efe"])
        _close(row["efe"], -np.array(reference["negative_efe"]))
        action = row["action"]
        _require(
            len(action) == len(model.factors)
            and all(
                type(a) is int and 0 <= a < len(model.actions[i])
                for i, a in enumerate(action)
            ),
            "GNN_AUDIT_ACTION",
        )
        if run.get("action_selection", "deterministic") == "deterministic":
            _require(action == reference["action"], "GNN_AUDIT_ACTION")
        prior = tuple(model.B[i][:, :, a] @ posterior[i] for i, a in enumerate(action))
        if sensor:
            before = len(environment.acquisition_log)
            observation = list(environment.step(action[-1]))
            _require(
                row["decision_before_acquisition"] is True
                and type(row["acquisition_count_before"]) is int
                and type(row["acquisition_count_after"]) is int
                and row["acquisition_count_before"] == before
                and row["acquisition_count_after"] == len(environment.acquisition_log)
                and row["revealed_observation"] == observation,
                "GNN_AUDIT_ACQUISITION",
            )
    final = posterior_reference(model, prior, observation) if sensor else posterior
    for actual, expected in zip(run["final_posterior"], final, strict=True):
        _close(actual, expected)
    if sensor:
        _require(
            run["evaluation"] == environment.evaluation_provenance(),
            "GNN_AUDIT_ACQUISITION_PROVENANCE",
        )
    else:
        exact = forward_backward(
            model, [r["observation"][0] for r in run["trajectory"]]
        )
        _require(run["exact_reference"] == exact, "GNN_AUDIT_SMOOTHING")
        error = float(
            np.max(
                np.abs(
                    np.array([r["posterior"][0] for r in run["trajectory"]])
                    - exact["filtered"]
                )
            )
        )
        _error_claim(run["oracle_max_error"], error)


def validate_gnn_study(
    output_dir: Path, gnn_root: Path, *, require_native: bool = False
) -> list[str]:
    try:
        payloads = {
            name: json.loads((output_dir / f"{name}.json").read_text())
            for name in ARTIFACTS
        }
        models = {
            "passive": compile_upstream(
                ROOT / "configs/gnn/passive_phase.md", gnn_root
            ),
            "sensor": compile_upstream(
                ROOT / "configs/gnn/fixture_sensor.md", gnn_root
            ),
        }
        for name, qualified in models.items():
            exported = CompiledModel.from_dict(payloads[name + "_model"])
            _require(
                exported.to_dict() == qualified.model.to_dict()
                and payloads["upstream"][name] == qualified.receipt,
                "GNN_AUDIT_MODEL_OR_UPSTREAM_DRIFT",
            )
        versions = toolchain_versions(
            ["inferactively-pymdp", "jax", "numpy", "capstone", "unicorn"]
        )
        for name in set(ARTIFACTS) - {"passive_model", "sensor_model"}:
            payload = payloads[name]
            valid, detail = validate_receipt_envelope(
                payload, expected_protocol=PROTOCOL
            )
            _require(valid, detail)
            _require(
                payload["input_hashes"] == source_hashes()
                and payload["toolchain_versions"] == versions
                and payload["status"] == "validated_bounded",
                "GNN_AUDIT_SOURCE_OR_TOOLCHAIN_DRIFT",
            )
        passive, sensor = (models[name].model for name in ("passive", "sensor"))
        native_path = output_dir / "native_comparison.json"
        if require_native:
            from .rxinfer_bridge import (
                JULIA_MANIFEST,
                JULIA_PROJECT,
                validate_backend_versions,
            )
            from .provenance import sha256_path

            native = json.loads(native_path.read_text())
            valid, detail = validate_receipt_envelope(
                native, expected_protocol=PROTOCOL
            )
            _require(valid and native["status"] == "validated_bounded", detail)
            validate_backend_versions(
                native["julia_version"], native["rxinfer_version"]
            )
            _require(
                native["model_digest"] == passive.tensor_digest
                and native["model_source_sha256"] == passive.source_sha256
                and native["backend"] == "RxInfer"
                and native["time_convention"] == "first_observation_D"
                and native["input_hashes"]
                == {
                    **source_hashes(),
                    **{
                        str(p.relative_to(ROOT)): sha256_path(p)
                        for p in (
                            ROOT / "src/rxinfer_bridge.py",
                            JULIA_PROJECT / "src/ActiveAssemblyRxInfer.jl",
                            JULIA_PROJECT / "Project.toml",
                            JULIA_MANIFEST,
                        )
                    },
                },
                "GNN_AUDIT_NATIVE_LINEAGE",
            )
            expected_observations = [
                r["observation"][0] for r in payloads["passive_run"]["trajectory"]
            ]
            _require(
                native["observations"] == expected_observations,
                "GNN_AUDIT_NATIVE_TRACE",
            )
            exact = forward_backward(passive, expected_observations)["smoothed"]
            _close(native["native_smoothed"], exact)
            _close(native["python_smoothed"], exact)
            error = float(np.max(np.abs(np.array(native["native_smoothed"]) - exact)))
            _require(
                native["tolerance"] == 1e-5
                and abs(native["max_abs_error"] - error) < 1e-12,
                "GNN_AUDIT_NATIVE_ERROR",
            )
            raw = json.loads((output_dir / "native_backend/result.json").read_text())
            _require(
                isinstance(raw, dict)
                and set(raw)
                == {
                    "model_digest",
                    "model_source_sha256",
                    "model_file_sha256",
                    "trace_sha256",
                    "time_convention",
                    "backend",
                    "julia_version",
                    "rxinfer_version",
                    "native_smoothed",
                    "free_energy",
                }
                and all(native[key] == value for key, value in raw.items()),
                "GNN_AUDIT_NATIVE_RESULT",
            )
            energy = np.asarray(native["free_energy"])
            _require(
                energy.dtype.kind in "fiu"
                and energy.shape == (8,)
                and np.isfinite(energy).all(),
                "GNN_AUDIT_NATIVE_FREE_ENERGY",
            )
            for name, field in (
                ("model", "model_file_sha256"),
                ("observations", "trace_sha256"),
            ):
                _require(
                    sha256_path(output_dir / f"native_backend/{name}.json")
                    == native[field],
                    "GNN_AUDIT_NATIVE_INPUT",
                )
            exported = CompiledModel.from_dict(
                json.loads((output_dir / "native_backend/model.json").read_text())
            )
            request = json.loads(
                (output_dir / "native_backend/observations.json").read_text()
            )
            _require(
                exported.to_dict() == passive.to_dict()
                and request["observations"] == expected_observations
                and request["model_digest"] == passive.tensor_digest
                and hashlib.sha256(request["tensor_contract_json"].encode()).hexdigest()
                == passive.tensor_digest,
                "GNN_AUDIT_NATIVE_CONTRACT",
            )
        for name, model in (("passive", passive), ("sensor", sensor)):
            run = payloads[name + "_run"]
            _check_run(model, run, "x86_accumulator")
            _require(run["seed"] == model.seed, "GNN_AUDIT_SEED")
            expected_steps = (
                2
                if name == "sensor"
                else len(
                    disassemble_bytes(
                        next(
                            f.code
                            for f in build_corpus()
                            if f.name == "x86_accumulator"
                        )
                    )
                )
            )
            _require(run["steps"] == expected_steps, "GNN_AUDIT_STUDY_BOUND")
            trajectory = payloads[name + "_trajectory"]
            _require(
                all(
                    trajectory[key] == run[key]
                    for key in (
                        "model_digest",
                        "model_source_sha256",
                        "backend",
                        "time_convention",
                        "steps",
                        "trajectory",
                    )
                ),
                "GNN_AUDIT_TRAJECTORY_DRIFT",
            )
        controls = payloads["controls"]
        digests = {"passive": passive.tensor_digest, "sensor": sensor.tensor_digest}
        _require(
            controls["model_digests"] == digests and controls["tolerance"] == 1e-5,
            "GNN_AUDIT_CONTROL_CONTRACT",
        )
        errors = [_oracle_error(sensor, payloads["sensor_run"])]
        variants = control_variants(sensor)
        _require(
            set(controls["controls"]) == set(variants), "GNN_AUDIT_CONTROL_INVENTORY"
        )
        original = payloads["sensor_run"]["trajectory"][0]
        for name, expected in variants.items():
            control = controls["controls"][name]
            model = CompiledModel.from_dict(control["model"])
            _require(
                model.tensor_digest == expected.tensor_digest, "GNN_AUDIT_CONTROL_MODEL"
            )
            _check_run(model, control["run"], "x86_accumulator")
            _require(
                control["run"]["steps"] == 1 and control["run"]["seed"] == model.seed,
                "GNN_AUDIT_CONTROL_RUN",
            )
            errors.append(_check_oracle_summary(model, control))
            delta = float(
                np.max(
                    np.abs(
                        np.array(control["run"]["trajectory"][0]["q_pi"])
                        - original["q_pi"]
                    )
                )
            )
            _require(
                delta > 1e-4
                and abs(delta - control["policy_delta"]) < 1e-10
                and control["passed"] is True,
                "GNN_AUDIT_INEFFECTIVE_CONTROL",
            )
        withheld = controls["withheld_invariance"]
        _check_run(sensor, withheld["bitwise_run"], "x86_bitwise")
        _require(
            withheld["bitwise_run"]["steps"] == 2
            and withheld["bitwise_run"]["seed"] == sensor.seed,
            "GNN_AUDIT_WITHHELD_RUN",
        )
        errors.append(_oracle_error(sensor, withheld["bitwise_run"]))
        _require(
            withheld["passed"] is True
            and withheld["revealed_posterior_diverges"] is True
            and withheld["accumulator"] == original
            and withheld["accumulator"]["q_pi"] == withheld["bitwise"]["q_pi"]
            and withheld["bitwise"] == withheld["bitwise_run"]["trajectory"][0]
            and payloads["sensor_run"]["final_posterior"][0]
            != withheld["bitwise_run"]["final_posterior"][0]
            and withheld["accumulator"]["revealed_observation"]
            != withheld["bitwise"]["revealed_observation"],
            "GNN_AUDIT_WITHHELD_LEAK",
        )
        horizon = controls["horizon_sensitivity"]
        model = CompiledModel.from_dict(horizon["model"])
        _require(
            horizon["horizon"] == model.horizon == 2 and model.policy_count == 9,
            "GNN_AUDIT_HORIZON",
        )
        _require(
            model.tensor_digest
            == replace(sensor, horizon=2, E=np.full(9, 1 / 9)).tensor_digest,
            "GNN_AUDIT_HORIZON_MODEL",
        )
        _check_run(model, horizon["run"], "x86_accumulator")
        _require(
            horizon["run"]["steps"] == 1 and horizon["run"]["seed"] == model.seed,
            "GNN_AUDIT_HORIZON_RUN",
        )
        errors.append(_oracle_error(model, horizon["run"]))
        baselines = controls["baselines"]
        _require(
            set(baselines) == {"fixed_idle", "fixed_static", "random_uniform_policy"},
            "GNN_AUDIT_BASELINE_INVENTORY",
        )
        for name, baseline in baselines.items():
            model = CompiledModel.from_dict(baseline["model"])
            if name == "random_uniform_policy":
                reference = policy_reference(sensor, sensor.D)
                log_e = -sensor.gamma * np.array(reference["negative_efe"])
                expected_e = np.exp(log_e - log_e.max())
                expected_e /= expected_e.sum()
            else:
                expected_e = np.full(3, 1e-6)
                expected_e[0 if name == "fixed_idle" else 1] = 1.0 - 2e-6
            _require(
                model.tensor_digest == replace(sensor, E=expected_e).tensor_digest,
                "GNN_AUDIT_BASELINE_MODEL",
            )
            _check_run(model, baseline["run"], "x86_accumulator")
            _require(
                baseline["run"]["steps"] == 1
                and baseline["run"]["seed"]
                == (11 if name == "random_uniform_policy" else 0),
                "GNN_AUDIT_BASELINE_RUN",
            )
            errors.append(_check_oracle_summary(model, baseline))
            row = baseline["run"]["trajectory"][0]
            if name == "random_uniform_policy":
                _close(row["q_pi"], np.full(3, 1 / 3))
                _require(
                    baseline["run"]["action_selection"] == "stochastic",
                    "GNN_AUDIT_BASELINE",
                )
            else:
                _require(
                    row["action"] == [0, 0 if name == "fixed_idle" else 1],
                    "GNN_AUDIT_BASELINE",
                )
        figures = payloads["figure_inputs"]
        _error_claim(controls["oracle_max_error"], max(errors))
        _require(
            figures["model_digests"] == digests
            and figures["sensor_actions"]
            == [r["action"] for r in payloads["sensor_run"]["trajectory"]]
            and figures["sensor_posteriors"]
            == [r["posterior"][0] for r in payloads["sensor_run"]["trajectory"]]
            and figures["passive_posterior"]
            == [r["posterior"][0] for r in payloads["passive_run"]["trajectory"]]
            and figures["passive_smoothed"]
            == payloads["passive_run"]["exact_reference"]["smoothed"]
            and figures["sensor_policies"] == original["q_pi"]
            and figures["negative_efe"] == original["negative_efe"]
            and figures["intervention_policies"]
            == {
                k: c["run"]["trajectory"][0]["q_pi"]
                for k, c in controls["controls"].items()
            },
            "GNN_AUDIT_FIGURE_INPUT_DRIFT",
        )
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        IndexError,
        StopIteration,
        RuntimeError,
        subprocess.TimeoutExpired,
    ) as exc:
        return [f"{type(exc).__name__}: {exc}"]
    return []
