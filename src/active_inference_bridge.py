"""pymdp-backed variational updates over assembly observations."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from pymdp.maths import nn

from .features import AssemblyObservation
from .generative_model import GenerativeModelSpec


@dataclass(frozen=True)
class InferenceResult:
    """Posterior beliefs and free-energy diagnostics for an observation stream."""

    posterior_by_step: tuple[tuple[float, ...], ...]
    free_energy_by_step: tuple[float, ...]
    most_likely_states: tuple[str, ...]
    final_posterior: tuple[float, ...]
    final_state: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation."""

        data = asdict(self)
        data["posterior_by_step"] = [list(row) for row in self.posterior_by_step]
        data["free_energy_by_step"] = list(self.free_energy_by_step)
        data["most_likely_states"] = list(self.most_likely_states)
        data["final_posterior"] = list(self.final_posterior)
        return data


def _variational_free_energy(logits: np.ndarray, posterior: np.ndarray) -> float:
    """Compute a finite categorical variational free-energy diagnostic."""

    safe_posterior = np.clip(posterior, 1e-12, 1.0)
    return float(np.sum(safe_posterior * (np.log(safe_posterior) - logits)))


def infer_hidden_states_reference(
    observations: list[AssemblyObservation],
    model: GenerativeModelSpec,
) -> InferenceResult:
    """Infer latent program-state classes from categorical observations."""

    if not observations:
        raise ValueError("observations must not be empty")
    model.validate()

    posterior = np.asarray(model.D, dtype=float)
    posterior_by_step: list[tuple[float, ...]] = []
    free_energy_by_step: list[float] = []
    most_likely_states: list[str] = []

    for observation in observations:
        log_prior = np.log(np.clip(posterior, 1e-12, 1.0))
        log_likelihood = np.log(
            np.clip(model.A[observation.observation_id, :], 1e-12, 1.0)
        )
        logits = log_prior + log_likelihood
        posterior = np.asarray(nn.softmax(logits), dtype=float)
        posterior = posterior / posterior.sum()
        posterior_by_step.append(tuple(float(value) for value in posterior))
        free_energy_by_step.append(_variational_free_energy(logits, posterior))
        most_likely_states.append(model.hidden_state_labels[int(np.argmax(posterior))])

    final_state = most_likely_states[-1]
    return InferenceResult(
        posterior_by_step=tuple(posterior_by_step),
        free_energy_by_step=tuple(free_energy_by_step),
        most_likely_states=tuple(most_likely_states),
        final_posterior=tuple(float(value) for value in posterior),
        final_state=final_state,
    )


def infer_hidden_states(
    observations: list[AssemblyObservation], model: GenerativeModelSpec
) -> InferenceResult:
    """Real Agent execution of the historical identity-transition reference.

    This retained static comparator repeatedly conditions one hypothesis;
    the canonical temporal GNN study instead uses its specified transitions.
    """
    if not observations:
        raise ValueError("observations must not be empty")
    model.validate()
    import hashlib
    import json
    from dataclasses import replace
    from pathlib import Path

    from .gnn_parser import compile_gnn
    from .pymdp_runtime import PymdpRuntime, beliefs

    canonical = compile_gnn(
        Path(__file__).resolve().parents[1] / "configs/gnn/passive_phase.md"
    )
    compiled = replace(
        canonical,
        A=(model.A,),
        D=(model.D,),
        B=(np.eye(len(model.D))[:, :, None],),
        source_sha256=hashlib.sha256(
            json.dumps(
                {"A": model.A.tolist(), "D": model.D.tolist()}, sort_keys=True
            ).encode()
        ).hexdigest(),
    )
    runtime = PymdpRuntime(compiled)
    rows = []
    energy = []
    labels = []
    for observation in observations:
        prior = np.asarray(runtime.prior[0])[0].astype(float)
        qs = runtime.infer((observation.observation_id,))
        posterior = np.array(beliefs(qs)[0])
        pi, _ = runtime.agent.infer_policies(qs)
        action = runtime.agent.sample_action(pi)
        logits = np.log(np.clip(prior, 1e-12, 1.0)) + np.log(
            np.clip(model.A[observation.observation_id], 1e-12, 1.0)
        )
        rows.append(tuple(float(v) for v in posterior))
        energy.append(_variational_free_energy(logits, posterior))
        labels.append(model.hidden_state_labels[int(np.argmax(posterior))])
        runtime.prior = runtime.agent.update_empirical_prior(action, qs)
        runtime.assert_unchanged()
    return InferenceResult(
        posterior_by_step=tuple(rows),
        free_energy_by_step=tuple(energy),
        most_likely_states=tuple(labels),
        final_posterior=rows[-1],
        final_state=labels[-1],
    )
