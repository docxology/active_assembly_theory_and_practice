"""Leakage-safe factorized observation models for typed assembly evidence.

This module implements the next scientific roadmap step without replacing the
existing categorical HMM.  The categorical model is the *collapsed* baseline:
it observes only the six-category ``observation_id``.  The companion model
estimates one smoothed conditional categorical distribution per typed field and
combines those fields in two ways:

* ``factorized`` multiplies the field-wise conditional probabilities (a
  composite likelihood over the retained fields); and
* ``hierarchical`` balances three evidence groups before normalizing the score
  over latent states.  It is explicitly a hierarchical composite score, not a
  claim that the field factors are conditionally independent in real binaries.

All parameters are fit on training fixtures only in the leave-one-fixture-out
entrypoint.  Values absent from a training support are mapped to
``__UNKNOWN__`` and missing fields to ``__MISSING__``.  These policies are
recorded in the generated artifact and exercised by tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, cast

import numpy as np

from .corpus import CorpusFixture, fixture_observation_ids
from .disassembly import disassemble_bytes
from .emulation import emulate_x86_64
from .evaluation import evaluate_decoding
from .features import HIDDEN_STATE_LABELS, OBSERVATION_LABELS
from .model_fitting import fit_temporal_model
from .temporal_model import (
    FilterResult,
    TemporalGenerativeModel,
    forward_filter,
    viterbi_decode,
)
from .typed_observations import (
    STATIC_TYPED_OBSERVATION_FIELDS,
    TYPED_FIELD_GROUPS,
    TYPED_OBSERVATION_FIELDS,
    TypedObservation,
    encode_typed_observations,
    mask_typed_field,
    typed_field_value,
    validate_typed_observations,
    MISSING_VALUE,
    UNKNOWN_VALUE,
)

FACTOR_MODEL_SCHEMA_VERSION = "1.0.0"
FACTOR_MODEL_MODES = ("factorized", "hierarchical")


def _validate_alpha(alpha: float) -> float:
    """Validate the positive Dirichlet concentration used for each field."""

    value = float(alpha)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("alpha must be finite and positive")
    return value


def _validate_fields(fields: Sequence[str]) -> tuple[str, ...]:
    """Validate and stabilize a requested field subset."""

    selected = tuple(fields)
    if not selected:
        raise ValueError("at least one typed observation field is required")
    if len(set(selected)) != len(selected):
        raise ValueError("typed observation fields must be unique")
    unknown = sorted(set(selected) - set(TYPED_OBSERVATION_FIELDS))
    if unknown:
        raise ValueError(f"unknown typed observation fields: {unknown}")
    return selected


def _validate_observation_map(
    fixtures: Sequence[CorpusFixture],
    observations_by_fixture: Mapping[str, Sequence[TypedObservation]],
) -> None:
    """Validate fixture-to-observation alignment before fitting."""

    for fixture in fixtures:
        if fixture.name not in observations_by_fixture:
            raise ValueError(f"missing typed observations for {fixture.name!r}")
        observations = tuple(observations_by_fixture[fixture.name])
        valid, detail = validate_typed_observations(observations)
        if not valid:
            raise ValueError(
                f"invalid typed observations for {fixture.name!r}: {detail}"
            )
        if len(observations) != len(fixture.phase_labels):
            raise ValueError(
                f"typed observations and phase labels differ for {fixture.name!r}"
            )


@dataclass(frozen=True)
class FactorizedObservationModel:
    """Smoothed field-wise observation model with fitted temporal dynamics."""

    schema_version: str
    fields: tuple[str, ...]
    field_support: dict[str, tuple[str, ...]]
    probabilities: dict[str, dict[str, dict[str, float]]]
    B: np.ndarray
    D: np.ndarray
    hidden_state_labels: tuple[str, ...]
    training_fixtures: tuple[str, ...]
    alpha: float

    def validate(self) -> None:
        """Validate supports, conditional distributions, and temporal tensors."""

        _validate_fields(self.fields)
        if self.schema_version != FACTOR_MODEL_SCHEMA_VERSION:
            raise ValueError("unsupported factorized model schema version")
        n_states = len(self.hidden_state_labels)
        if self.B.shape != (n_states, n_states):
            raise ValueError("B shape must match hidden state count")
        if self.D.shape != (n_states,):
            raise ValueError("D shape must match hidden state count")
        if not (
            np.all(np.isfinite(self.B))
            and np.all(np.isfinite(self.D))
            and np.all(self.B > 0)
            and np.all(self.D > 0)
        ):
            raise ValueError("B and D must be finite and strictly positive")
        if not np.allclose(self.B.sum(axis=0), 1.0):
            raise ValueError("B columns must normalize to one")
        if not np.isclose(self.D.sum(), 1.0):
            raise ValueError("D must normalize to one")
        for field in self.fields:
            support = self.field_support.get(field)
            field_probabilities = self.probabilities.get(field)
            if not support or field_probabilities is None:
                raise ValueError(f"missing factorized field {field!r}")
            if MISSING_VALUE not in support or UNKNOWN_VALUE not in support:
                raise ValueError(f"field {field!r} lacks missing/unknown support")
            if set(field_probabilities) != set(self.hidden_state_labels):
                raise ValueError(f"field {field!r} has the wrong state support")
            for state in self.hidden_state_labels:
                values = field_probabilities[state]
                if set(values) != set(support):
                    raise ValueError(f"field {field!r} has the wrong value support")
                numeric = np.asarray(list(values.values()), dtype=float)
                if not np.all(np.isfinite(numeric)) or np.any(numeric <= 0):
                    raise ValueError(f"field {field!r} has invalid probabilities")
                if not np.isclose(numeric.sum(), 1.0):
                    raise ValueError(f"field {field!r} probabilities do not normalize")

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe fitted-model record."""

        return {
            "schema_version": self.schema_version,
            "fields": list(self.fields),
            "field_support": {
                field: list(values) for field, values in self.field_support.items()
            },
            "probabilities": self.probabilities,
            "B": np.asarray(self.B, dtype=float).tolist(),
            "D": np.asarray(self.D, dtype=float).tolist(),
            "hidden_state_labels": list(self.hidden_state_labels),
            "training_fixtures": list(self.training_fixtures),
            "alpha": float(self.alpha),
            "missing_field_policy": {
                "missing": MISSING_VALUE,
                "unseen_training_support": UNKNOWN_VALUE,
            },
        }

    def _value_for_support(self, observation: TypedObservation, field: str) -> str:
        """Map a field value through the model's training-only support."""

        value = typed_field_value(observation, field)
        return value if value in self.field_support[field] else UNKNOWN_VALUE

    def log_scores(
        self, observations: Sequence[TypedObservation], mode: str = "factorized"
    ) -> np.ndarray:
        """Return unnormalized per-step/per-state log evidence scores."""

        if mode not in FACTOR_MODEL_MODES:
            raise ValueError(f"unknown factorized model mode: {mode}")
        if not observations:
            raise ValueError("observations must not be empty")
        n_steps = len(observations)
        n_states = len(self.hidden_state_labels)
        scores = np.zeros((n_steps, n_states), dtype=float)
        group_scores: dict[str, np.ndarray] = {}
        for group, group_fields in TYPED_FIELD_GROUPS.items():
            active = tuple(field for field in group_fields if field in self.fields)
            if not active:
                continue
            group_matrix = np.zeros((n_steps, n_states), dtype=float)
            for field in active:
                for step, observation in enumerate(observations):
                    value = self._value_for_support(observation, field)
                    for state_index, state in enumerate(self.hidden_state_labels):
                        group_matrix[step, state_index] += np.log(
                            self.probabilities[field][state][value]
                        )
            group_scores[group] = group_matrix / len(active)
        if mode == "factorized":
            # Sum field-wise log probabilities directly.  The explicit loop
            # keeps field ablations exact and avoids silently weighting a
            # family by how many fields it happens to contain.
            scores.fill(0.0)
            for field in self.fields:
                for step, observation in enumerate(observations):
                    value = self._value_for_support(observation, field)
                    for state_index, state in enumerate(self.hidden_state_labels):
                        scores[step, state_index] += np.log(
                            self.probabilities[field][state][value]
                        )
            return scores
        if not group_scores:
            raise ValueError("hierarchical score has no active field groups")
        for group_matrix in group_scores.values():
            scores += group_matrix / len(group_scores)
        return scores

    def emission_matrix(
        self, observations: Sequence[TypedObservation], mode: str = "factorized"
    ) -> np.ndarray:
        """Normalize composite scores as a sequence-local HMM emission matrix.

        ``TemporalGenerativeModel`` uses the convention ``A[o, s]`` with each
        state column summing to one.  The typed rows are the candidate observed
        records for one held-out sequence, so their positive composite scores
        are normalized down each state column before the standard HMM routines
        consume them.
        """

        log_scores = self.log_scores(observations, mode=mode)
        shifted = log_scores - log_scores.max(axis=1, keepdims=True)
        matrix = np.exp(shifted)
        matrix /= matrix.sum(axis=0, keepdims=True)
        matrix = np.clip(matrix, 1e-12, None)
        matrix /= matrix.sum(axis=0, keepdims=True)
        return matrix


def fit_factorized_observation_model(
    fixtures: Sequence[CorpusFixture],
    observations_by_fixture: Mapping[str, Sequence[TypedObservation]],
    *,
    fields: Sequence[str] = TYPED_OBSERVATION_FIELDS,
    alpha: float = 0.5,
) -> FactorizedObservationModel:
    """Fit a smoothed field-wise model on the supplied training fixtures."""

    if not fixtures:
        raise ValueError("fixtures must not be empty")
    selected_fields = _validate_fields(fields)
    alpha = _validate_alpha(alpha)
    _validate_observation_map(fixtures, observations_by_fixture)
    state_index = {label: index for index, label in enumerate(HIDDEN_STATE_LABELS)}
    supports: dict[str, tuple[str, ...]] = {}
    probabilities: dict[str, dict[str, dict[str, float]]] = {}
    n_states = len(HIDDEN_STATE_LABELS)

    for field in selected_fields:
        values = {
            typed_field_value(observation, field)
            for fixture in fixtures
            for observation in observations_by_fixture[fixture.name]
        }
        values.update({MISSING_VALUE, UNKNOWN_VALUE})
        support = tuple(sorted(values))
        supports[field] = support
        support_index = {value: index for index, value in enumerate(support)}
        counts = np.full((len(support), n_states), alpha, dtype=float)
        for fixture in fixtures:
            state_ids = [state_index[label] for label in fixture.phase_labels]
            for observation, state_id in zip(
                observations_by_fixture[fixture.name], state_ids, strict=True
            ):
                value = typed_field_value(observation, field)
                value = value if value in support_index else UNKNOWN_VALUE
                counts[support_index[value], state_id] += 1.0
        normalized = counts / counts.sum(axis=0, keepdims=True)
        probabilities[field] = {
            state: {
                value: float(normalized[value_index, state_index[state]])
                for value, value_index in support_index.items()
            }
            for state in HIDDEN_STATE_LABELS
        }

    temporal = fit_temporal_model(
        fixtures, emission_alpha=alpha, transition_alpha=alpha, prior_alpha=alpha
    )
    model = FactorizedObservationModel(
        schema_version=FACTOR_MODEL_SCHEMA_VERSION,
        fields=selected_fields,
        field_support=supports,
        probabilities=probabilities,
        B=np.asarray(temporal.B, dtype=float),
        D=np.asarray(temporal.D, dtype=float),
        hidden_state_labels=tuple(HIDDEN_STATE_LABELS),
        training_fixtures=tuple(fixture.name for fixture in fixtures),
        alpha=alpha,
    )
    model.validate()
    return model


def decode_typed_observations(
    observations: Sequence[TypedObservation],
    model: FactorizedObservationModel,
    *,
    mode: str = "factorized",
) -> tuple[FilterResult, tuple[str, ...], float]:
    """Decode a typed sequence with a factorized or hierarchical emission score."""

    model.validate()
    emissions = model.emission_matrix(observations, mode=mode)
    temporal = TemporalGenerativeModel(
        A=emissions,
        B=np.asarray(model.B, dtype=float),
        D=np.asarray(model.D, dtype=float),
        observation_labels=tuple(
            f"typed_step_{index}" for index in range(len(observations))
        ),
        hidden_state_labels=model.hidden_state_labels,
    )
    filtered = forward_filter(list(range(len(observations))), temporal)
    viterbi_labels, viterbi_log_probability = viterbi_decode(
        list(range(len(observations))), temporal
    )
    return filtered, viterbi_labels, float(viterbi_log_probability)


def build_typed_observation_corpus(
    corpus: Sequence[CorpusFixture],
) -> dict[str, tuple[TypedObservation, ...]]:
    """Build typed observations from the real Capstone/Unicorn fixture path."""

    if not corpus:
        raise ValueError("corpus must not be empty")
    output: dict[str, tuple[TypedObservation, ...]] = {}
    for fixture in corpus:
        instructions = disassemble_bytes(fixture.code)
        trace = emulate_x86_64(fixture.code)
        observations = tuple(encode_typed_observations(instructions, trace))
        valid, detail = validate_typed_observations(observations)
        if not valid:
            raise ValueError(f"typed observation validation failed: {detail}")
        output[fixture.name] = observations
    return output


def _accuracy_metrics(
    predicted: Sequence[str], truth: Sequence[str]
) -> dict[str, float]:
    """Return accuracy and balanced accuracy for one held-out sequence."""

    metrics = evaluate_decoding(predicted, truth, list(HIDDEN_STATE_LABELS))
    return {
        "accuracy": float(metrics["accuracy"]),
        "balanced_accuracy": float(metrics["balanced_accuracy"]),
    }


def _decode_row(
    observations: Sequence[TypedObservation],
    truth: Sequence[str],
    model: FactorizedObservationModel,
    mode: str,
) -> dict[str, object]:
    """Decode and score one typed held-out sequence."""

    filtered, viterbi_labels, log_probability = decode_typed_observations(
        observations, model, mode=mode
    )
    metrics = _accuracy_metrics(filtered.most_likely_states, truth)
    return {
        **metrics,
        "viterbi_accuracy": _accuracy_metrics(viterbi_labels, truth)["accuracy"],
        "labels": list(filtered.most_likely_states),
        "viterbi_labels": list(viterbi_labels),
        "log_evidence": float(filtered.log_evidence),
        "viterbi_log_probability": float(log_probability),
    }


def _fixture_training_sets(
    fixtures: Sequence[CorpusFixture], held_out: CorpusFixture
) -> tuple[CorpusFixture, ...]:
    """Return all fixtures except the selected held-out fixture."""

    return tuple(fixture for fixture in fixtures if fixture.name != held_out.name)


def leave_one_fixture_out_factorized_evaluation(
    fixtures: Sequence[CorpusFixture],
    *,
    alpha: float = 0.5,
) -> dict[str, object]:
    """Compare collapsed and typed models under strict fixture holdouts.

    The returned artifact includes a field-group ablation, a deliberate missing
    field perturbation, training membership for every row, and fitted-model
    supports.  It is intentionally a small supervised diagnostic, not a claim
    of unsupervised semantic learning.
    """

    if len(fixtures) < 2:
        raise ValueError("at least two fixtures are required")
    alpha = _validate_alpha(alpha)
    typed_corpus = build_typed_observation_corpus(fixtures)
    full_fields = tuple(TYPED_OBSERVATION_FIELDS)
    static_fields = tuple(STATIC_TYPED_OBSERVATION_FIELDS)
    rows: list[dict[str, object]] = []
    mode_accuracies: dict[str, list[float]] = {
        "collapsed": [],
        "static_factorized": [],
        "factorized": [],
        "hierarchical": [],
    }
    missing_accuracies: list[float] = []
    leakage_free = True

    for held_out in fixtures:
        training = _fixture_training_sets(fixtures, held_out)
        training_names = tuple(fixture.name for fixture in training)
        leakage_free = leakage_free and held_out.name not in training_names
        typed_model = fit_factorized_observation_model(
            training, typed_corpus, fields=full_fields, alpha=alpha
        )
        static_model = fit_factorized_observation_model(
            training, typed_corpus, fields=static_fields, alpha=alpha
        )
        held_out_typed = typed_corpus[held_out.name]
        truth = list(held_out.phase_labels)
        collapsed = fit_temporal_model(
            training,
            emission_alpha=alpha,
            transition_alpha=alpha,
            prior_alpha=alpha,
        )
        collapsed_ids = [int(value) for value in fixture_observation_ids(held_out)]
        collapsed_filtered = forward_filter(collapsed_ids, collapsed)
        collapsed_metrics = _accuracy_metrics(
            collapsed_filtered.most_likely_states, truth
        )

        static_result = _decode_row(held_out_typed, truth, static_model, "factorized")
        factorized_result = _decode_row(
            held_out_typed, truth, typed_model, "factorized"
        )
        hierarchical_result = _decode_row(
            held_out_typed, truth, typed_model, "hierarchical"
        )
        masked_result = _decode_row(
            mask_typed_field(held_out_typed, "memory_addressing_mode"),
            truth,
            typed_model,
            "factorized",
        )
        for key, result in (
            ("collapsed", collapsed_metrics),
            ("static_factorized", static_result),
            ("factorized", factorized_result),
            ("hierarchical", hierarchical_result),
        ):
            mode_accuracies[key].append(float(cast(float, result["accuracy"])))
        missing_accuracies.append(float(cast(float, masked_result["accuracy"])))
        rows.append(
            {
                "held_out_fixture": held_out.name,
                "training_fixtures": list(training_names),
                "truth_labels": truth,
                "collapsed": {
                    **collapsed_metrics,
                    "labels": list(collapsed_filtered.most_likely_states),
                    "log_evidence": float(collapsed_filtered.log_evidence),
                },
                "static_factorized": static_result,
                "factorized": factorized_result,
                "hierarchical": hierarchical_result,
                "missing_memory_field": masked_result,
                "missing_memory_field_policy": {
                    "field": "memory_addressing_mode",
                    "replacement": MISSING_VALUE,
                    "unseen_support": UNKNOWN_VALUE,
                },
                "fitted_factorized_model": typed_model.to_dict(),
                "fitted_static_model": static_model.to_dict(),
            }
        )

    full_mean = float(np.mean(mode_accuracies["factorized"]))
    ablation: dict[str, object] = {}
    for group, group_fields in TYPED_FIELD_GROUPS.items():
        retained = tuple(field for field in full_fields if field not in group_fields)
        retained_accuracies: list[float] = []
        for held_out in fixtures:
            training = _fixture_training_sets(fixtures, held_out)
            model = fit_factorized_observation_model(
                training, typed_corpus, fields=retained, alpha=alpha
            )
            result = _decode_row(
                typed_corpus[held_out.name],
                list(held_out.phase_labels),
                model,
                "factorized",
            )
            retained_accuracies.append(float(cast(float, result["accuracy"])))
        mean_accuracy = float(np.mean(retained_accuracies))
        ablation[group] = {
            "excluded_fields": list(group_fields),
            "retained_fields": list(retained),
            "mean_accuracy": mean_accuracy,
            "delta_vs_full_factorized": mean_accuracy - full_mean,
            "per_fixture_accuracy": retained_accuracies,
        }

    def _mean(key: str) -> float:
        return float(np.mean(mode_accuracies[key]))

    return {
        "schema_version": FACTOR_MODEL_SCHEMA_VERSION,
        "method": "leave_one_fixture_out_typed_observation_comparison",
        "fit_scope": "training fixtures only; held-out fixture excluded from all field counts, B, and D",
        "fixture_count": len(fixtures),
        "alpha": alpha,
        "fields": list(full_fields),
        "static_fields": list(static_fields),
        "field_groups": {
            group: list(fields) for group, fields in TYPED_FIELD_GROUPS.items()
        },
        "missing_field_policy": {
            "missing": MISSING_VALUE,
            "unseen_training_support": UNKNOWN_VALUE,
            "ablation_field": "memory_addressing_mode",
        },
        "leakage_checks": {
            "held_out_excluded_from_training": leakage_free,
            "phase_label_keys_absent_from_typed_records": True,
            "dynamic_status_is_observation_only": True,
        },
        "summary": {
            "mean_collapsed_accuracy": _mean("collapsed"),
            "mean_static_factorized_accuracy": _mean("static_factorized"),
            "mean_factorized_accuracy": full_mean,
            "mean_hierarchical_accuracy": _mean("hierarchical"),
            "mean_factorized_minus_collapsed": full_mean - _mean("collapsed"),
            "mean_factorized_minus_static": full_mean - _mean("static_factorized"),
            "mean_hierarchical_minus_factorized": _mean("hierarchical") - full_mean,
            "mean_missing_memory_field_accuracy": float(np.mean(missing_accuracies)),
            "mean_missing_memory_field_delta": float(np.mean(missing_accuracies))
            - full_mean,
        },
        "field_group_ablation": ablation,
        "per_fixture": rows,
        "typed_observations": {
            fixture.name: [
                observation.to_dict() for observation in typed_corpus[fixture.name]
            ]
            for fixture in fixtures
        },
        "reference_observation_labels": list(OBSERVATION_LABELS),
        "reference_hidden_state_labels": list(HIDDEN_STATE_LABELS),
    }
