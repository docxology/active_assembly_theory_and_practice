"""Bounded action-conditioned sensing over typed assembly evidence.

The original active-selection module scores already-decoded categories.  That
is a useful expected-free-energy diagnostic, but it does not change which
evidence is acquired.  This module adds the next roadmap step as a deliberately
small sensor experiment: an action chooses one instruction and one typed field;
only after the choice does the environment reveal that field value.

The action policy uses training-only field likelihoods to score candidate
outcomes.  It never receives the held-out fixture's typed value while choosing
an action.  Fixed-order and seeded-random policies are operational baselines;
the oracle policy is an explicitly inadmissible upper-bound reference that is
allowed to inspect the realized value before choosing.  All four policies use
the same held-out fixture, belief update, action budget, and normalized unit
costs.  The experiment is a mechanism demonstration on transparent fixtures,
not evidence of autonomous sensing over arbitrary binaries.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, cast

import numpy as np

from .corpus import CorpusFixture
from .factorized_model import (
    FactorizedObservationModel,
    build_typed_observation_corpus,
    fit_factorized_observation_model,
)
from .features import HIDDEN_STATE_LABELS
from .typed_observations import (
    TYPED_OBSERVATION_FIELDS,
    TypedObservation,
    typed_field_value,
)


ACTION_SENSING_SCHEMA_VERSION = "1.0.0"
ACTION_POLICY_NAMES = ("active", "fixed", "random", "oracle")
ACTION_CONFIDENCE_THRESHOLD = 0.85
ACTION_MAX_STEPS = 8
ACTION_COST_WEIGHT = 0.05
ACTION_COST = 1.0
ACTION_RANDOM_SEED = 417


@dataclass(frozen=True)
class SensingAction:
    """A candidate query over one instruction and one typed evidence field."""

    action_index: int
    step_index: int
    field: str
    cost: float = ACTION_COST

    @property
    def action_id(self) -> str:
        """Return a stable human-readable action identifier."""

        return f"step_{self.step_index}__{self.field}"

    def to_dict(self) -> dict[str, object]:
        """Return the candidate contract without revealing its field value."""

        return {
            "action_index": int(self.action_index),
            "action_id": self.action_id,
            "step_index": int(self.step_index),
            "field": self.field,
            "cost": float(self.cost),
        }


@dataclass(frozen=True)
class _ActionScore:
    """Internal predictive and realized score for one candidate action."""

    expected_information_gain: float
    expected_free_energy: float
    realized_information_gain: float
    posterior: np.ndarray
    revealed_value: str


def _normalize(vector: np.ndarray) -> np.ndarray:
    """Normalize a finite non-negative vector and fail closed on bad input."""

    values = np.asarray(vector, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("probability vector must be non-empty and one-dimensional")
    if not np.all(np.isfinite(values)) or np.any(values < 0.0):
        raise ValueError("probability vector must be finite and non-negative")
    total = float(values.sum())
    if total <= 0.0:
        raise ValueError("probability vector must have positive mass")
    return values / total


def _entropy(belief: np.ndarray) -> float:
    """Return Shannon entropy in nats."""

    normalized = _normalize(belief)
    safe = np.clip(normalized, 1e-12, 1.0)
    return float(-np.sum(safe * np.log(safe)))


def _field_distribution(
    model: FactorizedObservationModel, field: str
) -> tuple[tuple[str, ...], np.ndarray]:
    """Return training-support values and ``P(value | state)`` for a field."""

    if field not in model.fields:
        raise ValueError(f"field {field!r} is not present in the fitted model")
    support = tuple(model.field_support[field])
    likelihood = np.asarray(
        [
            [
                model.probabilities[field][state][value]
                for state in model.hidden_state_labels
            ]
            for value in support
        ],
        dtype=float,
    )
    if likelihood.ndim != 2 or likelihood.shape[1] != len(HIDDEN_STATE_LABELS):
        raise ValueError(f"invalid likelihood shape for field {field!r}")
    return support, likelihood


def _support_value(
    observation: TypedObservation,
    field: str,
    support: Sequence[str],
) -> str:
    """Map a revealed field value through the training-only support."""

    value = typed_field_value(observation, field)
    return value if value in support else "__UNKNOWN__"


def _posterior_for_value(
    belief: np.ndarray, likelihood: np.ndarray, value_index: int
) -> np.ndarray:
    """Apply one categorical sensor outcome to the current state belief."""

    prior = _normalize(belief)
    if value_index < 0 or value_index >= likelihood.shape[0]:
        raise ValueError("sensor outcome index is out of range")
    posterior = prior * likelihood[value_index, :]
    return _normalize(posterior)


def _expected_information_gain(belief: np.ndarray, likelihood: np.ndarray) -> float:
    """Compute mutual information between state and a candidate field value."""

    prior = _normalize(belief)
    prior_entropy = _entropy(prior)
    total = 0.0
    for value_index, row in enumerate(likelihood):
        predictive_probability = float(row @ prior)
        if predictive_probability <= 0.0:
            continue
        posterior = _posterior_for_value(prior, likelihood, value_index)
        total += predictive_probability * (prior_entropy - _entropy(posterior))
    return float(total)


def _score_action(
    action: SensingAction,
    belief: np.ndarray,
    model: FactorizedObservationModel,
    observation: TypedObservation | None = None,
) -> _ActionScore:
    """Score an action predictively, optionally applying a revealed value.

    ``observation=None`` is the pre-action path used by the active policy. It
    returns expected information gain without inspecting the held-out typed
    record. Passing an observation is reserved for post-choice revelation and
    the explicitly inadmissible oracle.
    """

    support, likelihood = _field_distribution(model, action.field)
    expected_gain = _expected_information_gain(belief, likelihood)
    expected_free_energy = -expected_gain + ACTION_COST_WEIGHT * action.cost
    if observation is None:
        return _ActionScore(
            expected_information_gain=float(expected_gain),
            expected_free_energy=float(expected_free_energy),
            realized_information_gain=float("nan"),
            posterior=_normalize(belief),
            revealed_value="__UNREVEALED__",
        )
    revealed_value = _support_value(observation, action.field, support)
    value_index = support.index(revealed_value)
    posterior = _posterior_for_value(belief, likelihood, value_index)
    realized_gain = _entropy(belief) - _entropy(posterior)
    return _ActionScore(
        expected_information_gain=float(expected_gain),
        expected_free_energy=float(expected_free_energy),
        realized_information_gain=float(realized_gain),
        posterior=posterior,
        revealed_value=revealed_value,
    )


def _candidate_actions(
    step_count: int, fields: Sequence[str]
) -> tuple[SensingAction, ...]:
    """Build the complete action pool without reading any field values."""

    if step_count <= 0:
        raise ValueError("step_count must be positive")
    selected_fields = tuple(fields)
    if not selected_fields or len(set(selected_fields)) != len(selected_fields):
        raise ValueError("fields must be a non-empty unique sequence")
    unknown = sorted(set(selected_fields) - set(TYPED_OBSERVATION_FIELDS))
    if unknown:
        raise ValueError(f"unknown sensing fields: {unknown}")
    return tuple(
        SensingAction(action_index=index, step_index=step, field=field)
        for index, (step, field) in enumerate(
            (
                pair
                for step in range(step_count)
                for pair in [(step, field) for field in selected_fields]
            )
        )
    )


def _run_policy(
    observation: Sequence[TypedObservation],
    model: FactorizedObservationModel,
    candidates: Sequence[SensingAction],
    policy: str,
    *,
    threshold: float,
    max_steps: int,
    seed: int,
) -> dict[str, object]:
    """Run one policy while separating selection from evidence revelation."""

    if policy not in ACTION_POLICY_NAMES:
        raise ValueError(f"unknown action-sensing policy: {policy}")
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0, 1]")
    if max_steps <= 0:
        raise ValueError("max_steps must be positive")
    if not candidates:
        raise ValueError("candidates must not be empty")
    if max(action.step_index for action in candidates) >= len(observation):
        raise ValueError("candidate step exceeds the observation sequence")

    model.validate()
    belief = _normalize(np.asarray(model.D, dtype=float))
    remaining = list(candidates)
    rng = np.random.default_rng(seed)
    random_order = list(rng.permutation(len(remaining)))
    actions: list[dict[str, object]] = []
    initial_confidence = float(np.max(belief))

    while remaining and len(actions) < max_steps and float(np.max(belief)) < threshold:
        if policy == "fixed":
            chosen = remaining[0]
        elif policy == "random":
            chosen_index = random_order[len(actions)]
            chosen = next(
                action for action in remaining if action.action_index == chosen_index
            )
        else:
            scored = [
                (
                    action,
                    _score_action(action, belief, model),
                )
                for action in remaining
            ]
            if policy == "active":
                chosen, chosen_score = min(
                    scored,
                    key=lambda pair: (
                        pair[1].expected_free_energy,
                        pair[0].action_index,
                    ),
                )
            else:
                # The oracle is intentionally inadmissible: it may inspect the
                # realized field value before choosing, and is only an upper
                # bound for the action-conditioned experiment.
                chosen, chosen_score = max(
                    [
                        (
                            action,
                            _score_action(
                                action,
                                belief,
                                model,
                                observation[action.step_index],
                            ),
                        )
                        for action in remaining
                    ],
                    key=lambda pair: (
                        pair[1].realized_information_gain,
                        -pair[0].action_index,
                    ),
                )
            del chosen_score

        chosen_score = _score_action(
            chosen, belief, model, observation[chosen.step_index]
        )
        previous_entropy = _entropy(belief)
        belief = chosen_score.posterior
        actions.append(
            {
                "action_index": int(chosen.action_index),
                "action_id": chosen.action_id,
                "step_index": int(chosen.step_index),
                "field": chosen.field,
                "cost": float(chosen.cost),
                "expected_information_gain": float(
                    chosen_score.expected_information_gain
                ),
                "expected_free_energy": float(chosen_score.expected_free_energy),
                "realized_information_gain": float(
                    chosen_score.realized_information_gain
                ),
                "revealed_value": chosen_score.revealed_value,
                "posterior_entropy": float(_entropy(belief)),
                "posterior_confidence": float(np.max(belief)),
                "posterior": [float(value) for value in belief],
            }
        )
        if not np.isfinite(previous_entropy):
            raise ValueError("non-finite pre-action entropy")
        remaining = [
            action for action in remaining if action.action_index != chosen.action_index
        ]

    final_confidence = float(np.max(belief))
    reached = final_confidence >= threshold
    return {
        "policy": policy,
        "seed": int(seed),
        "threshold": float(threshold),
        "max_steps": int(max_steps),
        "initial_confidence": initial_confidence,
        "final_confidence": final_confidence,
        "reached_confidence": bool(reached),
        "actions_used": len(actions),
        "actions_to_confidence": len(actions) if reached else None,
        "total_cost": float(sum(float(cast(Any, row["cost"])) for row in actions)),
        "selected_action_ids": [str(row["action_id"]) for row in actions],
        "actions": actions,
    }


def action_conditioned_sensing_payload(
    fixtures: Sequence[CorpusFixture],
    *,
    fields: Sequence[str] = TYPED_OBSERVATION_FIELDS,
    threshold: float = ACTION_CONFIDENCE_THRESHOLD,
    max_steps: int = ACTION_MAX_STEPS,
    random_seed: int = ACTION_RANDOM_SEED,
) -> dict[str, object]:
    """Evaluate active sensing with strict held-out fixture training.

    The active policy chooses among typed field queries from predictive field
    likelihoods.  The fixed and random policies use the same candidate pool;
    the oracle is marked as an inadmissible reference.  Every row records the
    training complement, action pool, revealed values after selection, and
    policy-specific confidence/cost outcomes.
    """

    corpus = tuple(fixtures)
    if len(corpus) < 2:
        raise ValueError("at least two fixtures are required")
    selected_fields = tuple(fields)
    typed_corpus = build_typed_observation_corpus(corpus)
    per_fixture: list[dict[str, object]] = []
    for held_out_index, held_out in enumerate(corpus):
        training = tuple(
            fixture for index, fixture in enumerate(corpus) if index != held_out_index
        )
        training_observations = {
            fixture.name: typed_corpus[fixture.name] for fixture in training
        }
        model = fit_factorized_observation_model(
            training,
            training_observations,
            fields=selected_fields,
        )
        candidates = _candidate_actions(
            len(typed_corpus[held_out.name]), selected_fields
        )
        policies = {
            policy: _run_policy(
                typed_corpus[held_out.name],
                model,
                candidates,
                policy,
                threshold=threshold,
                max_steps=max_steps,
                seed=random_seed + held_out_index,
            )
            for policy in ACTION_POLICY_NAMES
        }
        per_fixture.append(
            {
                "held_out_fixture": held_out.name,
                "training_fixtures": [fixture.name for fixture in training],
                "fields": list(selected_fields),
                "action_candidates": [action.to_dict() for action in candidates],
                "policies": policies,
                "leakage_free": held_out.name not in model.training_fixtures,
                "training_model": model.to_dict(),
            }
        )

    def _mean(policy: str, key: str) -> float:
        return float(
            np.mean(
                [float(row["policies"][policy][key]) for row in per_fixture]  # type: ignore[index]
            )
        )

    confidence_rates = {
        policy: float(
            np.mean(
                [
                    bool(row["policies"][policy]["reached_confidence"])  # type: ignore[index]
                    for row in per_fixture
                ]
            )
        )
        for policy in ACTION_POLICY_NAMES
    }
    active_actions = [
        int(row["policies"]["active"]["actions_used"])  # type: ignore[index]
        for row in per_fixture
    ]
    fixed_actions = [
        int(row["policies"]["fixed"]["actions_used"])  # type: ignore[index]
        for row in per_fixture
    ]
    return {
        "schema_version": ACTION_SENSING_SCHEMA_VERSION,
        "method": "bounded_action_conditioned_typed_field_sensing",
        "fields": list(selected_fields),
        "policies": list(ACTION_POLICY_NAMES),
        "confidence_threshold": float(threshold),
        "max_steps": int(max_steps),
        "cost": {
            "unit": "normalized action units",
            "per_action": ACTION_COST,
            "weight_in_expected_free_energy": ACTION_COST_WEIGHT,
            "interpretation": "equal proxy cost for this demonstrator; not a wall-clock measurement",
        },
        "selection_contract": {
            "active": "select from predictive field likelihoods before reading the held-out field value",
            "fixed": "reveal candidates in source step/field order",
            "random": "reveal a seeded permutation of the same candidate pool",
            "oracle": "inadmissible upper-bound reference that reads the realized value before selecting",
        },
        "per_fixture": per_fixture,
        "summary": {
            "fixture_count": len(per_fixture),
            "field_count": len(selected_fields),
            "mean_actions_used": {
                policy: _mean(policy, "actions_used") for policy in ACTION_POLICY_NAMES
            },
            "mean_total_cost": {
                policy: _mean(policy, "total_cost") for policy in ACTION_POLICY_NAMES
            },
            "mean_final_confidence": {
                policy: _mean(policy, "final_confidence")
                for policy in ACTION_POLICY_NAMES
            },
            "confidence_rate": confidence_rates,
            "active_minus_fixed_actions": float(
                np.mean(np.asarray(active_actions) - np.asarray(fixed_actions))
            ),
            "active_not_worse_count": int(
                sum(
                    active <= fixed
                    for active, fixed in zip(active_actions, fixed_actions, strict=True)
                )
            ),
            "action_pool_changes_evidence": bool(
                any(
                    len(
                        {
                            str(action["field"]) + "=" + str(action["revealed_value"])
                            for action in row["policies"]["active"]["actions"]  # type: ignore[index]
                        }
                    )
                    > 1
                    for row in per_fixture
                )
            ),
            "leakage_free": all(bool(row["leakage_free"]) for row in per_fixture),
        },
        "interpretation": (
            "This is a bounded sensor-choice mechanism test. Active selection is "
            "evaluated against fixed, seeded-random, and explicitly inadmissible "
            "oracle policies on held-out transparent fixtures; it is not evidence "
            "of autonomous sensing over arbitrary binaries."
        ),
    }


def validate_action_conditioned_sensing_payload(
    payload: dict[str, object],
) -> tuple[bool, str]:
    """Validate the action-sensing artifact contract without recomputation."""

    if payload.get("schema_version") != ACTION_SENSING_SCHEMA_VERSION:
        return False, "unsupported action-sensing schema version"
    fields = payload.get("fields")
    rows = payload.get("per_fixture")
    summary = payload.get("summary")
    if (
        not isinstance(fields, list)
        or not fields
        or not isinstance(rows, list)
        or not rows
        or not isinstance(summary, dict)
    ):
        return False, "action-sensing fields, rows, or summary missing"
    if summary.get("fixture_count") != len(rows) or summary.get("field_count") != len(
        fields
    ):
        return False, "action-sensing summary does not match rows"
    for row in rows:
        if not isinstance(row, dict):
            return False, "action-sensing row is not an object"
        held_out = row.get("held_out_fixture")
        training = row.get("training_fixtures")
        candidates = row.get("action_candidates")
        policies = row.get("policies")
        if (
            not isinstance(held_out, str)
            or not isinstance(training, list)
            or held_out in training
            or not isinstance(candidates, list)
            or not isinstance(policies, dict)
            or set(policies) != set(ACTION_POLICY_NAMES)
        ):
            return False, "action-sensing leakage or policy contract failed"
        candidate_ids = {
            str(candidate.get("action_id"))
            for candidate in candidates
            if isinstance(candidate, dict)
        }
        if len(candidate_ids) != len(candidates):
            return False, "action candidates are not unique"
        for policy in ACTION_POLICY_NAMES:
            result = policies[policy]
            if not isinstance(result, dict):
                return False, "policy result is not an object"
            actions = result.get("actions")
            max_steps = int(cast(Any, payload["max_steps"]))
            if not isinstance(actions, list) or len(actions) > max_steps:
                return False, "policy action count exceeds budget"
            if any(
                not isinstance(action, dict)
                or action.get("action_id") not in candidate_ids
                or not np.isfinite(float(cast(Any, action.get("expected_free_energy"))))
                or not np.isfinite(
                    float(cast(Any, action.get("realized_information_gain")))
                )
                for action in actions
            ):
                return False, "policy action row is invalid"
    return True, f"fixtures={len(rows)} policies={len(ACTION_POLICY_NAMES)}"
