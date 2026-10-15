"""Independent NumPy finite-state reference; no Agent inference/control imports."""

from __future__ import annotations

import itertools
import math

import numpy as np

from .gnn_model import CompiledModel


def joint_distribution(factors):
    if not factors or math.prod(np.asarray(factor).size for factor in factors) > 4096:
        raise ValueError("GNN_REFERENCE_RESOURCE_LIMIT")
    result = np.array(factors[0], dtype=float, copy=True)
    for factor in factors[1:]:
        result = np.multiply.outer(result, factor)
    return result


def posterior_reference(model: CompiledModel, prior, observation):
    joint = joint_distribution(prior)
    for state in np.ndindex(joint.shape):
        for i, deps in enumerate(model.A_dependencies):
            joint[state] *= model.A[i][
                (observation[i],) + tuple(state[d] for d in deps)
            ]
    mass = joint.sum()
    if mass <= 0:
        raise ValueError("GNN_IMPOSSIBLE_OBSERVATION")
    joint /= mass
    return [
        joint.sum(axis=tuple(j for j in range(joint.ndim) if j != i))
        for i in range(joint.ndim)
    ]


def policy_reference(model: CompiledModel, posterior):
    """Predictive-policy EFE: utility plus modality-wise state information gain."""
    utilities = []
    information = []
    for policy in model.policies:
        beliefs = [np.array(v, dtype=float, copy=True) for v in posterior]
        u = ig = 0.0
        for action in policy:
            beliefs = [model.B[i][:, :, action[i]] @ q for i, q in enumerate(beliefs)]
            for i, deps in enumerate(model.A_dependencies):
                state_probs = joint_distribution([beliefs[d] for d in deps])
                likelihood = model.A[i].reshape(len(model.outcomes[i]), -1)
                weights = state_probs.reshape(-1)
                predicted = likelihood @ weights
                u += float(predicted @ model.C[i])
                for o, s in itertools.product(
                    range(likelihood.shape[0]), range(likelihood.shape[1])
                ):
                    p = likelihood[o, s]
                    if p > 0 and weights[s] > 0 and predicted[o] > 0:
                        ig += float(weights[s] * p * np.log(p / predicted[o]))
        utilities.append(u)
        information.append(ig)
    neg = np.array(utilities) + np.array(information)
    scores = model.gamma * neg + np.where(
        model.E > 0, np.log(np.maximum(model.E, np.finfo(float).tiny)), -np.inf
    )
    pi = np.exp(scores - np.max(scores))
    pi /= pi.sum()
    return {
        "q_pi": pi.tolist(),
        "negative_efe": neg.tolist(),
        "utility": utilities,
        "information_gain": information,
        "action": model.policies[int(np.argmax(pi)), 0].tolist(),
    }


def forward_backward(model: CompiledModel, observations):
    if (
        len(model.factors) != 1
        or len(model.modalities) != 1
        or model.B[0].shape[2] != 1
    ):
        raise ValueError("GNN_NATIVE_UNSUPPORTED_MODEL")
    if not observations:
        raise ValueError("GNN_EMPTY_OBSERVATIONS")
    A, B, D = model.A[0], model.B[0][:, :, 0], model.D[0]
    filtered = []
    q = D.copy()
    for t, o in enumerate(observations):
        if t:
            q = B @ q
        q = A[o] * q
        if q.sum() <= 0:
            raise ValueError("GNN_IMPOSSIBLE_OBSERVATION")
        q = q / q.sum()
        filtered.append(q)
    smooth = [v.copy() for v in filtered]
    beta = np.ones_like(D)
    for t in range(len(observations) - 2, -1, -1):
        beta = B.T @ (A[observations[t + 1]] * beta)
        beta /= beta.sum()
        smooth[t] = filtered[t] * beta
        smooth[t] /= smooth[t].sum()
    return {
        "filtered": [v.tolist() for v in filtered],
        "smoothed": [v.tolist() for v in smooth],
    }
