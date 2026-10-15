"""Validated finite categorical model contract shared by executable backends."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


class GNNError(ValueError):
    """Stable compiler error with source or tensor context."""


@dataclass(frozen=True)
class CompiledModel:
    name: str
    factors: tuple[str, ...]
    modalities: tuple[str, ...]
    state_labels: tuple[tuple[str, ...], ...]
    actions: tuple[tuple[str, ...], ...]
    outcomes: tuple[tuple[str, ...], ...]
    A: tuple[np.ndarray, ...]
    B: tuple[np.ndarray, ...]
    C: tuple[np.ndarray, ...]
    D: tuple[np.ndarray, ...]
    E: np.ndarray
    A_dependencies: tuple[tuple[int, ...], ...]
    horizon: int
    gamma: float
    seed: int
    source_sha256: str
    annotations: str = ""

    def __post_init__(self) -> None:
        def names(values: tuple[str, ...]) -> bool:
            return (
                isinstance(values, tuple)
                and bool(values)
                and all(
                    isinstance(v, str) and v.strip() == v and bool(v) for v in values
                )
                and len(set(values)) == len(values)
            )

        if (
            not isinstance(self.name, str)
            or not self.name.strip()
            or not isinstance(self.annotations, str)
            or not names(self.factors)
            or not names(self.modalities)
            or not all(
                names(v) for v in (*self.actions, *self.outcomes, *self.state_labels)
            )
        ):
            raise GNNError("GNN-E102: invalid or duplicate names")
        if (
            type(self.horizon) is not int
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**32
            or type(self.gamma) not in (int, float)
        ):
            raise GNNError("GNN-E102: invalid horizon, seed or gamma type")
        if (
            not isinstance(self.source_sha256, str)
            or len(self.source_sha256) != 64
            or any(c not in "0123456789abcdef" for c in self.source_sha256)
        ):
            raise GNNError("GNN-E102: invalid source hash")
        f, m = len(self.factors), len(self.modalities)
        if not (
            f
            and m
            and len(self.B)
            == len(self.D)
            == len(self.actions)
            == len(self.state_labels)
            == f
            and len(self.A)
            == len(self.C)
            == len(self.outcomes)
            == len(self.A_dependencies)
            == m
        ):
            raise GNNError("GNN-E102: factor/modality contract mismatch")
        if not 1 <= self.horizon <= 3 or self.policy_count > 256:
            raise GNNError("GNN-E103: horizon or policy resource limit")
        if not np.isfinite(self.gamma) or self.gamma <= 0:
            raise GNNError("GNN-E102: gamma must be finite and positive")
        for group in ("A", "B", "C", "D"):
            arrays = []
            for index, value in enumerate(getattr(self, group)):
                if np.asarray(value).dtype.kind not in "fiu":
                    raise GNNError(
                        f"GNN-E102: {group}{index} must contain real numbers"
                    )
                a = np.array(value, dtype=float, copy=True)
                if not 1 <= a.ndim <= 4 or a.size > 4096 or not a.size:
                    raise GNNError(f"GNN-E103: {group}{index} tensor resource limit")
                if not np.isfinite(a).all():
                    raise GNNError(f"GNN-E101: {group}{index} nonfinite")
                if group != "C" and (
                    np.any(a < 0)
                    or not np.allclose(a.sum(axis=0), 1, atol=1e-10, rtol=0)
                ):
                    raise GNNError(f"GNN-E101: {group}{index} invalid distribution")
                a.setflags(write=False)
                arrays.append(a)
            object.__setattr__(self, group, tuple(arrays))
        for i in range(f):
            n = self.D[i].size
            if (
                len(self.state_labels[i]) != n
                or self.D[i].shape != (n,)
                or self.B[i].shape != (n, n, len(self.actions[i]))
                or not self.actions[i]
            ):
                raise GNNError(f"GNN-E102: B{i}/D{i} axes mismatch")
        for i, deps in enumerate(self.A_dependencies):
            if (
                not deps
                or len(set(deps)) != len(deps)
                or any(type(d) is not int or d < 0 or d >= f for d in deps)
            ):
                raise GNNError("GNN-E102: invalid A dependencies")
            shape = (len(self.outcomes[i]), *(self.D[d].size for d in deps))
            if self.A[i].shape != shape or self.C[i].shape != (shape[0],):
                raise GNNError(f"GNN-E102: A{i}/C{i} axes mismatch")
        if np.asarray(self.E).dtype.kind not in "fiu":
            raise GNNError("GNN-E102: E must contain real numbers")
        e = np.array(self.E, dtype=float, copy=True)
        if (
            e.shape != (self.policy_count,)
            or not np.isfinite(e).all()
            or np.any(e < 0)
            or not np.isclose(e.sum(), 1, atol=1e-10, rtol=0)
        ):
            raise GNNError("GNN-E101: invalid E policy distribution")
        e.setflags(write=False)
        object.__setattr__(self, "E", e)

    @property
    def policy_count(self) -> int:
        return math.prod(len(a) for a in self.actions) ** self.horizon

    @property
    def policies(self) -> np.ndarray:
        actions = list(itertools.product(*(range(len(a)) for a in self.actions)))
        result = np.asarray(
            list(itertools.product(actions, repeat=self.horizon)), dtype=int
        )
        result.setflags(write=False)
        return result

    @property
    def tensor_digest(self) -> str:
        payload = {
            k: v
            for k, v in self._export().items()
            if k not in ("source_sha256", "annotations", "name")
        }
        return hashlib.sha256(
            json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            ).encode()
        ).hexdigest()

    def _export(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "dialect": "active-assembly-gnn-1",
            "name": self.name,
            "factors": self.factors,
            "state_labels": self.state_labels,
            "modalities": self.modalities,
            "actions": self.actions,
            "outcomes": self.outcomes,
            "A": [a.tolist() for a in self.A],
            "B": [b.tolist() for b in self.B],
            "C": [c.tolist() for c in self.C],
            "D": [d.tolist() for d in self.D],
            "E": self.E.tolist(),
            "A_dependencies": self.A_dependencies,
            "B_dependencies": [[i] for i in range(len(self.B))],
            "B_action_dependencies": [[i] for i in range(len(self.B))],
            "policies": self.policies.tolist(),
            "horizon": self.horizon,
            "gamma": self.gamma,
            "seed": self.seed,
            "source_sha256": self.source_sha256,
            "annotations": self.annotations,
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._export(), "model_digest": self.tensor_digest}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CompiledModel:
        """Validate a shared export; never trust an exported policy/axis declaration."""
        try:
            if (
                payload["schema_version"] != "1.0"
                or payload["dialect"] != "active-assembly-gnn-1"
            ):
                raise GNNError("GNN-E100: unsupported exported schema")
            model = cls(
                name=payload["name"],
                factors=tuple(payload["factors"]),
                modalities=tuple(payload["modalities"]),
                state_labels=tuple(tuple(v) for v in payload["state_labels"]),
                actions=tuple(tuple(v) for v in payload["actions"]),
                outcomes=tuple(tuple(v) for v in payload["outcomes"]),
                A=tuple(np.asarray(v) for v in payload["A"]),
                B=tuple(np.asarray(v) for v in payload["B"]),
                C=tuple(np.asarray(v) for v in payload["C"]),
                D=tuple(np.asarray(v) for v in payload["D"]),
                E=np.asarray(payload["E"]),
                A_dependencies=tuple(tuple(v) for v in payload["A_dependencies"]),
                horizon=payload["horizon"],
                gamma=payload["gamma"],
                seed=payload["seed"],
                source_sha256=payload["source_sha256"],
                annotations=payload.get("annotations", ""),
            )
            if (
                set(payload) != set(model.to_dict())
                or payload["model_digest"] != model.tensor_digest
            ):
                raise GNNError(
                    "GNN-E102: unknown export fields or tensor digest mismatch"
                )
            if (
                payload["policies"] != model.policies.tolist()
                or payload["B_dependencies"] != [[i] for i in range(len(model.B))]
                or payload["B_action_dependencies"]
                != [[i] for i in range(len(model.B))]
            ):
                raise GNNError(
                    "GNN-E102: unsupported policy or transition dependencies"
                )
            return model
        except (KeyError, TypeError, ValueError, IndexError, OverflowError) as exc:
            if isinstance(exc, GNNError):
                raise
            raise GNNError("GNN-E102: invalid exported model") from exc
