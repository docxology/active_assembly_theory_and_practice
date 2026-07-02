"""Tests for the RxInfer bridge version pin and Julia-availability contract.

No mocks: the Julia-unavailable path is exercised with a genuinely absent
executable name resolved through the real ``shutil.which``, and the version
contract is exercised with plain string arguments.
"""

from __future__ import annotations

import pytest

from src.rxinfer_bridge import (
    EXPECTED_JULIA_VERSION,
    EXPECTED_RXINFER_VERSION,
    julia_available,
    resolve_julia_executable,
    validate_backend_versions,
)

# A name that cannot exist on PATH, used to drive the unavailable-toolchain path.
_MISSING_JULIA = "definitely-not-a-real-julia-executable-xyz"


def test_expected_versions_are_pinned_nonempty_strings() -> None:
    """The pinned toolchain versions are explicit, documented string constants."""
    assert isinstance(EXPECTED_JULIA_VERSION, str) and EXPECTED_JULIA_VERSION
    assert isinstance(EXPECTED_RXINFER_VERSION, str) and EXPECTED_RXINFER_VERSION


def test_validate_backend_versions_accepts_pinned_versions() -> None:
    # Must not raise when the reported versions match the pins.
    validate_backend_versions(EXPECTED_JULIA_VERSION, EXPECTED_RXINFER_VERSION)


def test_validate_backend_versions_rejects_julia_drift() -> None:
    with pytest.raises(RuntimeError, match="Julia"):
        validate_backend_versions("0.0.0", EXPECTED_RXINFER_VERSION)


def test_validate_backend_versions_rejects_rxinfer_drift() -> None:
    with pytest.raises(RuntimeError, match="RxInfer"):
        validate_backend_versions(EXPECTED_JULIA_VERSION, "0.0.0")


def test_validate_backend_versions_reports_both_mismatches() -> None:
    with pytest.raises(RuntimeError) as excinfo:
        validate_backend_versions("0.0.0", "0.0.0")
    message = str(excinfo.value)
    assert "Julia" in message
    assert "RxInfer" in message


def test_resolve_julia_executable_raises_when_unavailable() -> None:
    """Missing Julia is a hard, clearly-labeled failure, not a silent None."""
    with pytest.raises(RuntimeError, match="Julia executable not found"):
        resolve_julia_executable(_MISSING_JULIA)


def test_julia_available_is_false_when_unavailable() -> None:
    """The availability contract returns False (never raises) for a missing tool."""
    assert julia_available(_MISSING_JULIA) is False


def test_julia_available_returns_bool_for_default() -> None:
    """Whether or not Julia is installed, the contract answers with a bool."""
    assert isinstance(julia_available(), bool)
