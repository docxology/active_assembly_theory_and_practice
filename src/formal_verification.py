"""Run and receipt the kernel-checked formal subset.

The formal project is intentionally small and closed-world. This module makes
the build an executable research artifact: it records the pinned Lean
toolchain, exact source hashes, theorem names, and a digest of compiler
output. A successful receipt means that the checked subset compiled locally;
it does not certify the Python lifter, Capstone, Unicorn, or the whole x86 ISA.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FORMAL_PROJECT_DIR = PROJECT_ROOT / "formal"
FORMAL_ARTIFACT = "output/data/formal_verification.json"
FORMAL_SCHEMA_VERSION = "1.0.0"
FORMAL_TOOLCHAIN = "leanprover/lean4:v4.28.0"
FORMAL_THEOREMS = (
    "ActiveAssembly.inc_inc_eq_add_two",
    "ActiveAssembly.constant_add_equivalence",
    "ActiveAssembly.neutral_nop_equivalence",
    "ActiveAssembly.changed_subtraction_is_not_equivalent",
    "ActiveAssembly.run_nop_append",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _formal_source_hashes(project_root: Path = PROJECT_ROOT) -> dict[str, str]:
    """Hash formal sources and lock/configuration files, excluding build products."""

    formal_root = project_root / "formal"
    return {
        str(path.relative_to(project_root)): _sha256(path)
        for path in sorted(formal_root.rglob("*"))
        if path.is_file()
        if ".lake" not in path.parts
    }


def _elan_executable() -> str:
    """Resolve the explicit elan executable used by the formal gate."""

    executable = shutil.which("elan")
    if executable is None:
        raise RuntimeError("elan is required for the formal verification gate")
    return executable


def build_formal_project(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Compile the formal project with the pinned installed Lean toolchain."""

    formal_dir = project_root / "formal"
    if not formal_dir.is_dir():
        raise FileNotFoundError(f"missing formal project: {formal_dir}")
    command = (
        _elan_executable(),
        "run",
        FORMAL_TOOLCHAIN,
        "lake",
        "build",
    )
    completed = subprocess.run(
        command,
        cwd=formal_dir,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    output = f"{completed.stdout}\n{completed.stderr}".strip()
    if completed.returncode != 0:
        raise RuntimeError(
            "formal Lean build failed with exit code "
            f"{completed.returncode}: {output[-4000:]}"
        )
    return {
        "command": list(command),
        "returncode": int(completed.returncode),
        "build_completed": True,
    }


def formal_verification_payload(project_root: Path = PROJECT_ROOT) -> dict[str, Any]:
    """Build the deterministic receipt for the locally checked formal subset."""

    build = build_formal_project(project_root)
    return {
        "schema_version": FORMAL_SCHEMA_VERSION,
        "status": "locally_verified",
        "toolchain": FORMAL_TOOLCHAIN,
        "formal_project": "formal",
        "source_hashes": _formal_source_hashes(project_root),
        "theorems": list(FORMAL_THEOREMS),
        "build": build,
        "trusted_base": (
            "Lean kernel and imported Lean core/standard library; no external "
            "solver proof certificate"
        ),
        "checked_fragment": {
            "state": "64-bit modular accumulator word",
            "operations": [
                "mov immediate",
                "add immediate",
                "sub immediate",
                "inc",
                "nop",
            ],
            "relation": (
                "universal final accumulator equality over the formal Word64 domain"
            ),
        },
        "scope": (
            "formal algebraic correspondence prerequisite for the transparent "
            "quotient witnesses; not a complete x86 semantics, lifter, compiler "
            "correctness theorem, or external-validity result"
        ),
    }


def validate_formal_verification_payload(payload: dict[str, Any]) -> tuple[bool, str]:
    """Fail closed on a missing, stale-looking, or scope-inflated receipt."""

    if payload.get("schema_version") != FORMAL_SCHEMA_VERSION:
        return False, f"schema={payload.get('schema_version')}"
    if payload.get("status") != "locally_verified":
        return False, "formal receipt is not locally verified"
    if payload.get("toolchain") != FORMAL_TOOLCHAIN:
        return False, "formal toolchain drift"
    if set(payload.get("theorems", [])) != set(FORMAL_THEOREMS):
        return False, "formal theorem manifest drift"
    source_hashes = payload.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        return False, "formal source hashes are missing"
    scope = str(payload.get("scope", "")).lower()
    if "complete" in scope and "not a complete" not in scope:
        return False, "formal scope overclaims completeness"
    return True, f"theorems={len(FORMAL_THEOREMS)} sources={len(source_hashes)}"


def write_formal_verification(project_root: Path = PROJECT_ROOT) -> str:
    """Compile the formal project and write its source-bound receipt."""

    payload = formal_verification_payload(project_root)
    valid, detail = validate_formal_verification_payload(payload)
    if not valid:
        raise ValueError(f"formal verification receipt is invalid: {detail}")
    path = project_root / FORMAL_ARTIFACT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return FORMAL_ARTIFACT


__all__ = [
    "FORMAL_ARTIFACT",
    "FORMAL_SCHEMA_VERSION",
    "FORMAL_THEOREMS",
    "FORMAL_TOOLCHAIN",
    "build_formal_project",
    "formal_verification_payload",
    "validate_formal_verification_payload",
    "write_formal_verification",
]
