"""Run the pinned native Julia package tests through the project bridge."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rxinfer_bridge import (  # noqa: E402
    JULIA_PROJECT,
    EXPECTED_JULIA_VERSION,
    EXPECTED_RXINFER_VERSION,
    resolve_julia_executable,
    validate_backend_versions,
    validate_julia_manifest,
)


def main() -> int:
    """Validate the lockfile and run Julia's real package test suite."""

    validate_julia_manifest()
    julia = resolve_julia_executable()
    version = subprocess.run(
        [julia, "--version"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    reported_version = version.removeprefix("julia version ").strip()
    validate_backend_versions(reported_version, EXPECTED_RXINFER_VERSION)
    print(f"Using Julia {reported_version} ({EXPECTED_JULIA_VERSION} pinned)")
    completed = subprocess.run(
        [
            julia,
            f"--project={JULIA_PROJECT}",
            "-e",
            "using Pkg; Pkg.test(; allow_reresolve=false)",
        ],
        cwd=PROJECT_ROOT,
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
