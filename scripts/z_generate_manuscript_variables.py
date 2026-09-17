"""Generate manuscript variables for template rendering (thin orchestrator)."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

REEXEC_ENV = "ACTIVE_ASSEMBLY_PROJECT_UV"


def _missing_project_dependencies() -> list[str]:
    return [
        package
        for package in ("capstone", "pymdp", "unicorn")
        if importlib.util.find_spec(package) is None
    ]


def _reexec_in_project_environment_if_needed() -> None:
    missing = _missing_project_dependencies()
    if not missing or os.environ.get(REEXEC_ENV) == "1":
        return

    env = dict(os.environ)
    env[REEXEC_ENV] = "1"
    command = [
        "uv",
        "run",
        "--project",
        str(PROJECT_ROOT),
        "python",
        str(Path(__file__).resolve()),
        *sys.argv[1:],
    ]
    raise SystemExit(subprocess.call(command, env=env))


def main() -> int:
    _reexec_in_project_environment_if_needed()

    # Hydration logic lives in src/; the script only bootstraps the project
    # environment and delegates. The gate imports must stay behind the re-exec.
    from src.template_hydration import run_hydration_gate

    return run_hydration_gate()


if __name__ == "__main__":
    raise SystemExit(main())
