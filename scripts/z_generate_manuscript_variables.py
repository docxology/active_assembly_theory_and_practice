"""Generate manuscript variables for template rendering."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_ROOT = PROJECT_ROOT.parents[2] / "template"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if TEMPLATE_ROOT.exists() and str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

from infrastructure.rendering.manuscript_injection import write_resolved_manuscript_tree

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
    from src.manuscript_variables import generate_variables

    variables = generate_variables()
    out_dir = write_resolved_manuscript_tree(PROJECT_ROOT, variables)
    for key in sorted(variables):
        print(f"{key}={variables[key]}")
    print(f"resolved_manuscript={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
