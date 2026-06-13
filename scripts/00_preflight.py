"""Preflight imports for the assembly Active Inference scaffold."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rxinfer_bridge import resolve_julia_executable


def main() -> int:
    import capstone
    import pymdp
    import unicorn

    print("capstone", getattr(capstone, "__version__", "installed"))
    print("pymdp", getattr(pymdp, "__version__", "installed"))
    print("unicorn", getattr(unicorn, "__version__", "installed"))
    try:
        julia = resolve_julia_executable()
        completed = subprocess.run(
            [julia, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except RuntimeError:
        print("julia not found")
    else:
        print(completed.stdout.strip())
    print(f"project={PROJECT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
