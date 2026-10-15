import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.project_rendering import render_project

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--skip-manuscript-hydration", action="store_true", required=True
    )
    args = parser.parse_args()
    render_project(
        args.project_root,
        project_name="active_assembly_theory_and_practice",
        formats=("pdf", "html", "docx", "slides"),
    )
