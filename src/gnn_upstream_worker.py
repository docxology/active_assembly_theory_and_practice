from __future__ import annotations

import hashlib
from importlib import import_module
import json
from pathlib import Path
import sys


def main() -> int:
    root, source = (Path(value).resolve() for value in sys.argv[1:])
    sys.path.insert(0, str(root / "src"))
    MarkdownGNNParser = import_module("gnn.parsers.markdown_parser").MarkdownGNNParser

    raw = source.read_bytes()
    result = MarkdownGNNParser().parse_string(raw.decode("utf-8"))
    imported = {}
    for name, module in tuple(sys.modules.items()):
        if name == "gnn" or name.startswith("gnn."):
            if module.__file__ is None:
                raise ImportError(f"upstream module has no source: {name}")
            path = Path(module.__file__).resolve()
            imported[str(path.relative_to(root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    print(
        json.dumps(
            {
                "success": result.success,
                "errors": [str(value) for value in result.errors],
                "warnings": [str(value) for value in result.warnings],
                "name": result.model.model_name,
                "parameters": [
                    {"name": p.name, "value": p.value} for p in result.model.parameters
                ],
                "variables": [
                    {"name": v.name, "dimensions": v.dimensions}
                    for v in result.model.variables
                ],
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "import_hashes": imported,
            },
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
