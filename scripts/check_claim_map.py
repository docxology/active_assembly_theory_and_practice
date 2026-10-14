"""Claim-map drift guard.

Parses the ``CLM-xxx`` identifier column in ``docs/claim_map.md`` and asserts
set equality (both directions) with the claim IDs in
``output/data/claim_register.json``. The map's prose is ignored: only the
first table cell of each data row is scanned, so a claim ID mentioned inside
evidence or boundary text never satisfies the guard.

Exit codes:
    0 - map IDs and register IDs match exactly;
    1 - drift (missing/extra IDs, duplicated map IDs), or a missing/unreadable
        input, or a malformed register.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLAIM_MAP_PATH = PROJECT_ROOT / "docs" / "claim_map.md"
CLAIM_REGISTER_PATH = PROJECT_ROOT / "output" / "data" / "claim_register.json"

CLAIM_ID_PATTERN = re.compile(r"\bCLM-\d+\b")


def claim_map_ids() -> list[str]:
    """Return the claim IDs from the map's ID column, in row order."""

    ids: list[str] = []
    header_seen = False
    for line in CLAIM_MAP_PATH.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped.startswith("|"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not header_seen:
            if cells and cells[0].lower() == "id":
                header_seen = True
            continue
        if all(set(cell) <= {"-", " ", ":"} for cell in cells if cell):
            continue  # markdown table separator row
        # The ID column is the first cell; prose in later cells never counts.
        ids.extend(CLAIM_ID_PATTERN.findall(cells[0] if cells else ""))
    if not header_seen:
        raise ValueError("docs/claim_map.md has no '| ID | ...' table header")
    return ids


def claim_register_ids() -> list[str]:
    """Return the claim IDs recorded in the generated claim register."""

    payload = json.loads(CLAIM_REGISTER_PATH.read_text(encoding="utf-8"))
    claims = payload.get("claims")
    if not isinstance(claims, list) or not claims:
        raise ValueError("claim_register.json has no non-empty 'claims' list")
    ids = [str(claim["claim_id"]) for claim in claims]
    if not all(CLAIM_ID_PATTERN.fullmatch(claim_id) for claim_id in ids):
        raise ValueError("claim_register.json contains a malformed claim_id")
    return ids


def main() -> int:
    try:
        map_ids = claim_map_ids()
        register_ids = claim_register_ids()
    except (OSError, ValueError, json.JSONDecodeError, KeyError) as exc:
        print(f"claim-map drift guard: cannot run: {exc}", file=sys.stderr)
        return 1

    map_set = set(map_ids)
    register_set = set(register_ids)
    duplicated = sorted(
        {claim_id for claim_id in map_ids if map_ids.count(claim_id) > 1}
    )
    missing_from_map = sorted(register_set - map_set)
    missing_from_register = sorted(map_set - register_set)

    if not (missing_from_map or missing_from_register or duplicated):
        print(
            f"claim-map drift guard: OK — {len(register_set)} claim IDs "
            f"match docs/claim_map.md and output/data/claim_register.json exactly"
        )
        return 0

    if duplicated:
        print(
            f"claim IDs repeated across claim_map.md rows: {duplicated}",
            file=sys.stderr,
        )
    if missing_from_map:
        print(
            f"claim IDs absent from docs/claim_map.md: {missing_from_map}",
            file=sys.stderr,
        )
    if missing_from_register:
        print(
            f"claim IDs in docs/claim_map.md absent from claim_register.json: "
            f"{missing_from_register}",
            file=sys.stderr,
        )
    print("claim-map drift guard: FAILED", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
