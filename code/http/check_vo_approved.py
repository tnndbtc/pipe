#!/usr/bin/env python3
"""check_vo_approved.py — Exit 0 if VOPlan.{locale}.json has a vo_approval block.

Usage: python3 check_vo_approved.py <ep_dir> <locale>
Exits 0 if approved_at is non-empty, exits 1 if not.
"""

import json
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <ep_dir> <locale>", file=sys.stderr)
        sys.exit(1)

    ep_dir = Path(sys.argv[1])
    locale = sys.argv[2]
    path = ep_dir / f"VOPlan.{locale}.json"

    if not path.exists():
        sys.exit(1)

    try:
        m = json.loads(path.read_text(encoding="utf-8"))
        if bool(m.get("vo_approval", {}).get("approved_at")):
            sys.exit(0)
    except Exception:
        pass

    sys.exit(1)


if __name__ == "__main__":
    main()
