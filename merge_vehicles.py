#!/usr/bin/env python3
"""Transplant execution state from existing vehicles into fresh ones.

For every cell id present in both, copies outputs + execution_count
(+ exec_error metadata) from the OLD vehicle into the FRESH vehicle
(whose cell order may differ). Used when the vehicle layout changes
between runs so completed cells are not re-executed.

Usage: merge_vehicles.py OLD_DIR NEW_DIR
"""
import json, sys
from pathlib import Path

old_dir, new_dir = Path(sys.argv[1]), Path(sys.argv[2])
for fresh_path in sorted(new_dir.glob("*.ipynb")):
    old_path = old_dir / fresh_path.name
    if not old_path.exists():
        print("no old vehicle for", fresh_path.name)
        continue
    old = json.loads(old_path.read_text())
    fresh = json.loads(fresh_path.read_text())
    state = {c["id"]: c for c in old["cells"]}
    moved = 0
    for c in fresh["cells"]:
        o = state.get(c["id"])
        if o is not None and c["cell_type"] == "code" \
                and o.get("execution_count") is not None:
            c["outputs"] = o.get("outputs", [])
            c["execution_count"] = o["execution_count"]
            if "exec_error" in o.get("metadata", {}):
                c["metadata"]["exec_error"] = o["metadata"]["exec_error"]
            moved += 1
    fresh_path.write_text(json.dumps(fresh, indent=1))
    print(fresh_path.name, "transplanted", moved, "executed cells")
