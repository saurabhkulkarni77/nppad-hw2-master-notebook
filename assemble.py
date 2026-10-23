#!/usr/bin/env python3
"""Splice executed unit vehicles back into a copy of the master notebook.

Matching is by cell id. Every master code cell takes outputs +
execution_count from its executed vehicle copy; cells that were never
executed (the late-fusion GPU preset) keep null counts and no outputs.
The late-fusion local-variant note + run cells (vehicle-only ids) are
inserted right after master cell 178's position.

Usage: assemble.py MASTER.ipynb UNITS_DIR OUT.ipynb
"""
import json, sys
from pathlib import Path

UNITS = ["U0_taske", "U1_chronos", "U2_physics", "U3_sensorllm",
         "U4_latefusion"]
EXTRA_AFTER_ID = None  # resolved at runtime: id of master cell 178


def main():
    master = json.loads(Path(sys.argv[1]).read_text())
    units_dir = Path(sys.argv[2])
    out = Path(sys.argv[3])

    executed = {}   # cell id -> executed cell
    extras = []     # vehicle-only cells (LF note + variant), in order
    for u in UNITS:
        nb = json.loads((units_dir / f"{u}.ipynb").read_text())
        for c in nb["cells"]:
            if c["id"] in ("lf-local-variant-note", "lf-local-variant-run"):
                extras.append(c)
            else:
                executed[c["id"]] = c

    anchor_id = master["cells"][178]["id"]
    new_cells = []
    report = {"executed": 0, "unexecuted": [], "missing": []}
    for c in master["cells"]:
        ec = executed.get(c["id"])
        if c["cell_type"] == "code":
            if ec is None:
                report["missing"].append(c["id"])
            elif ec.get("execution_count") is not None:
                c["outputs"] = ec.get("outputs", [])
                c["execution_count"] = ec["execution_count"]
                report["executed"] += 1
            else:
                c["outputs"] = []
                c["execution_count"] = None
                report["unexecuted"].append(c["id"])
        new_cells.append(c)
        if c["id"] == anchor_id:
            new_cells.extend(extras)
    master["cells"] = new_cells
    out.write_text(json.dumps(master, indent=1))
    print(json.dumps(report, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
