#!/usr/bin/env python3
"""Order-change (Oct 24) source scrub for ASSEMBLED notebooks.

assemble.py keeps the master's cell sources; this pass rewrites the
sources of the derived assembled artifact so the deliverable presents
the remaining algorithms only:

- code cell merged-0121 (E4 combined SOTA table): the exact data-filter
  transform the execution vehicles use (split_units._sota_filter_transform),
  so the printed source is the code that produced the filtered outputs.
- code cell merged-0180 (SensorLLM comparison bars): the same bar filter
  as the vehicles (split_units._bars_filter_transform).
- narrative markdown cells: the removed model is taken out of model
  line-ups, and its rows are deleted from the authored markdown tables
  (merged-0119/0120 headline tables, merged-0123 cost table).

The SensorLLM section's own cells are deliberately NOT touched: the
method stays as the user ordered it restored. The master notebook file
is never modified - this runs only on assembled copies.

Usage: scrub_sources.py NOTEBOOK.ipynb  (in place)
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from split_units import _bars_filter_transform, _sota_filter_transform

MD_EDITS = {
    "merged-0000": [
        ("; **Chronos (t5) + Chronos-2** frozen foundation encoders "
         "with LogReg/RF/SVM heads |", " |"),
    ],
    "merged-e3-hdr": [
        ("\n- **Chronos** — the frozen foundation-model encoders "
         "(original Chronos t5 +\n  Chronos-2) with shallow heads, in "
         "their own sub-part below.", ""),
    ],
    "merged-sota-mr": [
        ("below and Chronos-2 + LogReg (the full\nChronos study — both "
         "variants × three heads — has its own sub-part below).",
         "below."),
    ],
    "merged-0117": [
        ("InceptionTime, Chronos-2 + LogReg, with TabPFN",
         "InceptionTime, with TabPFN"),
    ],
    "merged-0118": [
        ("; Chronos-2 embeddings + LogReg (the LLM companion /\n"
         "go-no-go probe); SensorLLM prepared", "; SensorLLM prepared"),
    ],
    "merged-0123": [
        ("(Chronos-2\nprobe, and SensorLLM's status)",
         "(SensorLLM's status)"),
    ],
}

MD_DELETE_LINES = {
    "merged-0119": "| **Chronos2-LogReg** |",
    "merged-0120": "| Chronos2-LogReg |",
    "merged-0123": "| Chronos2-LogReg | 15.5 min total |",
}


def joined(cell):
    src = cell.get("source", [])
    return "".join(src) if isinstance(src, list) else src


def main():
    p = Path(sys.argv[1])
    nb = json.loads(p.read_text())
    done = []
    for cell in nb["cells"]:
        cid = cell.get("id")
        src = joined(cell)
        if cell["cell_type"] == "code" and cid == "merged-0121":
            cell["source"] = _sota_filter_transform(src)
            done.append((cid, "sota-filter"))
        elif cell["cell_type"] == "code" and cid == "merged-0180":
            cell["source"] = _bars_filter_transform(src)
            done.append((cid, "bars-filter"))
        elif cell["cell_type"] == "markdown" and cid in MD_EDITS:
            for old, new in MD_EDITS[cid]:
                assert old in src, f"{cid}: pattern not found: {old[:60]}"
                src = src.replace(old, new)
            cell["source"] = src
            done.append((cid, "md-edits"))
        if cell["cell_type"] == "markdown" and cid in MD_DELETE_LINES:
            pat = MD_DELETE_LINES[cid]
            lines = joined(cell).splitlines(keepends=True)
            kept = [l for l in lines if pat not in l]
            assert len(kept) == len(lines) - 1, f"{cid}: row not found"
            cell["source"] = "".join(kept)
            done.append((cid, "md-row-delete"))
    p.write_text(json.dumps(nb, indent=1))
    print("scrubbed:", done)


if __name__ == "__main__":
    main()
