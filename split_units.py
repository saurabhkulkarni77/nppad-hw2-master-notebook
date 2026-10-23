#!/usr/bin/env python3
"""Split the FIXED master notebook into execution-unit vehicles.

Units respect kernel-state lineages of the restructured notebook:
  U0 TaskE flow : cells 0-91 + 100-131 (one lineage; Chronos sub-part is a
                  separate self-contained flow, split out as U1)
  U1 Chronos    : cells 92-99 (own setup, embedded adv_data, shared root)
  U2 Physics    : cells 132-153 (own setup, embedded files)
  U3 SensorLLM  : cells 154-173 (own setup, embedded library)
  U4 LateFusion : cells 174-184 with the CPU local variant: cells 179-181
                  (GPU preset: 19 GB 32B GGUF + llama.cpp server) are NOT
                  executed; a note cell + a variant cell running the
                  experts + prompt build on the 60-window stratified
                  subsample (the local study's design) are inserted
                  after cell 178.

All master cells are copied verbatim (ids preserved). Outputs are cleared.
Usage: split_units.py MASTER.ipynb OUTDIR
"""
import copy, json, sys
from pathlib import Path

UNITS = {
    # Master cell 107 (E4a) defines f1_table(), which the export cell 91
    # uses; in master order 91 precedes 107, so the vehicle executes 107
    # immediately before 91. Assembly splices by cell id, so the final
    # notebook keeps the master layout; only execution order differs.
    # Master cell 120 rebinds `summ` to the maturity-ladder frame,
    # but master cell 129 still reads the original E1 `summ`
    # (t0/vars/mean). The vehicle therefore runs 129 before 120, and a
    # scaffold cell (id 'scaffold-rebuild-summ', inserted in main())
    # rebuilds the original `summ` from its CSV right before 129, so
    # resumes (whose replay ends with the ladder frame) also work.
    "U0_taske": list(range(0, 91)) + [107, 91] + list(range(100, 107))
                + list(range(108, 120)) + [129, 120] + list(range(121, 129))
                + [130, 131],
    "U1_chronos": list(range(92, 100)),
    "U2_physics": list(range(132, 154)),
    "U3_sensorllm": list(range(154, 174)),
    "U4_latefusion": list(range(174, 185)),
}

LF_NOTE = {
    "cell_type": "markdown", "id": "lf-local-variant-note", "metadata": {},
    "source": (
        "### Execution note — late fusion on a CPU machine (local variant)\n"
        "\n"
        "The three cells after the variant cell below are the study's **GPU "
        "preset**: they download a ~19 GB DeepSeek-R1-Distill-Qwen-32B GGUF "
        "and serve it with llama.cpp using full GPU offload (`-ngl 99`). "
        "This machine has no GPU, so those cells are **left unexecuted** in "
        "this compiled run (no outputs — deliberately). What was executed "
        "instead is the local variant from the original study: the five "
        "subsystem experts are trained and the fusion prompts are built on "
        "the 60-window stratified test subsample. The LLM fusion step "
        "itself was not run here; the full 240-window GPU pipeline is the "
        "cells below, unchanged."),
}

LF_VARIANT = {
    "cell_type": "code", "id": "lf-local-variant-run",
    "metadata": {}, "execution_count": None, "outputs": [],
    "source": (
        "# --- LOCAL VARIANT (execution-copy adaptation): experts + prompt build ---\n"
        "# Runs the study modules written by the setup cells above, as the local\n"
        "# study did (experts -> prompts on the 60-window stratified subsample).\n"
        "# One glue fix is needed: the patched adv_data.py reads a bare\n"
        "# NPPAD_DATA_ROOT name (in a script process it does not resolve), so each\n"
        "# script is launched through a bootstrap that pre-seeds that module\n"
        "# global with the data root found by the setup cell. No study logic is\n"
        "# changed. The LLM fusion step itself is NOT run: no GPU / 32B server\n"
        "# on this machine (see note above).\n"
        "from pathlib import Path\n"
        "BOOT = '''\n"
        "import sys, runpy, importlib.util\n"
        "sys.path.insert(0, \".\"); sys.path.insert(0, \"src\")\n"
        "spec = importlib.util.spec_from_file_location(\"adv_data\", \"adv_data.py\")\n"
        "adv = importlib.util.module_from_spec(spec)\n"
        "adv.NPPAD_DATA_ROOT = NPPAD_ROOT_VALUE\n"
        "sys.modules[\"adv_data\"] = adv\n"
        "spec.loader.exec_module(adv)\n"
        "sys.argv = [sys.argv[1]] + sys.argv[2:]\n"
        "runpy.run_path(sys.argv[0], run_name=\"__main__\")\n"
        "'''\n"
        "root = str(Path(NPPAD_DATA_ROOT).resolve())\n"
        "Path('_run_local.py').write_text(BOOT.replace('NPPAD_ROOT_VALUE', repr(root)))\n"
        "import subprocess, sys\n"
        "for cmd in [[sys.executable, '_run_local.py', 'src/experts.py'],\n"
        "            [sys.executable, '_run_local.py', 'src/prep_prompts.py', '--n', '60']]:\n"
        "    print('RUNNING:', ' '.join(cmd), flush=True)\n"
        "    r = subprocess.run(cmd, capture_output=True, text=True)\n"
        "    print(r.stdout[-3500:])\n"
        "    if r.returncode != 0:\n"
        "        print('STDERR:', r.stderr[-2500:])\n"
        "    print('exit code:', r.returncode, flush=True)\n"),
}


SCAFFOLD_SUMM = {
    "cell_type": "code", "id": "scaffold-rebuild-summ",
    "metadata": {}, "execution_count": None, "outputs": [],
    "source": (
        "# --- execution scaffold (vehicle only; not part of the study) ---\n"
        "# Master cell 120 rebinds `summ` to the maturity-ladder frame, but\n"
        "# the next cell (master 129) reads the original E1 summary table.\n"
        "# Rebuild it here with the exact expressions of master cell 68\n"
        "# (which reads the same CSV), so cell 129 sees the table it expects\n"
        "# in both fresh runs and resumed runs.\n"
        "M = pd.read_csv(RES / \"main_metrics_lr_rf.csv\")\n"
        "summ = (M.groupby([\"t0\", \"vars\", \"model\", \"W\"])[\"macro_F1\"]"
        ".agg([\"mean\", \"std\", \"count\"]).reset_index())\n"
        "acc = M.groupby([\"t0\", \"vars\", \"model\", \"W\"])[\"accuracy\"]"
        ".mean().rename(\"acc_mean\").reset_index()\n"
        "summ = summ.merge(acc)\n"
        "print('summ rebuilt:', summ.shape)\n"),
}


def main():
    master = json.loads(Path(sys.argv[1]).read_text())
    outdir = Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    cells = master["cells"]
    for unit, idxs in UNITS.items():
        ucells = []
        for i in idxs:
            if unit == "U0_taske" and i == 129:
                ucells.append(copy.deepcopy(SCAFFOLD_SUMM))
            c = copy.deepcopy(cells[i])
            if c["cell_type"] == "code":
                c["outputs"] = []
                c["execution_count"] = None
            ucells.append(c)
            if unit == "U4_latefusion" and i == 178:
                ucells.append(copy.deepcopy(LF_NOTE))
                ucells.append(copy.deepcopy(LF_VARIANT))
        nb = {"nbformat": 4, "nbformat_minor": 5,
              "metadata": copy.deepcopy(master["metadata"]), "cells": ucells}
        p = outdir / f"{unit}.ipynb"
        p.write_text(json.dumps(nb, indent=1))
        print(unit, len(ucells), "cells ->", p)


if __name__ == "__main__":
    main()
