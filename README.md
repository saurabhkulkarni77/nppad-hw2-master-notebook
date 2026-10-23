# NPPAD HW2 — Master Notebook Execution (GitHub Actions)

Compiled (executed) run of Saurabh's HW2 all-models master notebook
(`NPPAD_All_Models_Merged_Colab_DriveData_Fixed_2026-10-23.ipynb`,
md5 `0819dfc9b6c7821f15617ccd45ae11e9`, 185 cells, restructured as
Task E items E1–E5 with a Chronos sub-part; quantum excluded by instruction).

## How it runs

The notebook is executed **section by section** via the
`Execute NPPAD master notebook` workflow (Actions tab → Run workflow):

| Unit | Content | Cells (master) |
|---|---|---|
| U0 | Data + Task E flow (E1, E2, E3 core + X1/X2, E4, R1/R2, E5 core) | 0–91 + 100–131 |
| U1 | Chronos sub-part (Chronos t5-large + Chronos-2, 3 heads, 5 windows) | 92–99 |
| U2 | Physics-informed models + space PINN | 132–153 |
| U3 | SensorLLM (two-stage, Qwen2.5-0.5B local backbone) | 154–173 |
| U4 | LLM late fusion — CPU local variant (see below) | 174–184 |
| assemble | Splice executed units into one notebook + render PDF | — |

Each dispatch runs ONE unit and saves the whole workspace as a
`workspace` artifact; the next dispatch resumes it via the
`state_run_id` input (the previous run's id). The executor
(`run_unit.py`) checkpoints after every cell and skips cells that
already ran, so a timed-out unit is simply re-dispatched.

## Data

The notebook's one data cell expects the NPPAD dataset in the working
folder. It is staged from the public source repo
`thu-inet/NuclearPowerPlantAccidentData` (sparse checkout of
`Operation_csv_data` + `Variable_Power_Data`, which includes the
`.mdb` files the robustness extension reads), so no Google Drive
access is attempted.

## Late fusion on CPU runners

Cells 179–181 of the master are the study's GPU preset (a ~19 GB
DeepSeek-R1-Distill-Qwen-32B GGUF served by llama.cpp with full GPU
offload). GitHub's runners have no GPU, so in this compiled run those
cells are left unexecuted (an explanatory note cell is inserted in the
assembled notebook) and the executed local variant runs the five
subsystem experts + the fusion-prompt build on the 60-window
stratified test subsample — the original study's local design. The
LLM fusion step itself is not run here.

## Outputs

- Per-unit artifacts: executed unit notebook + executor logs.
- `assembled` artifact (assemble dispatch): the full executed
  notebook + PDF.
