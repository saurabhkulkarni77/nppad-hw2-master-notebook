#!/usr/bin/env python3
"""Per-cell checkpointing notebook executor.

Executes the pending (execution_count is None) code cells of a vehicle
notebook in ONE kernel session, saving the notebook after every cell.
Resumable: re-running skips cells that already have an execution_count.
Stops at the first error/timeout and records it in the notebook
(cell metadata['exec_error']) and in a JSON sidecar log.

Usage:
  run_unit.py VEHICLE.ipynb --workdir DIR --kernel NAME \
      [--cell-timeout SECONDS] [--log LOG.json]
Env: NPPAD_DATA_DIR should point at the staged dataset folder.
"""
import argparse, json, sys, time
from pathlib import Path

from jupyter_client.manager import KernelManager


def cell_source(cell):
    s = cell.get("source", "")
    return "".join(s) if isinstance(s, list) else s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("vehicle")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--kernel", default="python3")
    ap.add_argument("--cell-timeout", type=int, default=14400)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    veh = Path(args.vehicle)
    nb = json.loads(veh.read_text())
    log = {"vehicle": str(veh), "kernel": args.kernel, "events": []}

    def save():
        tmp = veh.with_suffix(".tmp.ipynb")
        tmp.write_text(json.dumps(nb, indent=1))
        tmp.replace(veh)

    def record(ev):
        ev["t"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        log["events"].append(ev)
        print(ev, flush=True)
        if args.log:
            Path(args.log).write_text(json.dumps(log, indent=1))

    km = KernelManager(kernel_name=args.kernel)
    km.start_kernel(cwd=args.workdir)
    kc = km.client()
    kc.start_channels()
    try:
        kc.wait_for_ready(timeout=120)
    except Exception as e:
        record({"event": "kernel_not_ready", "error": str(e)})
        kc.stop_channels(); km.shutdown_kernel(now=True)
        return 2

    record({"event": "kernel_started", "kernel": args.kernel})
    exec_counter = max(
        [c.get("execution_count") or 0 for c in nb["cells"]] + [0])

    # Resume support: a fresh kernel lacks the namespace (imports, result
    # dicts) built by cells executed in earlier sessions. If this run is a
    # resume, replay those cells kernel-side only (their saved notebook
    # outputs stay untouched) so pending cells see the same state as in
    # one continuous run.
    done_cells = [(i, c) for i, c in enumerate(nb["cells"])
                  if c.get("cell_type") == "code"
                  and c.get("execution_count") is not None
                  and "exec_error" not in c.get("metadata", {})
                  and cell_source(c).strip()]
    has_pending = any(c.get("cell_type") == "code"
                      and (c.get("execution_count") is None
                           or "exec_error" in c.get("metadata", {}))
                      and cell_source(c).strip() for c in nb["cells"])
    if done_cells and has_pending:
        record({"event": "replay_start", "count": len(done_cells)})
        for idx, cell in done_cells:
            msg_id = kc.execute(cell_source(cell), store_history=False,
                                allow_stdin=False)
            deadline = time.time() + args.cell_timeout
            idle = False
            rerr = None
            while not idle:
                if time.time() > deadline:
                    rerr = f"replay timeout at cell {idx}"
                    break
                try:
                    msg = kc.get_iopub_msg(timeout=30)
                except Exception:
                    if not km.is_alive():
                        rerr = "kernel died during replay"
                        break
                    continue
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                if msg["msg_type"] == "error":
                    rerr = (f"replay cell {idx}: "
                            f"{msg['content']['ename']}: "
                            f"{msg['content']['evalue']}")
                elif msg["msg_type"] == "status" and \
                        msg["content"]["execution_state"] == "idle":
                    idle = True
            if rerr:
                record({"event": "replay_error", "error": rerr[:300]})
                kc.stop_channels()
                km.shutdown_kernel(now=True)
                record({"event": "unit_finished", "status": 4})
                return 4
        record({"event": "replay_done"})

    status = 0
    try:
        for idx, cell in enumerate(nb["cells"]):
            if cell.get("cell_type") != "code":
                continue
            if cell.get("execution_count") is not None \
                    and "exec_error" not in cell.get("metadata", {}):
                continue
            if "exec_error" in cell.get("metadata", {}):
                # previously failed cell: clear and re-run
                cell["outputs"] = []
                cell["execution_count"] = None
                del cell["metadata"]["exec_error"]
            src = cell_source(cell)
            if not src.strip():
                cell["execution_count"] = None
                continue
            exec_counter += 1
            record({"event": "cell_start", "index": idx,
                    "exec_count": exec_counter})
            msg_id = kc.execute(src, allow_stdin=False)
            outputs = []
            err = None
            deadline = time.time() + args.cell_timeout
            idle = False
            while not idle:
                if time.time() > deadline:
                    err = ("timeout", f"cell exceeded {args.cell_timeout}s")
                    break
                try:
                    msg = kc.get_iopub_msg(timeout=30)
                except Exception:
                    if not km.is_alive():
                        err = ("kernel_died", "kernel process died")
                        break
                    continue
                if msg["parent_header"].get("msg_id") != msg_id:
                    continue
                mt = msg["msg_type"]
                content = msg["content"]
                if mt == "stream":
                    if outputs and outputs[-1].get("output_type") == "stream" \
                            and outputs[-1].get("name") == content["name"]:
                        outputs[-1]["text"] += content["text"]
                    else:
                        outputs.append({"output_type": "stream",
                                        "name": content["name"],
                                        "text": content["text"]})
                elif mt in ("execute_result", "display_data"):
                    out = {"output_type": mt, "data": content["data"],
                           "metadata": content.get("metadata", {})}
                    if mt == "execute_result":
                        out["execution_count"] = exec_counter
                    outputs.append(out)
                elif mt == "error":
                    outputs.append({"output_type": "error",
                                    "ename": content["ename"],
                                    "evalue": content["evalue"],
                                    "traceback": content["traceback"]})
                    err = (content["ename"], content["evalue"])
                elif mt == "status" and content["execution_state"] == "idle":
                    idle = True
            cell["outputs"] = outputs
            cell["execution_count"] = exec_counter
            if err:
                cell.setdefault("metadata", {})["exec_error"] = \
                    f"{err[0]}: {err[1]}"
                record({"event": "cell_error", "index": idx,
                        "error": f"{err[0]}: {err[1]}"})
                save()
                status = 3
                break
            record({"event": "cell_done", "index": idx})
            save()
    finally:
        try:
            kc.stop_channels()
        except Exception:
            pass
        try:
            km.shutdown_kernel(now=True)
        except Exception:
            pass
    record({"event": "unit_finished", "status": status})
    return status


if __name__ == "__main__":
    sys.exit(main())
