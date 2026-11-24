"""Live activity view shared by Desktop Body and the read-only trace viewer."""
import argparse
import json
from pathlib import Path
import tkinter as tk
from tkinter.scrolledtext import ScrolledText


RUNS = Path(__file__).resolve().parent / "build/menu-runs"


def describe(row):
    kind = row.get("event")
    if kind == "decision":
        decision = row["decision"]
        return f'Attempt · {decision["kind"]}\n{decision["evidence"]}'
    if kind == "call":
        return f'MCP · {row["tool"]}({json.dumps(row["arguments"], ensure_ascii=False)})'
    if kind == "result":
        return ("Tool error" if row["error"] else "Tool result") + " · " + row["outcome"]
    if kind == "observation":
        return "Headset frame received. Waiting for model decision."
    if kind == "finished":
        return f'{row["status"]} · {row.get("detail") or row.get("model_report","")}'
    if kind == "objective":
        return row["text"]
    if kind == "operator_setup":
        return row["note"]
    if kind == "pointer_check":
        return f'Pointer measured at {row["tip"]}; requested {row["desired"]}'
    return row.get("outcome","")


def describe_supervised(row):
    event = row["event"]
    if event == "start":
        return "Supervisor", row.get("label", "Connected to the private VRChat session")
    if event == "supervisor_objective":
        return "Supervisor", row["task"]["objective"]
    if event == "inference_started":
        return "Omni", "Planning…"
    if event in ("omni_plan", "queued_plan"):
        plan = row["plan"]
        lines = [f'{action["tool"]}({json.dumps(action["arguments"], ensure_ascii=False)})'
                 for action in plan.get("actions", [])]
        for action in plan.get("actions", []):
            for check in action.get("checks", []):
                lines.append(f'Pending d1 check for {action["tool"]} · {check["question"]}')
        if plan.get("when"):
            lines.append("Waiting for · " + plan["when"]["question"])
        if plan.get("verify"):
            lines.append("Planned verification · " + plan["verify"]["question"])
        if plan.get("child"):
            lines.append("Delegated objective · " + plan["child"])
        return ("Omni" if event == "omni_plan" else "Prequeued actions"), "\n".join(lines) or plan.get("status", "")
    if event == "verification_started":
        return "d1", f'Checking {row["questions"]} condition(s)…'
    if event == "d1_check":
        result = "passed" if row["fired"] else "below threshold"
        return "d1", f'{row["question"]}\n{row["confidence"]:.0%} · {result}'
    if event == "trigger_expired":
        return "d1", f'Timed out · {row["question"]}'
    if event == "action_started":
        return "Controller", f'Starting {row["tool"]}({json.dumps(row["arguments"], ensure_ascii=False)})'
    if event == "action_result":
        tool = row.get("action", {}).get("tool", "action")
        return "Controller", f'{tool} → {row["result"]}'
    if event == "action_skipped":
        return "Controller", f'Skipped · {row["reason"]}'
    if event == "arrival_evidence":
        return "Stereo", f'Arrival {"verified" if row["arrived"] else "pending"} · {row["error_m"]:.2f} m from waypoint'
    if event == "held_alignment":
        return "Stereo", f'Landing alignment error · {row["error_m"]:.2f} m'
    if event in ("held_color_evidence", "settled_support_evidence"):
        flag = "held" if event == "held_color_evidence" else "supported"
        details = ", ".join(f'{name} {flag if item[flag] else "unverified"}' for name, item in row["colors"].items())
        return "Stereo", details
    if event == "supervisor_result":
        return "Supervisor", f'Checkpoint {row["checkpoint"]} · {row["status"].replace("_", " ")}'
    if event == "finished":
        return "Supervisor", f'Finished · {row["outcome"]}'
    if event == "supervisor_error":
        return "Supervisor", row["reason"]
    if "recovery" in event or event.startswith("cache_warm"):
        return "Harness", event.replace("_", " ").capitalize()
    return None


class ActivityColumn(tk.LabelFrame):
    def __init__(self, parent, *, inbox=None, trace=None):
        super().__init__(parent, text="Agent activity")
        self.source = tk.StringVar(value="Waiting for a run")
        tk.Label(self, textvariable=self.source, anchor="w").pack(fill="x")
        self.output = ScrolledText(self, width=38, wrap="word", state="disabled")
        self.output.pack(expand=True, fill="both")
        for role, color in {"Supervisor": "#553b80", "Omni": "#1956a2", "d1": "#a1490f",
                            "Prequeued actions": "#4b5966", "Controller": "#126047",
                            "Stereo": "#735915", "Harness": "#553b80"}.items():
            self.output.tag_configure(role, foreground=color, font=("Segoe UI", 10, "bold"))
        self.inbox = inbox
        self.trace = trace
        self.path = None
        self.offset = 0
        self.timer = self.after(0, self.poll)

    def append(self, text, *, role=None, stamp=""):
        at_end = self.output.yview()[1] >= .995
        self.output.configure(state="normal")
        if role:
            self.output.insert("end", f"{stamp}  {role}\n", role)
        self.output.insert("end", text+"\n\n")
        self.output.configure(state="disabled")
        if at_end:
            self.output.see("end")

    def poll(self):
        try:
            supervised = self.trace is not None or self.inbox is not None
            if self.inbox is not None:
                checkpoint = self.inbox / "checkpoint.json"
                path = Path(json.loads(checkpoint.read_text(encoding="utf-8"))["trace"]) if checkpoint.is_file() else None
            elif self.trace is not None:
                path = self.trace if self.trace.is_file() else None
            else:
                path = max(RUNS.glob("*/events.jsonl"), key=lambda p:p.parent.name, default=None)
            if path and path != self.path:
                self.path, self.offset = path, 0
                self.append(("Live session · " if supervised else "Menu trial · ") + path.parent.name)
            if path:
                with path.open("rb") as stream:
                    stream.seek(self.offset)
                    for line in stream:
                        if not line.endswith(b"\n"):
                            break
                        self.offset += len(line)
                        row = json.loads(line)
                        if supervised:
                            entry = describe_supervised(row)
                            if entry:
                                role, label = entry
                                self.source.set(f"Latest · {role}")
                                self.append(label, role=role, stamp=f'{row["seconds"]:6.2f}s')
                        else:
                            label = describe(row)
                            if label:
                                self.source.set("Menu trial · "+row["event"])
                                self.append(row["time"][11:19]+"  "+label)
        except (OSError, ValueError, KeyError) as error:
            self.source.set("Trace unavailable · "+str(error))
        self.timer = self.after(650, self.poll)

    def destroy(self):
        self.after_cancel(self.timer)
        super().destroy()


if __name__=="__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    source = cli.add_mutually_exclusive_group()
    source.add_argument("--inbox", type=Path, help="Follow a supervised run through its checkpoint file")
    source.add_argument("--trace", type=Path, help="Follow one supervised event log")
    args = cli.parse_args()
    root = tk.Tk()
    root.title("Desktop Body · Agent activity")
    root.geometry("530x720+25+80")
    ActivityColumn(root, inbox=args.inbox, trace=args.trace).pack(expand=True,fill="both")
    root.mainloop()
