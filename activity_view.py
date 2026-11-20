"""Live activity view shared by Desktop Body and the read-only trace viewer."""
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


class ActivityColumn(tk.LabelFrame):
    def __init__(self, parent):
        super().__init__(parent, text="Agent activity")
        self.source = tk.StringVar(value="Waiting for a run")
        tk.Label(self, textvariable=self.source, anchor="w").pack(fill="x")
        self.output = ScrolledText(self, width=38, wrap="word", state="disabled")
        self.output.pack(expand=True, fill="both")
        self.path = None
        self.offset = 0
        self.timer = self.after(0, self.poll)

    def append(self, text):
        at_end = self.output.yview()[1] >= .995
        self.output.configure(state="normal")
        self.output.insert("end",text+"\n\n")
        self.output.configure(state="disabled")
        if at_end:
            self.output.see("end")

    def poll(self):
        try:
            path = max(RUNS.glob("*/events.jsonl"), key=lambda p:p.parent.name, default=None)
            if path and path != self.path:
                self.path, self.offset = path, 0
                self.append("Menu trial · "+path.parent.name)
            if path:
                with path.open("rb") as stream:
                    stream.seek(self.offset)
                    for line in stream:
                        if not line.endswith(b"\n"):
                            break
                        self.offset += len(line)
                        row = json.loads(line)
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
    root = tk.Tk()
    root.title("Desktop Body · Agent activity")
    root.geometry("480x720")
    ActivityColumn(root).pack(expand=True,fill="both")
    root.mainloop()
