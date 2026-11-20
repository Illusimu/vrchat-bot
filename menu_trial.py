"""Observe and audit a local planner navigating the running Desktop Body menu."""
import argparse
import asyncio
import base64
from datetime import datetime
import html
import io
import json
from pathlib import Path
import time
import cv2
import numpy as np
from menu_selection import laser_tip, read_lines

from PIL import Image
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from vr_agent import decide


class Trace:
    def __init__(self, objective):
        self.path = Path("build/menu-runs") / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.path.mkdir(parents=True)
        self.objective = objective
        self.events = []
        self.log("objective", text=objective, model="vr-agent")
        print(self.path.resolve(), flush=True)

    def log(self, event, **fields):
        row = dict(time=datetime.now().astimezone().isoformat(timespec="seconds"), event=event, **fields)
        self.events.append(row)
        with (self.path/"events.jsonl").open("a", encoding="utf-8") as out:
            out.write(json.dumps(row, ensure_ascii=False)+"\n")
        blocks = []
        for item in self.events:
            fields = {k:v for k,v in item.items() if k!="image"}
            blocks.append("<pre>"+html.escape(json.dumps(fields,indent=2,ensure_ascii=False))+"</pre>")
            if "image" in item:
                blocks.append('<a href="{0}"><img width="330" src="{0}"></a>'.format(item["image"]))
        page = "<!doctype html><meta charset=utf-8><title>Local model menu trace</title>"
        if event!="finished": page += '<meta http-equiv="refresh" content="5">'
        page += "<style>body{font:16px system-ui;max-width:900px;margin:auto}pre{white-space:pre-wrap;background:#eee;padding:12px}</style>"
        temp = self.path/"index.tmp"
        temp.write_text(page+"<h1>"+html.escape(self.objective)+"</h1>"+"".join(blocks),encoding="utf-8")
        temp.replace(self.path/"index.html")


async def run(objective, panel=None):
    trace = Trace(objective)
    history = []
    aligned = False
    if panel:
        # Four measured corners in the normalized left-eye frame (TL,TR,BR,BL).
        corners = np.float32(panel).reshape(4,2)
        unit = np.float32([[0,0],[1,0],[1,1],[0,1]])
        to_eye = cv2.getPerspectiveTransform(unit,corners)
        trace.log("operator_setup",panel=corners.tolist(),note="Wrist menu stabilized and four corners supplied. Aim uses visual pointer feedback.")

    try:
        async with streamable_http_client("http://127.0.0.1:29746/mcp") as (r,w,*_):
            async with ClientSession(r,w) as client:
                await client.initialize()
                for step in range(1,31):
                    trace.log("call",step=step,tool="observe",arguments={"view":"headset"})
                    result = await client.call_tool("observe",{"view":"headset"})
                    if result.isError: raise ValueError(result.content)
                    data = base64.b64decode(result.content[0].data)
                    filename = f"{step:02}-headset.jpg"
                    (trace.path/filename).write_bytes(data)
                    trace.log("observation",step=step,image=filename)
                    start = time.monotonic()
                    observation = Image.open(io.BytesIO(data))
                    context = None
                    if panel:
                        width,height = observation.size
                        transform = cv2.getPerspectiveTransform(np.float32(corners*[width,height]),np.float32(unit*[640,800]))
                        observation = Image.fromarray(cv2.warpPerspective(np.asarray(observation),transform,(640,800)))
                        name = f"{step:02}-model.jpg"
                        observation.save(trace.path/name)
                        labels = read_lines(observation)
                        context = {"visible_text_normalized_xy": labels, "instruction":"This image is the wrist menu rectified to fill the frame. Aim in these image coordinates. Read current labels and hover tooltip on every step. A completed click does not establish a page change."}
                        trace.log("model_input",step=step,image=name,ocr=labels)
                    decision = await asyncio.to_thread(decide, objective,history,observation,body_state=context)
                    kind = decision["kind"]
                    trace.log("decision",step=step,seconds=round(time.monotonic()-start,2),decision=decision)
                    print(step,kind,decision["evidence"],flush=True)
                    if kind in ("blocked","done"):
                        trace.log("finished",status=kind,model_report=decision["evidence"])
                        return
                    if kind in ("menu_toggle","menu_click","menu_aim"):
                        tool,args = "menu",{"action":kind[5:]}
                        if kind=="menu_aim": args.update(zip(("x","y"),decision["values"]))
                    elif kind=="select_by_name":
                        tool,args="select_by_name",{k:decision[k] for k in ("label","expected","tooltip")}
                    elif kind in ("view","wrist"):
                        tool,args="target",dict(kind=kind,values=decision["values"],side=decision["side"])
                    else:
                        decision["outcome"]="This menu trial supports menu, selection, view and wrist controls."
                        trace.log("unavailable",step=step,outcome=decision["outcome"])
                        history.append(decision)
                        continue
                    if panel and tool=="menu" and args["action"]=="aim":
                        xy = np.float32([[[args["x"],args["y"]]]])
                        goal = cv2.perspectiveTransform(xy,to_eye)[0,0]
                        aim = np.clip(goal-[.025,.38],0,1)
                        aligned = False
                        for attempt in range(5):
                            command = dict(action="aim",x=float(aim[0]),y=float(aim[1]))
                            trace.log("call",step=step,tool=tool,arguments=command)
                            reply = await client.call_tool(tool,command)
                            trace.log("result",step=step,error=reply.isError,outcome=str(reply.content))
                            if reply.isError: break
                            await asyncio.sleep(.3)
                            trace.log("call",step=step,tool="observe",arguments={"view":"headset"})
                            shot = await client.call_tool("observe",{"view":"headset"})
                            frame = base64.b64decode(shot.content[0].data)
                            tip = laser_tip(Image.open(io.BytesIO(frame)))
                            name = f"{step:02}-aim-{attempt}.jpg"
                            (trace.path/name).write_bytes(frame)
                            trace.log("pointer_check",step=step,image=name,tip=tip,desired=goal.tolist())
                            if tip is None: break
                            error = goal-tip
                            if np.max(abs(error))<.018:
                                aligned=True
                                break
                            aim=np.clip(aim+.75*error,0,1)
                        decision["outcome"]="Pointer aligned; inspect fresh hover" if aligned else "Pointer alignment unconfirmed; adjust target"
                        history.append(decision)
                        continue
                    if panel and tool=="menu" and args["action"]=="click" and not aligned:
                        decision["outcome"]="Click withheld: aim was unconfirmed"
                        trace.log("withheld",step=step,outcome=decision["outcome"])
                        history.append(decision)
                        continue
                    aligned=False
                    trace.log("call",step=step,tool=tool,arguments=args)
                    result = await client.call_tool(tool,args)
                    decision["outcome"]=" ".join(i.text for i in result.content if i.type=="text")
                    trace.log("result",step=step,error=result.isError,outcome=decision["outcome"])
                    history.append(decision)
                    await asyncio.sleep(.6)
        trace.log("finished",status="step limit")
    except BaseException as error:
        trace.log("finished",status="interrupted",detail=str(error))
        raise

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("objective")
    parser.add_argument("--panel",type=float,nargs=8)
    args=parser.parse_args()
    asyncio.run(run(args.objective,args.panel))
