"""Visual, single-click menu selection on a frozen headset view."""
import csv
import io
from functools import lru_cache
from pathlib import Path
import re
import shutil
import subprocess

import cv2
import numpy as np
from PIL import Image, ImageOps

# English labels and tooltip wording observed in the in-game menus.
PRESETS = {
    "new instance": ("Create a new instance of this world", "Create New Instance"),
    "search worlds": ("Click to search all worlds", "Input"),
}


def words(text):
    return " ".join(re.findall(r"\w+", text.casefold()))


def read_lines(image):
    exe = shutil.which("tesseract")
    if not exe:
        raise ValueError("Menu selection requires Tesseract OCR on PATH")
    scale = min(4., 2400 / max(image.size))
    size = tuple(round(n*scale) for n in image.size)
    enlarged = ImageOps.invert(image.convert("L")).resize(size, Image.Resampling.LANCZOS)
    data = io.BytesIO()
    enlarged.save(data, "PNG")
    result = subprocess.run([exe,"stdin","stdout","--psm","11","tsv"],input=data.getvalue(),
                            capture_output=True,timeout=8,check=True,
                            creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
    groups = {}
    for row in csv.DictReader(io.StringIO(result.stdout.decode("utf-8")),delimiter="\t", quoting=csv.QUOTE_NONE):
        if row["level"] != "5" or not row["text"].strip():
            continue
        key = tuple(row[k] for k in ("block_num","par_num","line_num"))
        groups.setdefault(key, []).append(row)
    lines = []
    for group in groups.values():
        x = min(int(r["left"]) for r in group)
        y = min(int(r["top"]) for r in group)
        r = max(int(v["left"])+int(v["width"]) for v in group)
        b = max(int(v["top"])+int(v["height"]) for v in group)
        lines.append((words(" ".join(v["text"] for v in group)), (x+r)/2/size[0], (y+b)/2/size[1]))
    return lines


def laser_tip(image):
    # The virtual right pointer enters from below/right in the left eye.
    rgb = np.asarray(image.convert("RGB").resize((600,672)))
    mask = cv2.inRange(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV),(82,55,95),(106,225,255))
    lines = cv2.HoughLinesP(mask,1,np.pi/360,30,minLineLength=50,maxLineGap=9)
    candidates = []
    for x,y,u,v in (() if lines is None else lines[:,0]):
        if v<y:
            x,y,u,v = u,v,x,y
        if max(x,u)>410 and v>445 and v-y>24:
            candidates.append((np.hypot(u-x,v-y),x/600,y/672))
    return max(candidates)[1:] if candidates else None


@lru_cache(maxsize=5)
def reference(filename):
    path = Path(__file__).with_name("menu_templates") / (filename+".png")
    with Image.open(path) as image:
        return np.asarray(image.convert("L"))


def recognize_references(image):
    """Supplement OCR with tiny, human-read labels from the actual VRChat menu."""
    gray = cv2.cvtColor(np.asarray(image.convert("RGB").resize((600,672))),cv2.COLOR_RGB2GRAY)
    found = []
    references = [("new instance","new_instance"),("new instance","new_instance_hover"),(words(PRESETS["new instance"][0]),"create_tooltip"),
                  (words(PRESETS["new instance"][0]),"create_tooltip_light"),("create new instance","create_heading")]
    for angle in (0,-16,16):
        transform = cv2.getRotationMatrix2D((300,336),angle,1)
        inverse = cv2.invertAffineTransform(transform)
        flat = cv2.warpAffine(gray,transform,(600,672))
        for text,filename in references:
            original = reference(filename)
            for scale in (.85,1.,1.2):
                patch = cv2.resize(original,None,fx=scale,fy=scale)
                scores=cv2.matchTemplate(flat,patch,cv2.TM_CCOEFF_NORMED)
                for _ in range(3):
                    _,score,_,(x,y)=cv2.minMaxLoc(scores)
                    if score < (.88 if filename.startswith("new_instance") else .79):
                        break
                    h,w=patch.shape
                    px,py=inverse @ (x+w/2,y+h/2,1.)
                    hit=(text,px/600,py/672)
                    if not any(t==text and abs(a-hit[1])<.04 and abs(b-hit[2])<.025 for t,a,b in found):
                        found.append(hit)
                    scores[max(0,y-h):y+h,max(0,x-w):x+w]=0
    return found


def inspect_frame(image):
    lines = read_lines(image)
    for text,x,y in recognize_references(image):
        if not any(t==text and abs(a-x)<.04 and abs(b-y)<.025 for t,a,b in lines):
            lines.append((text,x,y))
    return lines, laser_tip(image)


def unchanged(a,b,location):
    if a.size != b.size:
        return False
    w,h = a.size
    x,y = location
    box = (int((x-.04)*w),int((y-.025)*h),int((x+.04)*w),int((y+.025)*h))
    def patch(im):
        return np.asarray(im.crop(box).convert("L"),dtype=float)
    return np.abs(patch(a)-patch(b)).mean()<8


class Selection:
    def __init__(self, name, expected, tooltip, poses, now):
        self.name = words(name)
        preset_tip,preset_screen = PRESETS.get(self.name,("",""))
        self.expected = words(expected or preset_screen)
        self.tooltip = words(tooltip or preset_tip or self.name)
        if not self.name or not self.expected:
            raise ValueError("Specify a visible control and a distinctive expected screen heading")
        self.poses = np.asarray(poses).copy()
        self.phase = "find"
        self.deadline = now+45
        self.after = now
        self.aim = None
        self.attempts = self.matches = 0
        self.pending = None
        self.clicked = False
        self.last_center = None

    def tick(self, solver, image, stamp, now):
        if now>self.deadline:
            return ("Clicked once; destination unverified" if self.clicked else "No click: selection timed out")
        if self.phase=="press":
            if now>=self.after:
                solver.keys.discard("x")
                self.phase="verify"
                self.after=now+.25
            return
        if self.pending is None:
            if stamp<=self.after:
                return
            self.sample=image
            self.after=stamp
            self.pending=solver.reader.submit(inspect_frame,image)
            return
        if not self.pending.done():
            return
        try:
            lines,tip=self.pending.result()
        except (ValueError,OSError,subprocess.SubprocessError) as error:
            return f"{'Clicked once' if self.clicked else 'No click'}: OCR unavailable ({error})"
        self.pending=None
        hits=[(x,y) for label,x,y in lines if label==self.name]
        destination=any(label==self.expected for label,_,_ in lines)
        if self.phase=="verify":
            self.matches = self.matches+1 if destination else 0
            if self.matches>=2:
                return f"Verified {self.expected} after clicking {self.name}"
            return
        if destination:
            return "No click: expected heading already visible"
        if len(hits)>1 or (not hits and self.last_center is None):
            return "No click: control label missing or ambiguous"
        # The laser can cover the lettering. Keep the last position only for
        # aiming; the current tooltip and laser must still agree before clicking.
        if hits:
            self.last_center=np.array(hits[0])
        center=self.last_center
        if self.aim is not None and tip is not None:
            error=center-tip
            tooltips=[(x,y) for label,x,y in lines if y>center[1]+.035
                      and (" "+self.tooltip+" ") in (" "+label+" ")]
            if (abs(error[0])<.033 and abs(error[1])<.034 and len(tooltips)==1
                    and unchanged(self.sample,image,center) and unchanged(self.sample,image,tooltips[0])):
                self.matches+=1
                if self.matches>=2:
                    solver.keys.add("x")
                    self.clicked=True
                    self.matches=0
                    self.phase="press"
                    self.after=solver.pulse_end=now+.12
                else:
                    self.after=now
                return
        self.matches=0
        if self.attempts>=9:
            return "No click: hover could not be confirmed"
        if self.aim is None:
            self.aim=np.clip(center-[.025,.38],0,1)
        elif tip is not None:
            self.aim=np.clip(self.aim+.65*(center-tip),0,1)
        else:
            # Probe around the initial estimate when the ray misses the panel.
            offsets=((0,-.08),(0,.08),(-.07,0),(.07,0),(0,-.16),(0,.16))
            self.aim=np.clip(center-[.025,.38]+offsets[(self.attempts-1)%len(offsets)],0,1)
        solver.hands[2]=solver.pointer_pose(self.poses[0],*self.aim)
        self.attempts+=1
        self.after=now+.22



