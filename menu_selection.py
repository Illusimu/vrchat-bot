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


def calibration_guide_area(image):
    """Measure the large bright blue regions used by VRChat's tracking guides."""
    rgb = np.asarray(image.convert("RGB").resize((480, 374)))
    hue, saturation, value = cv2.split(cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV))
    blue = ((84 <= hue) & (hue <= 118) & (saturation > 168) & (value > 165)).astype(np.uint8)
    _, _, stats, _ = cv2.connectedComponentsWithStats(blue)
    # Ignore the pointer beam, icon fragments, and small matching parts of the avatar.
    areas = [area for _, _, width, height, area in stats[1:] if width >= 16 and height >= 16 and area >= 300]
    edge_blue = ((84 <= hue) & (hue <= 118) & (saturation > 65) & (value > 165)).astype(np.uint8)
    _, _, edge_stats, _ = cv2.connectedComponentsWithStats(edge_blue)
    cropped_guides = [area for _, top, width, height, area in edge_stats[1:]
                      if top + height >= len(blue) - 1 and height >= 1.5 * width
                      and area >= 750 and area >= .45 * width * height]
    return float(max(sum(areas), max(cropped_guides, default=0)) / blue.size)


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


def laser_tip(image, previous=None, target=None, *, minimum_value=95):
    """Locate the upper end of a cyan beam, using an earlier pose when available."""
    rgb = np.asarray(image.convert("RGB").resize((600,672)))
    cyan = cv2.inRange(cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV),(82,55,minimum_value),(106,255,255))
    mask = cyan.copy() if previous is not None else cyan
    if previous is not None:
        old = np.asarray(previous.convert("RGB").resize((600,672)))
        changed = cv2.absdiff(rgb,old).max(axis=2) > 24
        mask[~changed] = 0
    lines = cv2.HoughLinesP(mask,1,np.pi/360,30,minLineLength=50,maxLineGap=9)
    candidates = []
    rows, columns = np.nonzero(cyan)
    for x,y,u,v in (() if lines is None else lines.reshape(-1, 4)):
        if v<y:
            x,y,u,v = u,v,x,y
        if u>360 and v>380 and v-y>35 and u>x+15:
            # Frame differencing and a changing background can split one visible beam.
            # Follow its current color back along the detected line to recover the tip.
            on_beam = np.abs(columns - (x + (rows - y) * (u - x) / (v - y))) <= 4
            tip_y = y
            for row in np.unique(rows[on_beam & (rows < y)])[::-1]:
                if tip_y - row > 16:
                    break
                tip_y = row
            tip_x = x if tip_y == y else float(np.median(columns[on_beam & (rows == tip_y)]))
            tip = (tip_x/600,tip_y/672)
            distance = np.linalg.norm(np.subtract(tip,target)) if target is not None else 0
            candidates.append((np.hypot(u-x,v-y)/(1+3*distance),*tip))
    return max(candidates)[1:] if candidates else None


# The game mirror crops the eye image. This seed still needs visual confirmation.
GAME_MIRROR_AIM_OFFSET = (.09, .54)


class PointerCalibration:
    """Fit screen-space motion from small, observed changes in commanded aim."""

    def __init__(self, target):
        self.target = np.asarray(target, dtype=float)
        self.initial = np.clip(self.target - [.02,.32],0,1)
        self.samples = []
        self.probes = iter(((0,0),(.08,0),(-.08,0),(0,-.065),(0,.065),
                            (.11,-.035),(-.11,-.035),(0,.13)))

    def probe(self):
        offset = next(self.probes, None)
        return None if offset is None else np.clip(self.initial + offset,0,1)

    def correct(self, aim, tip):
        self.samples.append((np.asarray(aim),np.asarray(tip)))
        error = self.target-tip
        if len(self.samples) >= 3:
            commands, tips = map(np.asarray, zip(*self.samples[-6:]))
            jacobian, _, rank, _ = np.linalg.lstsq(commands-commands.mean(axis=0),
                                                   tips-tips.mean(axis=0),rcond=None)
            if rank == 2 and np.linalg.cond(jacobian) < 12:
                step = np.linalg.solve(jacobian.T,error)
                return np.clip(aim+np.clip(step,-.12,.12),0,1)
        if len(self.samples) == 1:
            return np.clip(aim+[.025,0] if aim[0] < .96 else aim-[.025,0],0,1)
        if len(self.samples) == 2:
            return np.clip(aim+[0,.025] if aim[1] < .96 else aim-[0,.025],0,1)
        return np.clip(aim+np.clip(error*.65,-.12,.12),0,1)


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
        self.calibration = None

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
        if self.calibration is None or np.linalg.norm(self.calibration.target-center) > .04:
            self.calibration = PointerCalibration(center)
            self.aim = self.calibration.probe()
        elif self.aim is not None and tip is not None:
            self.aim = self.calibration.correct(self.aim,tip)
        else:
            self.aim = self.calibration.probe()
        if self.aim is None:
            return "No click: pointer unavailable near the control"
        solver.hands[2]=solver.pointer_pose(self.poses[0],*self.aim)
        self.attempts+=1
        self.after=now+.22



