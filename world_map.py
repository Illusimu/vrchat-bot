"""World-scoped stereo keyframes. Coordinates are camera-local estimates, in metres."""
from collections import Counter
import io
import math
from pathlib import Path
import re
import sqlite3
import time

import cv2
import numpy as np
from camera_pose import read_eye
from surface_memory import SurfaceMemory, coverage_keys

K = np.array([[320.,0,319.5],[0,360.,359.5],[0,0,1.]])
BASELINE = .064
KEYFRAME_LIMIT = 500
ROOT = Path(__file__).resolve().parent / "build/world-maps"
ORB = cv2.ORB_create(1600)
MATCH = cv2.BFMatcher(cv2.NORM_HAMMING)
STEREO = cv2.StereoSGBM_create(numDisparities=64,blockSize=5,P1=200,P2=800,
                              uniquenessRatio=12,speckleWindowSize=80,speckleRange=2,disp12MaxDiff=1)


def pack(**arrays):
    data=io.BytesIO()
    np.savez_compressed(data,**arrays)
    return data.getvalue()


def unpack(blob):
    with np.load(io.BytesIO(blob),allow_pickle=False) as archive:
        return dict(archive)


def centered_baseline(pose, screen):
    center=screen[:3]*[1,-1,1]
    offset=pose[:3,:3].T@(center-pose[:3,3])
    if .002 < offset[0] < .5 and np.linalg.norm(offset[1:]) < .001:
        return float(2*offset[0])
    return None


def observe_pair(pair):
    left,right=np.hsplit(np.asarray(pair.convert("RGB")),2)
    gray=cv2.cvtColor(left,cv2.COLOR_RGB2GRAY)
    rg=cv2.cvtColor(right,cv2.COLOR_RGB2GRAY)
    hud=read_eye(left)
    baseline=BASELINE
    if hud is not None:
        other=read_eye(right)
        if other is None:
            baseline=centered_baseline(hud["pose"],hud["screen"])
        else:
            relative=np.linalg.solve(hud["pose"],other["pose"])
            baseline=float(relative[0,3])
            if (not np.allclose(relative[:3,:3],np.eye(3),atol=.001)
                    or np.linalg.norm(relative[1:3,3])>.001 or not .004<baseline<1):
                baseline=None
        if baseline is None:return dict(hud=hud)
    mask=None
    if hud is not None:
        mask=np.full_like(gray,255)
        h,w=gray.shape
        low,high=(.78,.94) if hud["flipped_patch"] else (.06,.22)
        mask[int(low*h)-2:int(high*h)+3,int(.04*w)-2:int(.24*w)+3]=0
    features,desc=ORB.detectAndCompute(gray,mask)
    if desc is None or len(desc)<40:
        return dict(hud=hud) if hud else None
    disparity=STEREO.compute(gray,rg).astype(np.float32)/16
    if mask is not None:
        disparity[mask==0]=-1
    def lift(uv):
        x,y=np.rint(uv).astype(int).T
        d=disparity[y,x]
        z=K[0,0]*baseline/np.maximum(d,.01)
        xyz=np.column_stack(((uv[:,0]-K[0,2])*z/K[0,0],(uv[:,1]-K[1,2])*z/K[1,1],z))
        xyz[(d<1.4)|(z<.35)] = np.nan
        return xyz.astype(np.float32)
    uv=np.float32([p.pt for p in features])
    stride=4 if hud is not None else 8
    ys,xs=np.mgrid[4:716:stride,4:636:stride]
    pixels=np.column_stack((xs.ravel(),ys.ravel()))
    pts=lift(pixels)
    valid=np.isfinite(pts[:,0])
    xyz=lift(uv)
    if valid.sum()<80 or np.isfinite(xyz[:,0]).sum()<22:
        return dict(hud=hud) if hud else None
    jpg=cv2.imencode(".jpg",cv2.cvtColor(left,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,72])[1].tobytes()
    return dict(uv=uv,desc=desc,xyz=xyz,pts=pts[valid],rgb=left[pixels[valid,1],pixels[valid,0]],jpg=jpg,hud=hud,baseline=baseline)


def locate(reference, observation):
    matches=MATCH.knnMatch(reference["desc"],observation["desc"],k=2)
    good=[m for pair in matches if len(pair)==2 for m,n in [pair]
          if m.distance<.74*n.distance and np.isfinite(reference["xyz"][m.queryIdx,0])]
    if len(good)<22:return None
    a=np.float32([reference["xyz"][m.queryIdx] for m in good])
    b=np.float32([observation["uv"][m.trainIdx] for m in good])
    ok,r,t,inliers=cv2.solvePnPRansac(a,b,K,None,iterationsCount=120,reprojectionError=2.8,
                                      flags=cv2.SOLVEPNP_EPNP)
    if not ok or inliers is None or len(inliers)<max(20,.42*len(good)):return None
    ij=inliers.ravel()
    if cv2.contourArea(cv2.convexHull(b[ij]))<640*720*.025:return None
    R=cv2.Rodrigues(r)[0]
    current_xyz=observation["xyz"][[good[i].trainIdx for i in ij]]
    predicted=a[ij] @ R.T + t.ravel()
    agree=np.abs(predicted[:,2]-current_xyz[:,2]) < .2+.12*predicted[:,2]
    if agree.sum()<12:return None
    delta=np.eye(4)
    delta[:3,:3]=R.T
    delta[:3,3]=(-R.T@t).ravel()
    return reference["pose"]@delta,len(ij)


def frame_coverage(frame):
    points=frame["pts"]@frame["pose"][:3,:3].T+frame["pose"][:3,3]
    return [bytes(k) for k in coverage_keys(points)]


class WorldMap:
    def __init__(self,root=ROOT):
        self.root=Path(root)
        self.db=None
        self.identity=None
        self.frames=[]
        self.active=None
        self.pose=None
        self.misses=0
        self.revision=0
        self.status="Waiting for a confirmed scene"
        self.name=""
        self.world=""
        self.jump=None
        self.hud=None
        self.memory=None
        self.coverage=Counter()
        self.stats={}

    def switch(self,info):
        identity=(info.get("world_id"),info.get("instance_key")) if info.get("scene_ready") else None
        if identity==self.identity:return
        self.close()
        self.identity=identity
        self.frames=[]
        self.active=self.pose=self.jump=None
        self.hud=None
        self.memory=None
        self.coverage=Counter()
        self.stats={}
        self.last_good=None
        self.candidate=None
        self.misses=0
        self.revision+=1
        self.name=self.world=""
        self.status="Loading or between worlds"
        if not identity:return
        world,instance=identity
        if not re.fullmatch(r"wrld_[0-9a-fA-F-]{36}",world):
            raise ValueError("Invalid world ID")
        self.world=world
        self.name=info.get("world_name",world)
        self.root.mkdir(exist_ok=True,parents=True)
        self.db=sqlite3.connect(self.root/(world+".sqlite3"))
        self.db.execute("CREATE TABLE IF NOT EXISTS frames(id INTEGER PRIMARY KEY,region INTEGER,seen REAL,instance TEXT,photo BLOB,data BLOB,label TEXT DEFAULT '')")
        self.db.execute("CREATE TABLE IF NOT EXISTS breaks(seen REAL,instance TEXT,source INTEGER,destination INTEGER)")
        self.db.execute("""CREATE TABLE IF NOT EXISTS surfaces(
            x INTEGER,y INTEGER,z INTEGER,rgb INTEGER,seen REAL,
            PRIMARY KEY(x,y,z)) WITHOUT ROWID""")
        for fid,region,stamp,label,data in self.db.execute("SELECT id,region,seen,label,data FROM frames ORDER BY id"):
            arrays=unpack(data)
            if region==-1 and "baseline" not in arrays:
                if "screen_pose" not in arrays:continue
                baseline=centered_baseline(arrays["pose"],arrays["screen_pose"])
                if baseline is None:continue
                for key in ("pts","xyz"):
                    arrays[key]*=baseline/BASELINE
            self.frames.append(dict(id=fid,region=region,seen=stamp,label=label,**arrays))
        self.memory=SurfaceMemory(self.db)
        with self.db:
            self.db.execute("CREATE TABLE IF NOT EXISTS map_schema(version INTEGER PRIMARY KEY)")
            if not self.db.execute("SELECT 1 FROM map_schema WHERE version=1").fetchone():
                self.memory.import_confirmed(self.db.execute("SELECT x,y,z,rgb,seen FROM surfaces"))
                for f in self.frames:
                    if f["region"]==-1:
                        self.memory.integrate(f,f["seen"],K)
                self.db.execute("DELETE FROM surfaces")
                self.db.execute("INSERT INTO map_schema VALUES(1)")
        for f in self.frames:
            if f["region"]==-1:
                f["coverage"]=frame_coverage(f)
                self.coverage.update(f["coverage"])
        self.refresh_stats()
        self.status="Looking for a remembered view"

    def close(self):
        if self.db:self.db.close()
        self.db=None
        self.identity=None

    def refresh_stats(self):
        self.memory.cloud=self.memory.cloud_arrays()
        count=int(self.memory.cloud[2].sum())
        settled=sum(int(np.count_nonzero(rows["settled"] & (rows["miss"]==0))) for rows in self.memory.tiles.values())
        db_path=self.root/(self.world+".sqlite3")
        photo_bytes=self.db.execute("SELECT COALESCE(SUM(length(photo)),0) FROM frames").fetchone()[0]
        self.stats=dict(storage_bytes=db_path.stat().st_size,photo_bytes=photo_bytes,
                        confirmed_cells=count,settled_cells=settled,provisional_cells=len(self.memory.cloud[2])-count,
                        keyframes=len(self.frames),keyframe_limit=KEYFRAME_LIMIT)

    def retention(self,new):
        # A reference is valuable when it alone, or nearly alone, covers surface patches.
        candidates=[f for f in self.frames if "coverage" in f and not f["label"]]
        if "coverage" in new:candidates.append(new)
        if not candidates:return None
        added=set(new.get("coverage",[]))
        def contribution(f):
            cells=f["coverage"]
            return sum(1/(self.coverage[k]+(k in added))**2 for k in cells)
        return min(candidates,key=lambda f:(contribution(f),f["seen"]))

    def update(self,ob,now):
        self.hud=ob.get("hud") if ob else None
        if ob is None:
            self.pose=None
            self.status="Insufficient stereo detail"
            return
        # Recent references first; a longer search across rooms runs on loss.
        refs=sorted(self.frames,key=lambda f:(f["region"]==self.active,f["id"]),reverse=True)
        fit=None
        hud=ob.get("hud")
        for f in ([] if hud is not None else refs):
            if f["region"]==-1:continue  # A visual-only depth frame has no verified world scale.
            solved=locate(f,ob)
            if solved is not None:
                pose,n=solved
                fit=(f,pose,n)
                break
        old=self.active
        if hud is not None:
            self.active=-1  # Reserved for points anchored to the Unity world origin.
            pose=hud["pose"]
            self.status="Unity world pose from avatar HUD"
        elif fit:
            f,pose,n=fit
            self.active=f["region"]
            self.status=f"Tracking region {self.active} · {n} visual matches"
        else:
            self.misses+=1
            self.pose=None
            self.status="Location uncertain · checking for a view change"
            if self.misses<3:return
            # Require two geometrically consistent views before seeding a new region.
            if getattr(self,"candidate",None) is None or locate(self.candidate,ob) is None:
                self.candidate=dict(ob,pose=np.eye(4))
                return
            self.active=1+max([0]+[f["region"] for f in self.frames])
            pose=np.eye(4)
            self.status=f"New local region {self.active}"
        self.candidate=None
        self.misses=0
        displaced=False
        if self.last_good and old==self.active:
            previous,when=self.last_good
            displaced=np.linalg.norm(pose[:3,3]-previous[:3,3])>3.5+4*min(now-when,2)
        if old is not None and (old!=self.active or displaced):
            self.jump=(old,self.active,now)
            with self.db:
                self.db.execute("INSERT INTO breaks VALUES(?,?,?,?)",(now,self.identity[1],old,self.active))
                self.db.execute("DELETE FROM breaks WHERE rowid < (SELECT max(rowid)-999 FROM breaks)")
        self.pose=pose
        self.last_good=(pose,now)
        if "pts" not in ob:return
        grounded=hud is not None
        if grounded and not self.memory.sample_due(pose,now):return
        data={k:ob[k] for k in ("xyz","desc","pts","rgb")}
        data.update(pose=pose,baseline=ob["baseline"])
        if grounded:
            data.update(screen_pose=hud["screen"],photo_pose=hud["photo"],photo_available=hud["photo_available"])
        new=dict(region=self.active,seen=now,label="",**data)
        save=True
        if grounded:
            new["coverage"]=frame_coverage(new)
            cells=new["coverage"]
            # Keep some stereo references even when all geometry is already observed once.
            gain=sum(self.coverage[k]<2 for k in cells)/max(1,len(cells))
            save=gain>=.12
        for f in self.frames:
            if f["region"]!=self.active:continue
            d=np.linalg.norm(pose[:3,3]-f["pose"][:3,3])
            trace=np.trace(pose[:3,:3].T@f["pose"][:3,:3])
            if d<(.12 if grounded else .38) and trace>1+2*math.cos(math.radians(8 if grounded else 18)):
                save=False
                break
        retired=None
        if save and len(self.frames)>=KEYFRAME_LIMIT:
            retired=self.retention(new)
            if retired is None or retired is new:save=False;retired=None
        elif save and grounded:
            # Release a fully superseded reference before reaching the hard limit.
            extra=set(new["coverage"])
            retired=next((f for f in self.frames if not f["label"] and f.get("coverage")
                          and all(self.coverage[k]+(k in extra)>=3 for k in f["coverage"])),None)
        if not grounded and not save:return
        try:
            with self.db:
                if grounded:self.memory.integrate(new,now,K)
                if save:
                    if retired is not None:self.db.execute("DELETE FROM frames WHERE id=?",(retired["id"],))
                    cur=self.db.execute("INSERT INTO frames(region,seen,instance,photo,data) VALUES(?,?,?,?,?)",
                                        (self.active,now,self.identity[1],ob["jpg"],pack(**data)))
                    new["id"]=cur.lastrowid
        except Exception:
            # Discard tentative tile edits after SQLite rolls back.
            self.memory=SurfaceMemory(self.db)
            raise
        if retired is not None:
            self.frames.remove(retired)
            self.coverage.subtract(retired["coverage"])
            self.coverage+=Counter()
        if save:
            self.frames.append(new)
            self.coverage.update(new.get("coverage",[]))
        self.refresh_stats()
        self.revision+=1

    def photo(self,fid):
        row=self.db.execute("SELECT photo FROM frames WHERE id=?",(fid,)).fetchone()
        if row is None:raise ValueError("Unknown keyframe")
        return row[0]

    def snapshot(self):
        return dict(world=self.name,world_id=self.world,region=self.active,pose=self.pose,
                    status=self.status,revision=self.revision,jump=self.jump,
                    hud=self.hud,surfaces=self.memory.cloud if self.memory else None,stats=self.stats,frames=list(self.frames))


class MappingWorker:
    """One serial worker owns capture, database switches and all writes."""
    def __init__(self):
        self.map=WorldMap()
        self.ready_since=0

    def step(self,allow_capture=True):
        from vr_agent import observe
        from world_context import world_context
        info=world_context()
        identity=self.map.identity
        self.map.switch(info)
        if self.map.identity!=identity:
            self.map.candidate=None
            self.ready_since=time.monotonic()
        if self.map.db and allow_capture and time.monotonic()-self.ready_since>2:
            pair=observe(stereo=True)
            after=world_context()
            if after!=info:
                self.map.switch(after)
                self.map.candidate=None
                self.ready_since=time.monotonic()
            else:
                self.map.update(observe_pair(pair),time.time())
        return self.map.snapshot()
