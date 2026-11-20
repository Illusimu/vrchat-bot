"""Interactive, independent camera for the current world's stereo map."""
import datetime
import json
import math
from pathlib import Path
import sqlite3
import time
import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
from PIL import Image,ImageTk

from world_map import ROOT


class MapView:
    def __init__(self,root):
        self.win=root
        self.photo_panel=None
        bar=ttk.Frame(self.win);bar.pack(fill="x")
        self.region=tk.StringVar()
        ttk.Button(bar,text="Go to live pose",command=self.follow).pack(side="left")
        ttk.Button(bar,text="Save memory view",command=self.save).pack(side="left")
        self.stable=tk.BooleanVar(value=True)
        ttk.Checkbutton(bar,text="Confirmed only",variable=self.stable,command=self.rebuild).pack(side="left")
        self.info=tk.StringVar()
        ttk.Label(self.win,textvariable=self.info).pack(anchor="w")
        self.stats=tk.StringVar()
        ttk.Label(self.win,textvariable=self.stats).pack(anchor="w")
        self.canvas=tk.Canvas(self.win,bg="#10171d",highlightthickness=0,takefocus=True)
        self.canvas.pack(fill="both",expand=True)
        ttk.Label(self.win,text="Right-drag to look · WASD to fly · Q/E down/up · Wheel to dolly · Click a camera dot to see its photo").pack(anchor="w")
        self.canvas.bind("<Button-1>",self.pick)
        self.canvas.bind("<Button-3>",self.start_drag)
        self.canvas.bind("<B3-Motion>",self.drag)
        self.canvas.bind("<MouseWheel>",lambda e:self.move(0,0,np.sign(e.delta)*.45))
        self.keys=set()
        self.canvas.bind("<KeyPress>",lambda e:self.keys.add(e.keysym.lower()))
        self.canvas.bind("<KeyRelease>",lambda e:self.keys.discard(e.keysym.lower()))
        self.canvas.bind("<FocusOut>",lambda e:self.keys.clear())
        self.state={}
        self.cloud=np.empty((0,3))
        self.position=np.array([0.,-1.,-2.])
        self.yaw=self.pitch=0.
        self.lastrev=None
        self.dots=[]
        self.shot=None
        self.draw()

    def update(self,state):
        old=self.state
        self.state=state
        self.info.set(f"{state['world'] or 'Between worlds'}   ·   {state['status']}")
        stats=state.get("stats")
        self.stats.set((f"{stats['storage_bytes']/1048576:.1f} MiB on disk  ·  "
                        f"{stats['keyframes']}/{stats['keyframe_limit']} photos  ·  "
                        f"{stats['provisional_cells']:,} provisional  ·  {stats['confirmed_cells']:,} confirmed"
                        f" ({stats['settled_cells']:,} settled)")
                       if stats else "")
        rev=(state["world_id"],state["revision"],state["region"])
        if self.lastrev!=rev:
            surfaces=state.get("surfaces")
            grounded=(surfaces is not None and len(surfaces[0])>0) or state["region"]==-1
            selected="World" if grounded else self.region_name(state["region"])
            if old.get("world_id")!=state["world_id"] or selected!=self.region.get():
                self.region.set(selected)
                if selected==self.region_name(state["region"]):
                    self.follow()
                elif grounded:
                    last=next((f for f in reversed(state["frames"]) if f["region"]==-1),None)
                    if last:self.face_pose(last["pose"])
                self.dismiss_photo()
            self.rebuild()
            self.lastrev=rev

    @staticmethod
    def region_name(region):
        return "World" if region==-1 else str(region or "")

    def rebuild(self):
        p,c=[],[]
        if self.region.get()=="World" and self.state.get("surfaces") is not None:
            points,colors,confirmed=self.state["surfaces"]
            if self.stable.get():points,colors=points[confirmed],colors[confirmed]
            self.cloud,self.colors=points,colors
            return
        voxel=.1
        for f in self.state.get("frames",[]):
            if self.region_name(f["region"])!=self.region.get():continue
            a=f["pts"]@f["pose"][:3,:3].T+f["pose"][:3,3]
            _,idx=np.unique(np.floor(a/voxel).astype(int),axis=0,return_index=True)
            p.append(a[idx]);c.append(f["rgb"][idx])
        if not p:
            self.cloud=np.empty((0,3));return
        p=np.vstack(p);c=np.vstack(c)
        _,idx,n=np.unique(np.floor(p/voxel).astype(int),axis=0,return_index=True,return_counts=True)
        if self.stable.get():
            idx=idx[n>=2]
        self.cloud=p[idx];self.colors=c[idx]

    def follow(self):
        if self.region.get()!=self.region_name(self.state.get("region")):return
        p=self.state.get("pose")
        if p is None:
            self.position=np.array([0.,-1.,-2.]);self.yaw=self.pitch=0.
            return
        self.face_pose(p)

    def face_pose(self,p):
        self.position=p[:3,3]-1.6*p[:3,2]-.55*p[:3,1]
        forward=p[:3,2]
        self.yaw=math.atan2(forward[0],forward[2])
        self.pitch=math.asin(np.clip(forward[1],-1,1))+.15
        self.rebuild()

    def basis(self):
        sy,cy=math.sin(self.yaw),math.cos(self.yaw)
        sp,cp=math.sin(self.pitch),math.cos(self.pitch)
        return np.array([[cy,-sy*sp,sy*cp],[0,cp,sp],[-sy,-cy*sp,cy*cp]])

    def move(self,x,y,z):
        self.position+=self.basis()@np.array([x,y,z])

    def start_drag(self,e):
        self.canvas.focus_set()
        self.xy=e.x,e.y

    def drag(self,e):
        self.yaw+=(e.x-self.xy[0])*.006
        self.pitch=np.clip(self.pitch+(e.y-self.xy[1])*.006,-1.48,1.48)
        self.xy=e.x,e.y

    def draw(self):
        if self.win.winfo_viewable() and self.photo_panel is None:
            k=self.keys
            self.move(.12*(("d" in k)-("a" in k)),.12*(("q" in k)-("e" in k)),.12*(("w" in k)-("s" in k)))
            w=max(64,self.canvas.winfo_width());h=max(64,self.canvas.winfo_height())
            im=np.full((h,w,3),(16,23,29),np.uint8)
            def project(pts):
                local=(pts-self.position)@self.basis()
                uv=local[:,:2]*(.75*h/np.maximum(.04,local[:,2:]))+[w/2,h/2]
                return uv,local[:,2]
            if len(self.cloud):
                uv,z=project(self.cloud)
                ids=np.flatnonzero((z>.08)&(uv[:,0]>=2)&(uv[:,0]<w-2)&(uv[:,1]>=2)&(uv[:,1]<h-2))
                ids=ids[np.argsort(-z[ids])]
                x,y=uv[ids].astype(int).T
                for dx,dy in ((0,0),(-1,0),(1,0),(0,-1),(0,1)):im[y+dy,x+dx]=self.colors[ids]
            self.dots=[]
            for f in self.state.get("frames",[]):
                if self.region_name(f["region"])!=self.region.get():continue
                pos=f["pose"][:3,3]
                uv,z=project(np.array([pos,pos+.22*f["pose"][:3,2]]))
                if min(z)>.08 and np.max(np.abs(uv))<30000:
                    a,b=uv.astype(int)
                    cv2.line(im,tuple(a),tuple(b),(247,183,83),2)
                    cv2.circle(im,tuple(a),4,(247,183,83),-1);self.dots.append((a,f))
            pose=self.state.get("pose")
            if pose is not None and self.region_name(self.state["region"])==self.region.get():
                uv,z=project(pose[:3,3][None])
                if z[0]>.08 and np.max(np.abs(uv))<30000:cv2.circle(im,tuple(uv[0].astype(int)),8,(70,235,177),2)
            self.shot=Image.fromarray(im)
            self.tkimage=ImageTk.PhotoImage(self.shot);self.canvas.delete("all")
            self.canvas.create_image(0,0,anchor="nw",image=self.tkimage)
            self.canvas.create_text(12,12,anchor="nw",fill="white",text=f"Memory camera  {self.position.round(2)}   ·   amber: saved cameras   green: live pose")
        self.win.after(45,self.draw)

    def pick(self,e):
        self.canvas.focus_set()
        near=[(np.linalg.norm(a-[e.x,e.y]),f) for a,f in self.dots]
        if not near:return
        dist,f=min(near,key=lambda item:item[0])
        if dist>14:return
        import io
        db=sqlite3.connect((ROOT/(self.state["world_id"]+".sqlite3")).as_uri()+"?mode=ro",uri=True)
        try:row=db.execute("SELECT photo FROM frames WHERE id=?",(f["id"],)).fetchone()
        finally:db.close()
        if row is None:return
        photo=row[0]
        self.dismiss_photo()
        self.photo_panel=ttk.Frame(self.canvas)
        self.photo_panel.place(x=0,y=0,relwidth=1,relheight=1)
        ttk.Button(self.photo_panel,text="Back to map",command=self.dismiss_photo).pack()
        remembered=Image.open(io.BytesIO(photo))
        remembered.thumbnail((max(1,self.canvas.winfo_width()),max(1,self.canvas.winfo_height()-52)))
        self.photo=ImageTk.PhotoImage(remembered)
        ttk.Label(self.photo_panel,text=datetime.datetime.fromtimestamp(f["seen"]).isoformat(timespec="seconds")).pack()
        ttk.Label(self.photo_panel,image=self.photo).pack()

    def dismiss_photo(self):
        if self.photo_panel is not None:
            self.photo_panel.destroy()
            self.photo_panel=None

    def save(self):
        if self.shot is None or not self.region.get() or not self.state.get("world_id"):return
        folder=ROOT/self.state["world_id"];folder.mkdir(exist_ok=True)
        path=folder/f"memory-{time.time_ns()}.png"
        self.shot.save(path)
        path.with_suffix(".json").write_text(json.dumps(dict(region=self.region.get(),position=self.position.tolist(),
                yaw=self.yaw,pitch=self.pitch,source="reconstruction",world=self.state["world_id"])))
        self.info.set(f"Saved {path.name} · {self.state['world']}")
