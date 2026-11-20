"""Visibility-aware surface evidence in sparse 1.6 metre tiles, stored per world."""
import zlib

import cv2
import numpy as np

CELL = .05
SIDE = 32
# One vote per accepted observation. Separate hit and free-space counters
# preserve confirmed walls behind foreground occluders.
LEGACY = np.dtype([("slot","<u2"),("rgb","u1",3),("hits","u1"),("miss","u1"),("seen","<f8")])
RECORD = np.dtype(LEGACY.descr+[("first","<f8"),("direction","i1",3),("spread","u1"),
                                ("appearance","u1"),("score","u1"),("settled","?")])
FORMAT = b"DBPOSE2:"
CONFIRM = 3
PENDING_TTL = 600


def decode_tile(blob):
    if blob.startswith(FORMAT):
        return np.frombuffer(zlib.decompress(blob[len(FORMAT):]),RECORD)
    old=np.frombuffer(zlib.decompress(blob),LEGACY)
    rows=np.zeros(len(old),RECORD)
    for name in LEGACY.names:rows[name]=old[name]
    rows["first"]=old["seen"]
    rows["score"]=np.minimum(old["hits"],12)*3
    return rows


def update_confidence(rows,dest,direction,color,now):
    # Direction is from this fixed world cell to the eye. Head rotation alone
    # cannot increase the angular evidence for a stationary viewer.
    ref=rows["direction"][dest].astype(np.float32)
    empty=~ref.any(axis=1)
    ref[empty]=np.rint(direction[empty]*127)
    rows["direction"][dest]=ref
    dot=(ref*direction).sum(axis=1)/np.maximum(1,np.linalg.norm(ref,axis=1))
    angle=np.degrees(np.arccos(np.clip(dot,-1,1))).astype(np.uint8)
    rows["spread"][dest]=np.maximum(rows["spread"][dest],angle)
    delta=np.abs(color.astype(np.int16)-rows["rgb"][dest].astype(np.int16)).mean(axis=1)
    consistent=(delta<=24)|(rows["hits"][dest]==0)
    appearance=np.clip(rows["appearance"][dest].astype(int)+np.where(consistent,1,-2),0,8)
    rows["appearance"][dest]=appearance
    hits=np.minimum(rows["hits"][dest].astype(int)+1,12)
    age=np.clip((now-rows["first"][dest])/600,0,1)
    score=(35*hits/12+25*np.minimum(rows["spread"][dest]/30,1)+25*age+15*appearance/8)
    rows["score"][dest]=np.floor(score).astype(np.uint8)
    rows["settled"][dest]=score>=99.999


def coords(slots):
    s=slots.astype(np.int32)
    return np.column_stack((s//1024,(s//32)%32,s%32))


def coverage_keys(points):
    # 20 cm samples for reference-photo coverage; full precision stays in the surface map.
    grid=np.ascontiguousarray(np.floor(points/.2),dtype=np.int32)
    return np.unique(grid.view(np.dtype((np.void,12))).ravel())


class SurfaceMemory:
    def __init__(self,db):
        self.db=db
        db.execute("CREATE TABLE IF NOT EXISTS surface_tiles(x INTEGER,y INTEGER,z INTEGER,data BLOB,PRIMARY KEY(x,y,z)) WITHOUT ROWID")
        self.tiles={(x,y,z):decode_tile(b) for x,y,z,b in db.execute("SELECT * FROM surface_tiles")}
        self.recent=[]
        self.last_sample=None
        self.last_prune=0
        self.cloud=self.cloud_arrays()

    def cloud_arrays(self):
        points=[];colors=[];confirmed=[]
        for key,rows in self.tiles.items():
            points.append(((coords(rows["slot"])+np.array(key)*SIDE+.5)*CELL).astype(np.float32))
            colors.append(rows["rgb"]);confirmed.append(rows["hits"]>=CONFIRM)
        if not points:return np.empty((0,3),np.float32),np.empty((0,3),np.uint8),np.empty(0,bool)
        return np.concatenate(points),np.concatenate(colors),np.concatenate(confirmed)

    def due(self,pose,now):
        # Nearby observations within a burst aren't independent confirmations.
        self.recent=[(p,t) for p,t in self.recent if now-t<15]
        for p,_ in self.recent:
            if np.linalg.norm(p[:3,3]-pose[:3,3])<.18 and np.trace(p[:3,:3].T@pose[:3,:3])>2.95:
                return False
        return True

    def sample_due(self,pose,now):
        if self.last_sample is None:return True
        p,t=self.last_sample
        return (now-t>=3 or np.linalg.norm(p[:3,3]-pose[:3,3])>=.05
                or np.trace(p[:3,:3].T@pose[:3,:3])<2.995)

    def integrate(self,frame,now,K):
        pose=frame["pose"];cam=frame["pts"]
        vote=self.due(pose,now)
        far=cam[:,2].max()
        world=cam@pose[:3,:3].T+pose[:3,3]
        vox=np.floor(world/CELL).astype(np.int32)
        tiles=vox//SIDE
        keys,group=np.unique(tiles,axis=0,return_inverse=True)
        order=np.argsort(group,kind="stable")
        groups=np.split(order,np.flatnonzero(np.diff(group[order]))+1)
        observed={tuple(key):inds for key,inds in zip(keys,groups)}
        # Conservative depth buffer. A hole or nearby silhouette prevents a clearing vote.
        uv=cam[:,:2]/cam[:,2,None]*[K[0,0],K[1,1]]+K[:2,2]
        ij=np.floor(uv/8).astype(int)
        depth=np.full((90,80),np.inf,np.float32)
        valid=(ij[:,0]>=0)&(ij[:,0]<80)&(ij[:,1]>=0)&(ij[:,1]<90)
        np.minimum.at(depth,(ij[valid,1],ij[valid,0]),cam[valid,2])
        depth[~np.isfinite(depth)]=0
        depth=cv2.erode(depth,np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)
        prune=now-self.last_prune>=60
        updated={}
        for key in self.tiles.keys() | observed.keys():
            old=self.tiles.get(key,np.empty(0,RECORD))
            inds=observed.get(key)
            # Skip tiles wholly outside the observed volume unless provisional points expired.
            center=((np.array(key)+.5)*SIDE*CELL-pose[:3,3])@pose[:3,:3]
            radius=SIDE*CELL*.87
            in_view=(center[2]+radius>0 and center[2]-radius<far and
                     max(abs(center[0]),abs(center[1]))<=center[2]+2*radius)
            expired=(old["hits"]<CONFIRM)&(old["seen"]<now-PENDING_TTL) if prune else np.zeros(len(old),bool)
            if inds is None and not in_view and not expired.any():continue
            rows=old.copy()
            if vote and len(rows) and in_view:
                pts=((coords(rows["slot"])+np.array(key)*SIDE+.5)*CELL-pose[:3,3])@pose[:3,:3]
                z=pts[:,2]
                pixel=np.floor((pts[:,:2]/np.maximum(.01,z[:,None])*[K[0,0],K[1,1]]+K[:2,2])/8).astype(int)
                visible=(z>.15)&(pixel[:,0]>=0)&(pixel[:,0]<80)&(pixel[:,1]>=0)&(pixel[:,1]<90)
                sample=depth[pixel[:,1].clip(0,89),pixel[:,0].clip(0,79)]
                free=visible & (sample>z+.12+.04*z)
                rows["miss"][free]=np.minimum(10,rows["miss"][free]+1)
                rows["score"][free]=np.maximum(0,rows["score"][free].astype(int)-12)
            if inds is not None:
                local=vox[inds]%SIDE
                slots,first=np.unique(local@[1024,32,1],return_index=True)
                union=np.union1d(rows["slot"],slots)
                merged=np.zeros(len(union),RECORD);merged["slot"]=union
                merged[np.searchsorted(union,rows["slot"])]=rows
                dest=np.searchsorted(union,slots)
                previous=merged["hits"][dest]
                color=frame["rgb"][inds[first]].astype(np.uint16)
                merged["first"][dest[previous==0]]=now
                # Established surfaces only need an occasional refresh unless
                # their color or free-space evidence has changed.
                refresh=(~merged["settled"][dest])|(now-merged["seen"][dest]>=60)|(merged["miss"][dest]>0)
                refresh |= np.abs(color.astype(int)-merged["rgb"][dest]).mean(axis=1)>24
                if vote:
                    origins=pose[:3,3]-((vox[inds[first]]+.5)*CELL)
                    directions=origins/np.maximum(1e-6,np.linalg.norm(origins,axis=1)[:,None])
                    update_confidence(merged,dest[refresh],directions[refresh],color[refresh],now)
                dest=dest[refresh];previous=previous[refresh];color=color[refresh]
                weight=np.minimum(previous,4).astype(np.uint16)[:,None]
                merged["rgb"][dest]=(merged["rgb"][dest]*weight+color)//(weight+1)
                merged["hits"][dest]=np.maximum(1,np.minimum(12,previous+int(vote)))
                merged["miss"][dest]=0;merged["seen"][dest]=now
                rows=merged
            remove=rows["miss"]>=np.where(rows["settled"],6,np.where(rows["hits"]>=CONFIRM,3,2))
            if prune:remove |= (rows["hits"]<CONFIRM)&(rows["seen"]<now-PENDING_TTL)
            rows=rows[~remove]
            if not np.array_equal(rows,old):updated[key]=rows
        self.write(updated)
        if vote:self.recent.append((pose.copy(),now))
        self.last_sample=(pose.copy(),now)
        if prune:self.last_prune=now

    def import_confirmed(self,old_rows):
        tiles={}
        for x,y,z,color,seen in old_rows:
            key=(x//SIDE,y//SIDE,z//SIDE)
            slot=(x%SIDE)*1024+(y%SIDE)*32+z%SIDE
            tiles.setdefault(key,[]).append((slot,(color>>16,(color>>8)&255,color&255),CONFIRM,0,seen,
                                             seen,(0,0,0),0,0,0,False))
        self.write({key:np.sort(np.array(rows,RECORD),order="slot") for key,rows in tiles.items()})

    def write(self,updates):
        for key,rows in updates.items():
            address=tuple(map(int,key))
            if len(rows):
                self.db.execute("INSERT OR REPLACE INTO surface_tiles VALUES(?,?,?,?)",(*address,FORMAT+zlib.compress(rows.tobytes(),1)))
                self.tiles[key]=rows
            else:
                self.db.execute("DELETE FROM surface_tiles WHERE x=? AND y=? AND z=?",address)
                self.tiles.pop(key,None)
