"""Local OpenVR targets for short, symmetric full-body exercises."""
import math


def smooth(value):
    t=max(0.,min(1.,value))
    return t*t*(3-2*t)


def blend(start,end,t):
    return [tuple(a+(b-a)*t for a,b in zip(p,q)) for p,q in zip(start,end)]


def symmetric(head,hand,hip,foot):
    # Targets are x, y, z, pitch, roll. Mirror hand/foot roll with the limb.
    def mirror(p):
        x,y,z,pitch,roll=p
        return -x,y,z,pitch,-roll
    return [head,mirror(hand),hand,hip,mirror(foot),foot]


def exercise(name,t):
    stand=symmetric((0,1.65,0,0,0),(.32,1.15,-.30,0,0),(0,.95,0,0,0),(.13,.06,0,0,0))
    squat=symmetric((0,1.34,-.14,-15,0),(.32,.92,.09,15,0),(0,.69,.09,-18,0),(.13,.06,0,0,0))
    if name=="jumping-jacks":
        spread=math.sin(2*math.pi*t)**2
        bounce=.11*abs(math.sin(4*math.pi*t))
        arc=math.radians(-75+150*spread)
        hand=(.19+.57*math.cos(arc),1.38+.57*math.sin(arc),-.06,0,145*spread)
        p=symmetric((0,1.65,0,0,0),hand,(0,.95,0,0,0),(.13+.25*spread,.06,0,0,0))
        p=[(x,y+bounce,z,a,r) for x,y,z,a,r in p]
        return blend(stand,p,smooth(min(t,1-t)/.09))
    if name=="push-ups":
        hands=(.24,.09,-.62,-72,0)
        plank=symmetric((0,.74,-.69,-66,0),hands,(0,.46,-.04,-68,0),(.13,.09,.76,-65,0))
        low=symmetric((0,.35,-.72,-82,0),hands,(0,.25,-.04,-80,0),(.13,.09,.76,-65,0))
        enter=min(t,1-t)
        if enter<.10:return blend(stand,squat,smooth(enter/.10))
        if enter<.24:return blend(squat,plank,smooth((enter-.10)/.14))
        return blend(plank,low,math.sin(2*math.pi*(t-.24)/.52)**2)
    if t<.20:return blend(stand,squat,smooth(t/.20))
    if t>.80:return blend(squat,stand,smooth((t-.80)/.20))
    air=(t-.20)/.60
    tuck=math.sin(math.pi*air)**2
    local=blend(squat,symmetric((0,1.39,-.15,-30,0),(.23,.86,-.37,-25,0),
                                (0,.86,0,-8,0),(.17,.58,-.24,-28,0)),tuck)
    angle=math.tau*smooth(air)
    lift=1.12*4*air*(1-air)
    result=[]
    for x,y,z,pitch,roll in local:
        dy=y-.85
        result.append((x,.85+lift+dy*math.cos(angle)-z*math.sin(angle),
                       dy*math.sin(angle)+z*math.cos(angle),pitch+math.degrees(angle),roll))
    return result
