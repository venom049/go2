"""Bounded receive-only diagnostic. No publishers or motion RPCs."""
import json, time, threading, math, struct
from collections import Counter
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_, LowCmd_

lock=threading.Lock()
states={}
counts=Counter()
gaps={"state":[],"command":[]}
def callback(kind):
    def receive(msg):
        now=time.monotonic()
        if kind=='state':
            row={"q":[m.q for m in msg.motor_state[:12]],"dq":[m.dq for m in msg.motor_state[:12]],"tick":msg.tick}
        else:
            row={key:[getattr(m,key) for m in msg.motor_cmd[:12]] for key in ('q','dq','kp','kd','tau')}
        with lock:
            if kind in states:gaps[kind].append(now-states[kind][0])
            states[kind]=(now,row);counts[kind]+=1
    return receive

def f32(x):return struct.unpack('f',struct.pack('f',x))[0]
lo=[f32(x) for x in [-.8377580409572781,-1.5708,-2.7227]*2+[-.8377580409572781,-.5236,-2.7227]*2]
hi=[f32(x) for x in [.8377580409572781,3.4907,-.83776]*2+[.8377580409572781,4.5379,-.83776]*2]
eff=[23.7,23.7,45.43]*4
ChannelFactoryInitialize(0,'eth0')
subs=[]
for name,topic,typ in [('state','rt/lowstate',LowState_),('command','rt/lowcmd',LowCmd_)]:
    s=ChannelSubscriber(topic,typ);s.Init(callback(name),1);subs.append(s)
reasons=Counter();worst={};rows=0;last=None;deadline=time.monotonic()+6
while time.monotonic()<deadline:
    time.sleep(.002)
    with lock:s=states.get('state');c=states.get('command')
    if s is None or c is None:continue
    rows+=1;now=time.monotonic();issues=[]
    if now-c[0]>.02:issues.append('command_older_than_20ms')
    if now-s[0]>.1:issues.append('state_older_than_100ms')
    state,cmd=s[1],c[1]
    for i in range(12):
        q,dq=state['q'][i],state['dq'][i]
        fq,fdq,kp,kd,tau=[cmd[k][i] for k in ('q','dq','kp','kd','tau')]
        if not all(math.isfinite(x) for x in [q,dq,fq,fdq,kp,kd,tau]):issues.append(f'{i}:nonfinite');continue
        if (kp,kd) not in [(100.,5.),(40.,1.)]:issues.append(f'{i}:factory_gains')
        if not lo[i]<=q<=hi[i]:issues.append(f'{i}:actual_angle')
        if not lo[i]<=fq<=hi[i]:issues.append(f'{i}:factory_target_angle')
        if abs(fdq)>.01:issues.append(f'{i}:factory_target_velocity')
        if abs(dq)>.8:issues.append(f'{i}:actual_speed')
        if abs(fq-q)>.18:issues.append(f'{i}:tracking')
        torque=kp*(fq-q)+kd*(fdq-dq)+tau
        ff=torque-40*(fq-q)+dq
        if abs(tau)>eff[i] or abs(torque)>eff[i] or abs(ff)>eff[i]:issues.append(f'{i}:torque')
    reasons.update(issues)
    if issues and not worst:worst={'issues':issues,'state':state,'factory':cmd,'command_age_s':now-c[0]}
    last={'state':state,'factory':cmd,'issues':issues,'command_age_s':now-c[0]}
for s in subs:s.Close()
print(json.dumps({'read_only':True,'rows':rows,'received':dict(counts),'reasons':dict(reasons),'max_receive_gap_s':{k:max(v) if v else None for k,v in gaps.items()},'first_issue':worst,'last':last},indent=2),flush=True)
