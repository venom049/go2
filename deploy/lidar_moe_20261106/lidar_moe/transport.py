"""Versioned local protocol; CLOCK_MONOTONIC belongs to the same host.

A command echoes the controller's handoff epoch and identifies the sensor
process. Source-health is separate from geometry readiness. Query packets
never change the accepted command, its source age, or the feedback peer.
"""
import math,struct
PACKET=struct.Struct('<4sIQQffffIIQQ')
FEEDBACK=struct.Struct('<4sIQIQ')
QUERY=struct.Struct('<4sI')
MODES={'WAIT':0,'RUN':1,'HOLD':2,'FAULT':3}
def encode(sequence,source_mono,command,mode,*,healthy=None,epoch=0,session=1):
    if len(command)!=3 or not all(math.isfinite(v) for v in command):raise ValueError('nonfinite command')
    if not 0<=command[0]<=.35 or abs(command[1])>.10 or abs(command[2])>.35:raise ValueError('command outside envelope')
    if mode!='RUN' and any(command):raise ValueError('non-running command must be zero')
    if healthy is None:healthy=mode in ('RUN','HOLD')
    if not math.isfinite(source_mono) or source_mono<0 or sequence<=0 or session<=0:raise ValueError('invalid source identity')
    return PACKET.pack(b'G2L2',2,sequence,int(source_mono*1e9),*command,1. if mode in ('RUN','HOLD') else 0.,MODES[mode],int(bool(healthy)),epoch,session)
def feedback(data,now):
    if len(data)!=FEEDBACK.size:raise ValueError('malformed feedback')
    magic,version,stamp,active,epoch=FEEDBACK.unpack(data)
    if magic!=b'G2F2' or version!=2 or active not in (0,1,2,3) or not 0<=now-stamp*1e-9<.4:raise ValueError('invalid feedback')
    return active,epoch
