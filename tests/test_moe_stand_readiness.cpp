#include "StandReadiness.h"
#include <cassert>
#include <iostream>
using namespace go2_first_test;
int main() {
    GuardInput s; s.state_available=true; s.quaternion_wxyz={1.,0.,0.,0.};
    s.q={0.,.8,-1.5,0.,.8,-1.5,0.,1.,-1.5,0.,1.,-1.5};
    const auto target=s.q;
    s.q[0]=.30; assert(settled_pose(s,target));
    s.q[0]=.300001; assert(!settled_pose(s,target));
    s.q[0]=-.30; assert(settled_pose(s,target));
    s.q[0]=-.300001; assert(!settled_pose(s,target));
    s.q=target; s.dq[0]=.800001; assert(!settled_pose(s,target));
    s.dq[0]=0.; s.gyro[0]=.350001; assert(!settled_pose(s,target));
    s.gyro[0]=0.; auto invalid=target; invalid[2]=-.7;
    s.q[2]=-.9; assert(!settled_pose(s,invalid)); s.q=target;
    StandReadiness gate;
    for(int i=0;i<=75;++i) {s.tick=i+1; assert(gate.update(s,target,i*.01,true)==(i==75));}
    s.tick++; s.q[0]=.31; assert(!gate.update(s,target,.76,true));
    s.q=target; s.tick++; assert(!gate.update(s,target,.77,true));
    s.tick++; assert(!gate.update(s,target,.9,true));
    std::cout<<"PASS: 0.30 rad boundary, independent limits, dwell and reset\n";
}
