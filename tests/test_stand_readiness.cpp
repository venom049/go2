#include "StandReadiness.h"
#include <iostream>
#include <limits>
#include <cstdlib>
using namespace go2_first_test;
#define CHECK(x) do { if(!(x)) {std::cerr<<"FAIL line "<<__LINE__<<'\n';std::exit(1);} } while(false)
GuardInput state() {
    GuardInput s;s.state_available=true;s.quaternion_wxyz={1.,0.,0.,0.};
    s.q={0.,.8,-1.5,0.,.8,-1.5,0.,1.,-1.5,0.,1.,-1.5};
    s.tick=100;return s;
}
int main() {
    auto s=state(); auto target=s.q;
    CHECK(settled_pose(s,target));
    for(std::size_t i=0;i<12;++i) {
        s=state();s.q[i]+=.181;CHECK(!settled_pose(s,target));
        s=state();s.dq[i]=.801;CHECK(!settled_pose(s,target));
        s=state();s.q[i]=std::numeric_limits<double>::quiet_NaN();CHECK(!settled_pose(s,target));
    }
    s=state();s.q[2]=-2.78;CHECK(!settled_pose(s,target));
    s=state();s.quaternion_wxyz={std::cos(.15),std::sin(.15),0.,0.};CHECK(!settled_pose(s,target));
    s=state();s.gyro[0]=.36;CHECK(!settled_pose(s,target));
    s=state();StandReadiness gate;
    for(int i=0;i<76;++i) {s.tick++; CHECK(gate.update(s,target,i*.01,true)==(i>=75));}
    s.dq[0]=1.;CHECK(!gate.update(s,target,.76,true));s.dq[0]=0.;
    for(int i=77;i<152;++i) {s.tick++; CHECK(!gate.update(s,target,i*.01,true));}
    s.tick++; CHECK(gate.update(s,target,1.53,true));
    // Repeated stale samples cannot keep a standing latch.
    CHECK(!gate.update(s,target,1.65,true));
    gate.reset();s=state();
    for(int i=0;i<100;++i) CHECK(!gate.update(s,target,i*.01,true));
    gate.reset();s=state();
    for(int i=0;i<100;++i) {s.tick++;CHECK(!gate.update(s,target,i*.01,false));}
    gate.reset();s=state();CHECK(!gate.update(s,target,1.,true));
    s.tick--;CHECK(!gate.update(s,target,1.01,true));
    CHECK(!gate.update(s,target,std::numeric_limits<double>::infinity(),true));
    std::cout<<"PASS: stand readiness pose, folded, tilt, motion, dwell, stale, reset and tick tests; no hardware.\n";
}
