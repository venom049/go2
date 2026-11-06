#include "MoePolicy.h"
#include "HandoffHold.h"
#include "StandReadiness.h"
#include <cassert>
#include <iostream>
#include <limits>

using namespace go2_moe;
go2_first_test::GuardInput standing() {
    go2_first_test::GuardInput s;s.state_available=true;s.quaternion_wxyz={1,0,0,0};
    s.q=stand_sdk();s.tick=1;return s;
}
int main() {
    auto s=standing();Action a{};auto f=frame(s,.6f,a);
    assert(f[5]==-1.f && f[6]==1.2f);
    for(size_t i=9;i<45;++i) assert(f[i]==0.f);
    s.dq[0]=2.;s.dq[3]=4.;f=frame(s,0.f,a);
    assert(f[21]==.2f && f[24]==.1f); // FL/FR permutation.
    History h;auto obs=h.append(f);
    for(size_t i=0;i<5;++i) assert(obs[105+i*12]==.2f);
    s.dq[3]=8.;obs=h.append(frame(s,0.f,a));
    assert(obs[105]==.2f && obs[105+48]==.4f);
    h.clear();obs=h.append(frame(s,0.f,a));assert(obs[105]==.4f);
    for(size_t i=0;i<12;++i) a[i]=static_cast<float>(i)*.1f;
    const auto cmd=decode(a);const auto inverse=inverse_submitted(cmd);
    for(size_t i=0;i<12;++i) {
        assert(std::abs(inverse[i]-a[i])<1e-6);
        assert(cmd.kp[kMap[i]]==20. && cmd.kd[kMap[i]]==.5);
        assert(cmd.dq[kMap[i]]==0. && cmd.tau[kMap[i]]==0.);
    }
    a[0]=std::numeric_limits<float>::infinity();bool failed=false;
    try {decode(a);} catch(...) {failed=true;} assert(failed);
    s=standing();auto zero=decode(Action{});go2_first_test::Guard guard;
    assert(!guard.filter(s,zero,0.,false).command_allowed);
    ++s.tick;assert(guard.filter(s,zero,.002,false).command_allowed);
    s.dq[2]=30.;++s.tick;assert(guard.filter(s,zero,.004,false).command_allowed);
    s.dq[2]=std::nextafter(30.f,std::numeric_limits<float>::infinity());++s.tick;
    assert(guard.filter(s,zero,.006,false).latched);
    s=standing();assert(guard.filter(s,zero,.008,false).latched); // no rearm
    go2_first_test::Guard tracking;auto far=zero;far.q[1]+=1.;
    assert(tracking.filter(s,far,0.,false).reason=="tracking_error_above_0.9rad");
    go2_first_test::Guard tilt;s.quaternion_wxyz={std::cos(.5),std::sin(.5),0.,0.};
    assert(tilt.filter(s,zero,0.,false).reason=="body_tilt_above_45_degrees");
    s=standing();go2_first_test::Guard timeout;
    timeout.filter(s,zero,0.,true);
    for(int i=1;i<1000;++i) {++s.tick;assert(timeout.filter(s,zero,i*.002,true).command_allowed);}
    ++s.tick;assert(timeout.filter(s,zero,2.,true).reason=="two_second_test_complete");
    go2_first_test::Guard stale;stale.filter(s,zero,0.,false);
    assert(stale.filter(s,zero,.101,false).latched);
    auto factory=zero;factory.kp.fill(100.);factory.kd.fill(5.);factory.q[1]+=.01;
    go2_first_test::GuardOutput hold;
    assert(go2_first_test::make_handoff_hold(s,factory,hold));
    assert(hold.kp[1]==40. && hold.kd[1]==1.);
    const auto torque=[](const auto& c,const auto& x,size_t i) {
        return c.kp[i]*(c.q[i]-x.q[i])+c.kd[i]*(c.dq[i]-x.dq[i])+c.tau[i];
    };
    for(size_t i=0;i<12;++i) assert(std::abs(torque(factory,s,i)-torque(hold,s,i))<1e-12);
    factory.kp[0]=999.;assert(!go2_first_test::make_handoff_hold(s,factory,hold));
    std::cout<<"MoE mapping/history/action, guard boundaries/latch/deadline, and handoff torque tests passed\n";
}
