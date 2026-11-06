#include "HandoffHold.h"
#include <cstdlib>
#include <iostream>
#include <limits>
using namespace go2_first_test;
#define CHECK(x) do {if(!(x)){std::cerr<<"FAIL "<<__LINE__<<'\n';std::exit(1);}}while(false)
int main() {
    GuardInput s;s.state_available=true;s.quaternion_wxyz={1.,0.,0.,0.};s.tick=1;
    s.q={-.02,.666,-1.363,.02,.670,-1.367,-.081,.653,-1.357,.08,.66,-1.352};
    GuardOutput f;
    f.q={0.,.67,-1.3,0.,.67,-1.3,0.,.67,-1.3,0.,.67,-1.3};
    f.kp.fill(100.);f.kd.fill(5.);s.dq.fill(.07);
    GuardOutput h;
    CHECK(make_handoff_hold(s,f,h));
    for(std::size_t i=0;i<12;++i) {
        const double original=f.kp[i]*(f.q[i]-s.q[i])+f.kd[i]*(f.dq[i]-s.dq[i])+f.tau[i];
        const double hold=h.kp[i]*(h.q[i]-s.q[i])+h.kd[i]*(h.dq[i]-s.dq[i])+h.tau[i];
        CHECK(std::abs(original-hold)<1e-12);CHECK(h.kp[i]==40.&&h.kd[i]==1.);
    }
    Guard g;CHECK(!g.filter(s,h,0.,false).latched);s.tick++;
    CHECK(g.filter(s,h,.002,false).command_allowed);
    auto bad=f;bad.kp[0]=101.;CHECK(!make_handoff_hold(s,bad,h));
    bad=f;bad.q[0]+=1.;CHECK(!make_handoff_hold(s,bad,h));
    bad=f;bad.tau[0]=100.;CHECK(!make_handoff_hold(s,bad,h));
    bad=f;bad.kd[0]=std::numeric_limits<double>::quiet_NaN();CHECK(!make_handoff_hold(s,bad,h));
    auto folded=s;folded.q[2]=-2.79;CHECK(!make_handoff_hold(folded,f,h));
    f.kp.fill(40.);f.kd.fill(1.);CHECK(make_handoff_hold(s,f,h));
    for(double tau:h.tau) CHECK(std::abs(tau)<1e-12);
    std::cout<<"PASS: handoff hold PD estimate continuity, unchanged guard acceptance, and invalid/folded rejection.\n";
}
