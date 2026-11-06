#pragma once
#include "FirstTestGuard.h"
#include <algorithm>

namespace go2_first_test {
// Conservative software gate, not proof of foot contact or mechanical support.
inline bool settled_pose(const GuardInput& s, const JointArray& target) {
    if(!s.state_available) return false;
    double norm = 0.;
    for(double v : s.quaternion_wxyz) {
        if(!std::isfinite(v)) return false;
        norm += v*v;
    }
    if(std::abs(norm-1.) > .05) return false;
    const double x = s.quaternion_wxyz[1], y = s.quaternion_wxyz[2];
    const double tilt = std::acos(std::clamp(1.-2.*(x*x+y*y)/norm, -1., 1.));
    if(tilt > .25) return false;
    for(double g : s.gyro) if(!std::isfinite(g) || std::abs(g) > .35) return false;
    for(std::size_t i=0;i<12;++i) {
        if(!std::isfinite(s.q[i]) || !std::isfinite(s.dq[i]) || !std::isfinite(target[i]) ||
           s.q[i] < kTargetLower[i] || s.q[i] > kTargetUpper[i] ||
           target[i] < kTargetLower[i] || target[i] > kTargetUpper[i] ||
           std::abs(s.q[i]-target[i]) > .18 || std::abs(s.dq[i]) > .8) return false;
    }
    return true;
}

class StandReadiness {
public:
    bool update(const GuardInput& s, const JointArray& target, double now, bool finished) {
        if(!std::isfinite(now) || now < 0 || (last_now_ && now < *last_now_)) {
            reset(); return false;
        }
        if(last_now_ && now - *last_now_ > .1) stable_since_.reset();
        last_now_=now;
        if(!finished || !settled_pose(s,target)) { stable_since_.reset(); return false; }
        if(last_tick_ && s.tick == *last_tick_) {
            if(now-last_advance_ > .1) stable_since_.reset();
            return ready(now);
        }
        if(last_tick_ && std::uint32_t(s.tick-*last_tick_) >= UINT32_C(0x80000000)) {
            reset(); return false;
        }
        last_tick_=s.tick; last_advance_=now;
        if(!stable_since_) stable_since_=now;
        return ready(now);
    }
    void reset() { stable_since_.reset(); last_now_.reset(); last_tick_.reset(); }
private:
    bool ready(double now) const { return stable_since_ && now-*stable_since_ >= .75; }
    std::optional<double> stable_since_, last_now_;
    std::optional<std::uint32_t> last_tick_;
    double last_advance_=0.;
};
}
