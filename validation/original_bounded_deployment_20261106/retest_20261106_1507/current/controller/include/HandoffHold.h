#pragma once
#include "FirstTestGuard.h"
namespace go2_first_test {
// Convert an observed factory stand command to the model's 40/1 gains while
// matching its instantaneous PD+feedforward estimate at this measured state.
// This is not measured actuator torque or a guarantee about actual forces.
// No command or measurement is clipped and all original Guard limits remain.
inline bool make_handoff_hold(const GuardInput& s, const GuardOutput& factory,
                              GuardOutput& hold) {
    GuardOutput candidate;
    for(std::size_t i=0;i<12;++i) {
        for(double v:{s.q[i],s.dq[i],factory.q[i],factory.dq[i],factory.kp[i],factory.kd[i],factory.tau[i]})
            if(!std::isfinite(v)) return false;
        const bool observed_stand=(factory.kp[i]==100. && factory.kd[i]==5.) ||
                                  (factory.kp[i]==40. && factory.kd[i]==1.);
        if(!observed_stand || s.q[i]<kTargetLower[i] || s.q[i]>kTargetUpper[i] ||
           factory.q[i]<kTargetLower[i] || factory.q[i]>kTargetUpper[i] ||
           std::abs(factory.dq[i])>.01 || std::abs(s.dq[i])>.8 ||
           std::abs(factory.q[i]-s.q[i])>.18 || std::abs(factory.tau[i])>kEffortUpper[i]) return false;
        const double torque=factory.kp[i]*(factory.q[i]-s.q[i])+
                            factory.kd[i]*(factory.dq[i]-s.dq[i])+factory.tau[i];
        if(!std::isfinite(torque) || std::abs(torque)>kEffortUpper[i]) return false;
        candidate.q[i]=factory.q[i]; candidate.dq[i]=0.;
        candidate.kp[i]=40.; candidate.kd[i]=1.;
        candidate.tau[i]=torque-40.*(factory.q[i]-s.q[i])+s.dq[i];
        if(!std::isfinite(candidate.tau[i]) || std::abs(candidate.tau[i])>kEffortUpper[i]) return false;
    }
    hold=candidate;return true;
}
}
