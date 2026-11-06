#pragma once
#include "FSMState.h"
#include <algorithm>

class State_FixStand : public FSMState {
public:
    State_FixStand(int state, std::string name="FixStand") : FSMState(state,std::move(name)) {
        const auto qs=param::config["FSM"]["FixStand"]["qs"].as<std::vector<std::vector<float>>>();
        if(qs.empty() || qs.back().size()!=12) throw std::runtime_error("Invalid hold target");
        std::copy(qs.back().begin(),qs.back().end(),target_.begin());
    }
    void enter() override {
        for(std::size_t i=0;i<12;++i) {
            const auto& m=lowcmd->msg_.motor_cmd()[i];
            initial_.q[i]=m.q(); initial_.dq[i]=m.dq(); initial_.kp[i]=m.kp();
            initial_.kd[i]=m.kd(); initial_.tau[i]=m.tau();
        }
        started_=steady_seconds();
    }
    void run() override {
        const double t=std::clamp((steady_seconds()-started_)/2.,0.,1.);
        const double a=t*t*(3.-2.*t);
        for(std::size_t i=0;i<12;++i) {
            auto& m=lowcmd->msg_.motor_cmd()[i];
            m.q()=initial_.q[i]+a*(target_[i]-initial_.q[i]);
            m.dq()=(1.-a)*initial_.dq[i];
            m.kp()=initial_.kp[i]+a*(40.-initial_.kp[i]);
            m.kd()=initial_.kd[i]+a*(1.-initial_.kd[i]);
            m.tau()=(1.-a)*initial_.tau[i];
        }
    }
protected:
    go2_first_test::JointArray target_{};
private:
    go2_first_test::GuardOutput initial_{};
    double started_=0.;
};
REGISTER_FSM(State_FixStand)
