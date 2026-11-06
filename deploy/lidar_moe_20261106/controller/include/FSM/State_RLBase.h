#pragma once
#include "FSMState.h"
#include "MoeRunner.h"
#include "PerceptionCommand.h"
#include "PolicyTrace.h"
#include <chrono>

// MoE-specific state. Uses a single mutex-protected SDK-order snapshot and the
// shared final publication guard; no legacy HIM observation/action managers.
class State_RLBase : public FSMState {
public:
    State_RLBase(int state,std::string name);
    void enter() override;
    void run() override;
    void exit() override;
    void warmup_without_commands();
    bool perception_ready();
    void begin_post_handoff(std::uint64_t epoch);
    void flush_trace() {trace_.flush(); if(trace_.failed() || trace_.dropped()) request_stop();}
private:
    go2_moe::Runner runner_;
    go2_lidar::CommandSocket commands_;
    go2_lidar::PolicyTrace trace_;
    go2_moe::History history_;
    go2_first_test::GuardOutput proposed_;
    std::chrono::steady_clock::time_point next_,last_result_;
    bool have_result_=false;
    bool parked_=false;
    double parked_at_=0.;
    go2_first_test::GuardOutput hold_;
    go2_lidar::Qualification qualification_;
    std::uint64_t handoff_epoch_=0, sensor_session_=0, motion_session_=0;
    bool qualification_reported_=false;
};
REGISTER_FSM(State_RLBase)
