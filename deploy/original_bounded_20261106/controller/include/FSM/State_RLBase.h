#pragma once
#include "FSMState.h"
#include "MoeRunner.h"
#include <chrono>

// MoE-specific state. Uses a single mutex-protected SDK-order snapshot and the
// shared final publication guard; no legacy HIM observation/action managers.
class State_RLBase : public FSMState {
public:
    State_RLBase(int state,std::string name);
    void enter() override;
    void run() override;
    void exit() override {}
    void warmup_without_commands();
private:
    go2_moe::Runner runner_;
    go2_moe::History history_;
    go2_first_test::GuardOutput proposed_;
    std::chrono::steady_clock::time_point next_,last_result_;
    bool have_result_=false;
};
REGISTER_FSM(State_RLBase)
