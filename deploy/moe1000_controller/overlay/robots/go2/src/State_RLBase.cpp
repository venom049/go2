#include "FSM/State_RLBase.h"
#include "StairsCommandBuffer.h"

State_RLBase::State_RLBase(int state,std::string name)
 : FSMState(state,name),runner_(go2_moe::kCandidateDirectory) {
    auto cfg=param::config["FSM"][name];
    if(cfg["safety_joint_limit_profile"].as<std::string>("")!=go2_joint_limits::kProfileId ||
       !cfg["safety_enabled"].as<bool>(false)) throw std::runtime_error("MoE protection profile mismatch");
    stairs_command_buffer.configure(.6f,.6f,.1f,.6f);
}
void State_RLBase::warmup_without_commands() {
    go2_moe::History temporary;
    const auto obs=temporary.append(go2_moe::frame(snapshot(),0.f,{}));
    for(int i=0;i<10;++i) (void)runner_.infer(obs);
}
void State_RLBase::enter() {
    if(!have_submitted_command || !stairs_command_buffer.armed())
        throw std::runtime_error("MoE requires a submitted stable hold and explicit speed arm");
    history_.clear();have_result_=false;
    // Do not change the previous hold in enter(): the transition cycle is
    // checked/published normally before the first policy inference.
    next_=std::chrono::steady_clock::now();
}
void State_RLBase::run() {
    using clock=std::chrono::steady_clock;
    const auto now=clock::now();
    if(now>=next_) {
        const auto obs=history_.append(go2_moe::frame(snapshot(),.6f,
                       go2_moe::inverse_submitted(last_submitted_command)));
        proposed_=go2_moe::decode(runner_.infer(obs));
        last_result_=clock::now();have_result_=true;
        if(std::chrono::duration<double>(last_result_-now).count()>.020) {
            latch_stop("moe_policy_exceeded_20ms");return;
        }
        next_=now+std::chrono::milliseconds(20);
    }
    if(!have_result_ || std::chrono::duration<double>(clock::now()-last_result_).count()>.040) {
        latch_stop("moe_policy_result_stale");return;
    }
    // The shared post_run checks fresh measured state against this complete
    // proposal, and replaces all outputs with latched damping on any violation.
    write_command(proposed_);
}
