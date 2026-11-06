#include "FSM/State_RLBase.h"
#include "StairsCommandBuffer.h"

State_RLBase::State_RLBase(int state,std::string name)
 : FSMState(state,name),runner_(go2_moe::kCandidateDirectory),
   commands_((std::filesystem::path(go2_moe::kCandidateDirectory).parent_path()/"run/perception.sock").string()),
   trace_(param::config["FSM"][name]["trace_file"].as<std::string>()) {
    auto cfg=param::config["FSM"][name];
    if(cfg["safety_joint_limit_profile"].as<std::string>("")!=go2_joint_limits::kProfileId ||
       !cfg["safety_enabled"].as<bool>(false)) throw std::runtime_error("MoE protection profile mismatch");
    lidar_service=[this](bool policy,bool stopped) {
        auto now=steady_seconds();commands_.poll(now,(stopped || parked_) ? 2 : (policy ? 1 : (handoff_epoch_ ? 3 : 0)));
        if(handoff_epoch_ && !policy) {
            const auto session=commands_.command().session;
            const bool changed=session && sensor_session_ && session!=sensor_session_;
            const bool qualified=qualification_.update(now,commands_.command(),commands_.healthy(now));
            if(changed || (qualification_reported_ && !qualified)) {
                stairs_command_buffer.begin_selection();
                spdlog::warn("POST_HANDOFF_SENSOR_RESET: hold continues; requalification and new manual arm required");
            }
            if(session) sensor_session_=session;
            if(qualified && !qualification_reported_) spdlog::info("POST_HANDOFF_SENSORS_LOCKED: 4 seconds continuous fresh post-handoff source data; geometry still required");
            qualification_reported_=qualified;
        }
        lidar_ready_for_start=commands_.ready(now) && commands_.command().mode==1 &&
            handoff_epoch_ && qualification_.qualified();
    };
    stairs_command_buffer.configure(.6f,.6f,.1f,.6f);
}
bool State_RLBase::perception_ready() {
    auto t=steady_seconds();commands_.poll(t,handoff_epoch_ ? 3 : 0);
    return commands_.ready(t) && commands_.command().mode==1 &&
        (!handoff_epoch_ || qualification_.qualified());
}
void State_RLBase::begin_post_handoff(std::uint64_t epoch) {
    handoff_epoch_=epoch;qualification_.begin(epoch);commands_.set_epoch(epoch);
    lidar_ready_for_start=false;qualification_reported_=false;sensor_session_=0;
    stairs_command_buffer.begin_selection();
}
void State_RLBase::exit() {commands_.poll(steady_seconds(),2);}
void State_RLBase::warmup_without_commands() {
    go2_moe::History temporary;
    const auto obs=temporary.append(go2_moe::frame(snapshot(),0.f,{}));
    for(int i=0;i<10;++i) (void)runner_.infer(obs);
}
void State_RLBase::enter() {
    if(!have_submitted_command || !stairs_command_buffer.armed())
        throw std::runtime_error("MoE requires a submitted stable hold and explicit speed arm");
    if(!perception_ready()) throw std::runtime_error("Lidar geometry is not fresh/ready");
    motion_session_=commands_.command().session;
    history_.clear();have_result_=false;parked_=false;lidar_parked=false;
    // Do not change the previous hold in enter(): the transition cycle is
    // checked/published normally before the first policy inference.
    next_=std::chrono::steady_clock::now();
}
void State_RLBase::run() {
    using clock=std::chrono::steady_clock;
    const auto now=clock::now();
    const double seconds=steady_seconds();
    commands_.poll(seconds,parked_ ? 2 : 1);
    if(parked_) {
        const double blend=std::clamp((seconds-parked_at_)/1.,0.,1.);
        hold_.kp.fill(20.+20.*blend);hold_.kd.fill(.5+.5*blend);
        write_command(hold_);return;
    }
    if(!commands_.ready(seconds)) {latch_stop("lidar_command_stale_invalid_or_fault");return;}
    if(commands_.command().session!=motion_session_) {latch_stop("sensor_restarted_during_motion");return;}
    const auto command=commands_.command();
    if(command.mode==2) {
        const auto measured=snapshot();
        double norm=0.;for(double v:measured.quaternion_wxyz) norm+=v*v;
        const double tilt=std::acos(std::clamp(1.-2.*(measured.quaternion_wxyz[1]*measured.quaternion_wxyz[1]+
            measured.quaternion_wxyz[2]*measured.quaternion_wxyz[2])/norm,-1.,1.));
        for(double speed:measured.dq) if(!std::isfinite(speed) || std::abs(speed)>.8) {
            latch_stop("lidar_park_joint_speed_not_settled");return;
        }
        if(!std::isfinite(tilt) || tilt>.25) {latch_stop("lidar_park_pose_not_settled");return;}
        hold_={};hold_.q=measured.q;hold_.kp.fill(20.);hold_.kd.fill(.5);
        parked_=true;parked_at_=seconds;lidar_parked=true;guard.finish_motion();
        write_command(hold_);spdlog::info("LIDAR_PARK_HOLD: stationary position hold; no policy reentry");return;
    }
    if(now>=next_) {
        const auto measured=snapshot();
        const std::array<float,3> velocity{command.vx,command.vy,command.wz};
        const auto obs=history_.append(go2_moe::frame(measured,velocity,
                       go2_moe::inverse_submitted(last_submitted_command)));
        const auto action=runner_.infer(obs);proposed_=go2_moe::decode(action);
        go2_lidar::PolicyTrace::Row row{};size_t index=0;
        row[index++]=seconds;row[index++]=command.sequence;
        for(float v:velocity) row[index++]=v;
        for(double v:measured.q) row[index++]=v;
        for(double v:measured.dq) row[index++]=v;
        for(float v:action) row[index++]=v;
        // Targets are logged in model FL/FR/RL/RR order, q/dq in SDK order.
        for(int joint:go2_moe::kMap) row[index++]=proposed_.q[joint];
        for(float v:obs) row[index++]=v;
        trace_.push(row);
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
