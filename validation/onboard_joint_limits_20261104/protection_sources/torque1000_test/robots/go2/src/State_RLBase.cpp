#include "FSM/State_RLBase.h"
#include "unitree_articulation.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "StairsCommandBuffer.h"
#include <algorithm>
#include <array>
#include <cctype>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>
#include <unordered_map>

namespace isaaclab
{

class AdaptiveStiffnessJointPositionAction : public ActionTerm
{
public:
    AdaptiveStiffnessJointPositionAction(YAML::Node cfg, ManagerBasedRLEnv* env)
    : ActionTerm(cfg, env)
    {
        if(env->robot->data.joint_ids_map.size() != joint_action_dim)
        {
            throw std::runtime_error("Go2 stair policy requires exactly 12 joints.");
        }

        position_scale = cfg["position_scale"].as<float>();
        action_clip = cfg["action_clip"].as<float>();
        kp_default = cfg["kp_default"].as<float>();
        kp_action_scale = cfg["kp_action_scale"].as<float>();
        kp_min = cfg["kp_range"][0].as<float>();
        kp_max = cfg["kp_range"][1].as<float>();
        kd_sqrt_scale = cfg["kd_sqrt_scale"].as<float>();
        offsets = cfg["offset"].as<std::vector<float>>();
        if(offsets.size() != joint_action_dim)
        {
            throw std::runtime_error("Go2 stair policy requires 12 default joint offsets.");
        }

        raw.resize(policy_action_dim, 0.0f);
        processed.resize(joint_action_dim * 3, 0.0f);
        reset();
    }

    int action_dim() override { return policy_action_dim; }

    std::vector<float> raw_actions() override { return raw; }

    std::vector<float> processed_actions() override { return processed; }

    void process_actions(std::vector<float> actions) override
    {
        if(actions.size() != policy_action_dim)
        {
            throw std::runtime_error("Go2 stair policy must output 16 actions.");
        }

        // Validate the entire original vector BEFORE clipping or mutating raw.
        // std::clamp(+/-Inf) is finite, which would otherwise hide a bad model
        // output from both the legacy safety check and publication guard.
        for(float value : actions) {
            if(!std::isfinite(value))
                throw std::runtime_error("Nonfinite raw policy output before clipping");
        }
        for(std::size_t i = 0; i < policy_action_dim; ++i)
        {
            raw[i] = std::clamp(actions[i], -action_clip, action_clip);
        }

        for(std::size_t i = 0; i < joint_action_dim; ++i)
        {
            processed[i] = offsets[i] + position_scale * raw[i];
        }

        for(std::size_t leg = 0; leg < stiffness_action_dim; ++leg)
        {
            const float kp = std::clamp(
                kp_default + kp_action_scale * raw[joint_action_dim + leg],
                kp_min,
                kp_max
            );
            const float kd = kd_sqrt_scale * std::sqrt(kp);
            for(std::size_t joint = 0; joint < 3; ++joint)
            {
                const std::size_t index = leg * 3 + joint;
                processed[joint_action_dim + index] = kp;
                processed[2 * joint_action_dim + index] = kd;
            }
        }
    }

    void reset() override
    {
        std::fill(raw.begin(), raw.end(), 0.0f);
        process_actions(raw);
    }

private:
    static constexpr std::size_t joint_action_dim = 12;
    static constexpr std::size_t stiffness_action_dim = 4;
    static constexpr std::size_t policy_action_dim =
        joint_action_dim + stiffness_action_dim;

    float position_scale = 0.25f;
    float action_clip = 100.0f;
    float kp_default = 40.0f;
    float kp_action_scale = 20.0f;
    float kp_min = 10.0f;
    float kp_max = 70.0f;
    float kd_sqrt_scale = 0.2f;
    std::vector<float> offsets;
    std::vector<float> raw;
    std::vector<float> processed;
};

REGISTER_ACTION(AdaptiveStiffnessJointPositionAction);

REGISTER_OBSERVATION(applied_action)
{
    // Never feed proposed/unpublished policy outputs back as applied targets.
    // For the isolated 1100 adapter, these 16 SDK actions inverse-encode the
    // last guard-approved target handed to the publisher. The final four gain
    // actions are ignored by the HIM wrapper. This is not an actuator ack.
    if(!FSMState::have_submitted_command) {
        // Constructor/read-only robot-print only: no motor command exists.
        // Do not invent an applied history for inference-only diagnostics.
        return std::vector<float>(16, 0.0f);
    }
    const auto cfg = env->cfg["actions"]["AdaptiveStiffnessJointPositionAction"];
    const auto offsets = cfg["offset"].as<std::vector<float>>();
    const auto scale = cfg["position_scale"].as<float>();
    if(offsets.size() != 12 || !std::isfinite(scale) || scale <= 0.0f)
        throw std::runtime_error("Invalid first-test inverse target encoding");
    std::vector<float> raw(16, 0.0f);
    for(std::size_t i = 0; i < 12; ++i) {
        const auto sdk_id = static_cast<std::size_t>(env->robot->data.joint_ids_map[i]);
        if(sdk_id >= 12) throw std::runtime_error("Invalid first-test joint mapping");
        raw[i] = (static_cast<float>(FSMState::last_submitted_command.q[sdk_id]) - offsets[i]) / scale;
        if(!std::isfinite(raw[i])) throw std::runtime_error("Nonfinite inverse target");
    }
    return raw;
}

REGISTER_OBSERVATION(keyboard_stairs_velocity_commands)
{
    static std::unordered_map<ManagerBasedRLEnv*, float> speeds;
    float& speed = speeds[env];

    const float min_speed = params["min_forward_speed"].as<float>();
    const float max_speed = params["max_forward_speed"].as<float>();
    const float speed_step = params["speed_step"].as<float>();
    const float initial_speed =
        params["initial_forward_speed"].as<float>(min_speed);
    stairs_command_buffer.configure(
        min_speed, max_speed, speed_step, initial_speed
    );

    if(env->episode_length == 0)
    {
        speed = stairs_command_buffer.selected_speed();
        stairs_command_buffer.reset_auto_zero_report();
    }

    bool changed = false;
    if(stairs_command_buffer.auto_zero_due(env->episode_length, env->step_dt) &&
       speed != 0.0f)
    {
        speed = 0.0f;
        changed = true;
        if(stairs_command_buffer.consume_auto_zero_report())
        {
            spdlog::warn(
                "Stairs auto-zero reached: vx command set to 0.00 m/s."
            );
        }
    }

    std::string key;
    while(FSMState::keyboard && FSMState::keyboard->pop_key(key))
    {
        if(key.size() == 1)
        {
            key[0] = static_cast<char>(
                std::tolower(static_cast<unsigned char>(key[0]))
            );
        }

        if(key == "x" || key == "0" || key == " ")
        {
            FSMState::latch_stop("operator_stop_key");
            speed = 0.0f;
            changed = true;
        }
    }

    if(changed)
    {
        speed = std::round(speed * 100.0f) / 100.0f;
        spdlog::info("Stairs keyboard command: vx={:.2f} m/s", speed);
    }

    return {speed, 0.0f, 0.0f};
}

}

namespace
{
static constexpr std::array<const char*, 12> kGo2JointNames = {
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
};

struct SafetySnapshot
{
    bool triggered = false;
    std::string reason;
    std::string joint;
    float metric = 0.0f;
    float max_q_error = 0.0f;
    std::string max_q_error_joint;
    float max_abs_dq = 0.0f;
    std::string max_abs_dq_joint;
    float max_abs_raw_action = 0.0f;
    int max_raw_action_index = -1;
    float max_kp = 0.0f;
    float orientation_angle = 0.0f;
    int kp_saturated_frames = 0;
};

std::string joint_name(std::size_t index)
{
    if(index < kGo2JointNames.size()) {
        return kGo2JointNames[index];
    }
    return "joint_" + std::to_string(index);
}

SafetySnapshot evaluate_safety(
    const isaaclab::ArticulationData& data,
    const std::vector<float>& raw,
    const std::vector<float>& processed,
    float q_error_limit,
    float dq_limit,
    float raw_action_limit,
    float kp_saturation_limit,
    int kp_saturation_frames_limit,
    int kp_saturated_frames,
    float bad_orientation_limit,
    bool safety_enabled
)
{
    SafetySnapshot snapshot;
    snapshot.kp_saturated_frames = kp_saturated_frames;

    const std::size_t joint_count = static_cast<std::size_t>(data.joint_pos.size());
    if(processed.size() >= joint_count)
    {
        for(std::size_t i = 0; i < joint_count; ++i)
        {
            const float q_error = std::abs(data.joint_pos[i] - processed[i]);
            if(q_error > snapshot.max_q_error)
            {
                snapshot.max_q_error = q_error;
                snapshot.max_q_error_joint = joint_name(i);
            }
        }
    }

    for(std::size_t i = 0; i < joint_count; ++i)
    {
        const float abs_dq = std::abs(data.joint_vel[i]);
        if(abs_dq > snapshot.max_abs_dq)
        {
            snapshot.max_abs_dq = abs_dq;
            snapshot.max_abs_dq_joint = joint_name(i);
        }
    }

    for(std::size_t i = 0; i < raw.size(); ++i)
    {
        const float abs_raw = std::abs(raw[i]);
        if(abs_raw > snapshot.max_abs_raw_action)
        {
            snapshot.max_abs_raw_action = abs_raw;
            snapshot.max_raw_action_index = static_cast<int>(i);
        }
    }

    if(processed.size() >= joint_count * 2)
    {
        for(std::size_t i = joint_count; i < joint_count * 2; ++i) {
            snapshot.max_kp = std::max(snapshot.max_kp, processed[i]);
        }
    }

    const float gravity_z = data.projected_gravity_b[2];
    if(std::isfinite(gravity_z))
    {
        const float cos_angle = std::clamp(-gravity_z, -1.0f, 1.0f);
        snapshot.orientation_angle = std::acos(cos_angle);
    }
    else
    {
        snapshot.orientation_angle = std::numeric_limits<float>::infinity();
    }

    if(!safety_enabled) {
        return snapshot;
    }

    if(snapshot.max_q_error > q_error_limit)
    {
        snapshot.triggered = true;
        snapshot.reason = "max_q_error";
        snapshot.joint = snapshot.max_q_error_joint;
        snapshot.metric = snapshot.max_q_error;
        return snapshot;
    }
    if(snapshot.max_abs_dq > dq_limit)
    {
        snapshot.triggered = true;
        snapshot.reason = "max_abs_dq";
        snapshot.joint = snapshot.max_abs_dq_joint;
        snapshot.metric = snapshot.max_abs_dq;
        return snapshot;
    }
    if(snapshot.max_abs_raw_action > raw_action_limit)
    {
        snapshot.triggered = true;
        snapshot.reason = "max_abs_raw_action";
        snapshot.joint = "action_" + std::to_string(snapshot.max_raw_action_index);
        snapshot.metric = snapshot.max_abs_raw_action;
        return snapshot;
    }
    if(
        snapshot.max_kp >= kp_saturation_limit &&
        kp_saturated_frames >= kp_saturation_frames_limit
    )
    {
        snapshot.triggered = true;
        snapshot.reason = "kp_saturated";
        snapshot.joint = "all";
        snapshot.metric = snapshot.max_kp;
        return snapshot;
    }
    if(
        bad_orientation_limit > 0.0f &&
        snapshot.orientation_angle > bad_orientation_limit
    )
    {
        snapshot.triggered = true;
        snapshot.reason = "bad_orientation";
        snapshot.joint = "base";
        snapshot.metric = snapshot.orientation_angle;
        return snapshot;
    }

    return snapshot;
}

void override_stairs_speed(YAML::Node cfg, float speed)
{
    auto obs = cfg["observations"]["keyboard_stairs_velocity_commands"];
    if(!obs)
    {
        throw std::runtime_error(
            "Policy deploy.yaml has no keyboard_stairs_velocity_commands observation."
        );
    }
    obs["params"]["min_forward_speed"] = speed;
    obs["params"]["max_forward_speed"] = speed;
    obs["params"]["initial_forward_speed"] = speed;
}
}

State_RLBase::State_RLBase(int state_mode, std::string state_string)
: FSMState(state_mode, state_string) 
{
    auto cfg = param::config["FSM"][state_string];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
    if(cfg["speed_override"])
    {
        const float speed = cfg["speed_override"].as<float>();
        override_stairs_speed(deploy_cfg, speed);
        spdlog::info("Stairs speed override enabled: vx={:.2f} m/s", speed);
    }

    const double auto_zero_seconds = cfg["auto_zero_seconds"].as<double>(0.0);
    stairs_command_buffer.set_auto_zero_seconds(auto_zero_seconds);
    if(auto_zero_seconds > 0.0)
    {
        spdlog::warn(
            "Stairs auto-zero enabled: vx command becomes 0.00 m/s after {:.2f} seconds.",
            auto_zero_seconds
        );
    }

    auto_lie_seconds_ = cfg["auto_lie_seconds"].as<double>(0.0);
    if(auto_lie_seconds_ > 0.0)
    {
        spdlog::warn(
            "Stairs auto LieDown enabled: {:.2f} seconds.",
            auto_lie_seconds_
        );
        this->registered_checks.emplace_back(
            std::make_pair(
                [this]()->bool{
                    const double elapsed = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - enter_time_
                    ).count();
                    if(elapsed < auto_lie_seconds_) {
                        return false;
                    }
                    if(!auto_lie_reported_)
                    {
                        spdlog::warn(
                            "Stairs auto LieDown timeout reached after {:.2f} seconds.",
                            elapsed
                        );
                        auto_lie_reported_ = true;
                    }
                    return true;
                },
                FSMStringMap.right.at("LieDown")
            )
        );
    }

    safety_enabled_ = cfg["safety_enabled"].as<bool>(true);
    safety_q_error_limit_ = cfg["safety_q_error_limit"].as<float>(0.90f);
    safety_dq_limit_ = cfg["safety_dq_limit"].as<float>(13.0f);
    safety_raw_action_limit_ =
        cfg["safety_raw_action_limit"].as<float>(5.0f);
    safety_kp_saturation_limit_ =
        cfg["safety_kp_saturation_limit"].as<float>(69.5f);
    safety_bad_orientation_limit_ =
        cfg["bad_orientation_limit"].as<float>(1.0f);
    safety_kp_saturation_frames_limit_ =
        cfg["safety_kp_saturation_frames"].as<int>(3);
    if(safety_kp_saturation_frames_limit_ < 1) {
        safety_kp_saturation_frames_limit_ = 1;
    }
    if(safety_enabled_)
    {
        spdlog::warn(
            "Stairs safety stop enabled: q_err>{:.2f} rad, |dq|>{:.2f} rad/s, "
            "|raw|>{:.2f}, kp>={:.1f} for {} policy frames.",
            safety_q_error_limit_,
            safety_dq_limit_,
            safety_raw_action_limit_,
            safety_kp_saturation_limit_,
            safety_kp_saturation_frames_limit_
        );
    }
    else
    {
        spdlog::warn("Stairs safety stop is disabled.");
    }
    this->registered_checks.emplace_back(
        std::make_pair(
            [this]()->bool{ return safety_stop_requested_.load(); },
            FSMStringMap.right.at("LieDown")
        )
    );

    env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        deploy_cfg,
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate)
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");

    if(cfg["trace_file"] && !cfg["trace_file"].as<std::string>().empty())
    {
        auto trace_path = std::filesystem::path(
            cfg["trace_file"].as<std::string>()
        );
        if(trace_path.is_relative()) {
            trace_path = param::proj_dir / trace_path;
        }
        if(trace_path.has_parent_path()) {
            std::filesystem::create_directories(trace_path.parent_path());
        }

        auto trace = std::make_shared<std::ofstream>(trace_path);
        if(!trace->is_open())
        {
            throw std::runtime_error(
                "Failed to open stairs trace file: " + trace_path.string()
            );
        }
        *trace << "time_s,policy_step,ang_vel_x,ang_vel_y,ang_vel_z,"
                  "gravity_x,gravity_y,gravity_z";
        for(const char* name : kGo2JointNames)
        {
            *trace << ',' << name << "_q_actual"
                   << ',' << name << "_dq_actual"
                   << ',' << name << "_q_target"
                   << ',' << name << "_kp"
                   << ',' << name << "_kd";
        }
        for(std::size_t i = 0; i < 16; ++i) {
            *trace << ",raw_action_" << i;
        }
        *trace << ",safety_stop,safety_reason,safety_joint,safety_metric,"
                  "safety_max_q_error,safety_max_q_error_joint,"
                  "safety_max_abs_dq,safety_max_abs_dq_joint,"
                  "safety_max_abs_raw_action,safety_max_raw_action_index,"
                  "safety_max_kp,safety_kp_saturated_frames,"
                  "safety_orientation_angle";
        *trace << '\n';

        const auto trace_start = std::chrono::steady_clock::now();
        policy_step_hook = [this, trace, trace_start]()
        {
            const auto raw = env->action_manager->raw_actions();
            const auto processed = env->action_manager->processed_actions();
            const std::size_t joint_count = env->robot->data.joint_ids_map.size();
            if(raw.size() != 16 || processed.size() != joint_count * 3) {
                return;
            }

            float max_kp = 0.0f;
            for(std::size_t i = joint_count; i < joint_count * 2; ++i) {
                max_kp = std::max(max_kp, processed[i]);
            }
            if(env->episode_length != safety_last_kp_episode_length_)
            {
                safety_last_kp_episode_length_ = env->episode_length;
                if(max_kp >= safety_kp_saturation_limit_) {
                    safety_kp_saturation_frames_ += 1;
                } else {
                    safety_kp_saturation_frames_ = 0;
                }
            }

            const auto& data = env->robot->data;
            const auto safety = evaluate_safety(
                data,
                raw,
                processed,
                safety_q_error_limit_,
                safety_dq_limit_,
                safety_raw_action_limit_,
                safety_kp_saturation_limit_,
                safety_kp_saturation_frames_limit_,
                safety_kp_saturation_frames_,
                safety_bad_orientation_limit_,
                safety_enabled_
            );
            if(!safety_stop_requested_.load() && safety.triggered)
            {
                safety_stop_requested_.store(true);
                safety_stop_reason_ = safety.reason;
                safety_stop_joint_ = safety.joint;
                safety_stop_metric_ = safety.metric;
                if(!safety_stop_reported_)
                {
                    spdlog::critical(
                        "STAIRS_SAFETY_STOP reason={} joint={} metric={:.3f} "
                        "q_err={:.3f}({}) dq={:.3f}({}) raw={:.3f} kp={:.1f} "
                        "kp_frames={} orient={:.3f}rad",
                        safety.reason,
                        safety.joint,
                        safety.metric,
                        safety.max_q_error,
                        safety.max_q_error_joint,
                        safety.max_abs_dq,
                        safety.max_abs_dq_joint,
                        safety.max_abs_raw_action,
                        safety.max_kp,
                        safety.kp_saturated_frames,
                        safety.orientation_angle
                    );
                    safety_stop_reported_ = true;
                }
            }

            const double time_s = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - trace_start
            ).count();
            *trace << std::fixed << std::setprecision(7)
                   << time_s << ',' << env->episode_length
                   << ',' << data.root_ang_vel_b[0]
                   << ',' << data.root_ang_vel_b[1]
                   << ',' << data.root_ang_vel_b[2]
                   << ',' << data.projected_gravity_b[0]
                   << ',' << data.projected_gravity_b[1]
                   << ',' << data.projected_gravity_b[2];
            for(std::size_t i = 0; i < joint_count; ++i)
            {
                *trace << ',' << data.joint_pos[i]
                       << ',' << data.joint_vel[i]
                       << ',' << processed[i]
                       << ',' << processed[joint_count + i]
                       << ',' << processed[2 * joint_count + i];
            }
            for(float value : raw) {
                *trace << ',' << value;
            }
            *trace << ',' << (safety_stop_requested_.load() ? 1 : 0)
                   << ',' << safety.reason
                   << ',' << safety.joint
                   << ',' << safety.metric
                   << ',' << safety.max_q_error
                   << ',' << safety.max_q_error_joint
                   << ',' << safety.max_abs_dq
                   << ',' << safety.max_abs_dq_joint
                   << ',' << safety.max_abs_raw_action
                   << ',' << safety.max_raw_action_index
                   << ',' << safety.max_kp
                   << ',' << safety.kp_saturated_frames
                   << ',' << safety.orientation_angle;
            *trace << '\n';
            if(env->episode_length % 25 == 0) {
                trace->flush();
            }
        };
        spdlog::info("Stairs trace enabled: {}", trace_path.string());
    }

    this->registered_checks.emplace_back(
        std::make_pair(
            [this]()->bool{
                return isaaclab::mdp::bad_orientation(
                    env.get(), safety_bad_orientation_limit_
                );
            },
            FSMStringMap.right.at("Passive")
        )
    );
}

void State_RLBase::run()
{
    using clock = std::chrono::steady_clock;
    const auto now = clock::now();
    if(now >= next_policy_step_) {
        const auto started = clock::now();
        env->step();
        if(policy_step_hook) policy_step_hook();
        const auto finished = clock::now();
        // This is a post-return deadline check, NOT an independent watchdog.
        // A hung native inference blocks this FSM and requires hardware stop.
        if(std::chrono::duration<double>(finished - started).count() > 0.020) {
            FSMState::latch_stop("policy_step_exceeded_20ms");
            return;
        }
        last_policy_result_ = finished;
        have_policy_result_ = true;
        next_policy_step_ = started + std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(env->step_dt));
    }
    if(!have_policy_result_ || std::chrono::duration<double>(
            clock::now() - last_policy_result_).count() > 0.040) {
        FSMState::latch_stop("policy_result_missing_or_stale");
        return;
    }
    env->robot->update();
    auto action = env->action_manager->processed_actions();
    auto raw = env->action_manager->raw_actions();
    const std::size_t joint_count = env->robot->data.joint_ids_map.size();
    const bool has_adaptive_gains = action.size() == joint_count * 3;
    if(action.size() != joint_count && !has_adaptive_gains)
    {
        spdlog::critical(
            "Invalid processed action size: expected {} or {}, got {}",
            joint_count,
            joint_count * 3,
            action.size()
        );
        FSMState::latch_stop("invalid_processed_action_size");
        return;
    }

    if(has_adaptive_gains)
    {
        float max_kp = 0.0f;
        for(std::size_t i = joint_count; i < joint_count * 2; ++i) {
            max_kp = std::max(max_kp, action[i]);
        }
        if(env->episode_length != safety_last_kp_episode_length_)
        {
            safety_last_kp_episode_length_ = env->episode_length;
            if(max_kp >= safety_kp_saturation_limit_) {
                safety_kp_saturation_frames_ += 1;
            } else {
                safety_kp_saturation_frames_ = 0;
            }
        }
    }

    const auto safety = evaluate_safety(
        env->robot->data,
        raw,
        action,
        safety_q_error_limit_,
        safety_dq_limit_,
        safety_raw_action_limit_,
        safety_kp_saturation_limit_,
        safety_kp_saturation_frames_limit_,
        safety_kp_saturation_frames_,
        safety_bad_orientation_limit_,
        safety_enabled_
    );
    if(!safety_stop_requested_.load() && safety.triggered)
    {
        safety_stop_requested_.store(true);
        safety_stop_reason_ = safety.reason;
        safety_stop_joint_ = safety.joint;
        safety_stop_metric_ = safety.metric;
        if(!safety_stop_reported_)
        {
            spdlog::critical(
                "STAIRS_SAFETY_STOP reason={} joint={} metric={:.3f} "
                "q_err={:.3f}({}) dq={:.3f}({}) raw={:.3f} kp={:.1f} "
                "kp_frames={} orient={:.3f}rad",
                safety.reason,
                safety.joint,
                safety.metric,
                safety.max_q_error,
                safety.max_q_error_joint,
                safety.max_abs_dq,
                safety.max_abs_dq_joint,
                safety.max_abs_raw_action,
                safety.max_kp,
                safety.kp_saturated_frames,
                safety.orientation_angle
            );
            safety_stop_reported_ = true;
        }
    }

    if(safety_stop_requested_.load())
    {
        static constexpr float kSafetyHoldKp = 20.0f;
        static constexpr float kSafetyHoldKd = 3.0f;
        for(std::size_t i = 0; i < joint_count; ++i)
        {
            const auto sdk_id = static_cast<std::size_t>(
                env->robot->data.joint_ids_map[i]
            );
            auto& motor = lowcmd->msg_.motor_cmd()[sdk_id];
            motor.q() = env->robot->data.joint_pos[i];
            motor.dq() = 0.0f;
            motor.kp() = kSafetyHoldKp;
            motor.kd() = kSafetyHoldKd;
            motor.tau() = 0.0f;
        }
        return;
    }

    for(std::size_t i = 0; i < joint_count; ++i)
    {
        const auto sdk_id = static_cast<std::size_t>(
            env->robot->data.joint_ids_map[i]
        );
        auto& motor = lowcmd->msg_.motor_cmd()[sdk_id];
        motor.q() = action[i];
        if(has_adaptive_gains)
        {
            motor.kp() = action[joint_count + i];
            motor.kd() = action[2 * joint_count + i];
        }
    }
}
