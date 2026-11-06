// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include <atomic>
#include <chrono>
#include <functional>
#include <string>

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        enter_time_ = std::chrono::steady_clock::now();
        auto_lie_reported_ = false;
        safety_stop_requested_.store(false);
        safety_stop_reported_ = false;
        safety_stop_reason_.clear();
        safety_stop_joint_.clear();
        safety_stop_metric_ = 0.0f;
        safety_kp_saturation_frames_ = 0;
        safety_last_kp_episode_length_ = -1;

        // set gain
        for (int i = 0; i < env->robot->data.joint_stiffness.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].kp() = env->robot->data.joint_stiffness[i];
            lowcmd->msg_.motor_cmd()[i].kd() = env->robot->data.joint_damping[i];
            lowcmd->msg_.motor_cmd()[i].dq() = 0;
            lowcmd->msg_.motor_cmd()[i].tau() = 0;
        }

        // First-test overlay: one owner for env/data/history/safety state.
        // No worker thread races run() or registered transition checks.
        env->reset();
        next_policy_step_ = std::chrono::steady_clock::now();
        have_policy_result_ = false;
    }

    void run();
    
    void exit() {}

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::chrono::steady_clock::time_point next_policy_step_;
    std::chrono::steady_clock::time_point last_policy_result_;
    bool have_policy_result_ = false;
    std::function<void()> policy_step_hook;
    std::chrono::steady_clock::time_point enter_time_;
    double auto_lie_seconds_ = 0.0;
    bool auto_lie_reported_ = false;
    bool safety_enabled_ = true;
    std::atomic<bool> safety_stop_requested_{false};
    bool safety_stop_reported_ = false;
    std::string safety_stop_reason_;
    std::string safety_stop_joint_;
    float safety_stop_metric_ = 0.0f;
    float safety_q_error_limit_ = 0.90f;
    float safety_dq_limit_ = 13.0f;
    float safety_raw_action_limit_ = 5.0f;
    float safety_kp_saturation_limit_ = 69.5f;
    float safety_bad_orientation_limit_ = 1.0f;
    int safety_kp_saturation_frames_limit_ = 3;
    int safety_kp_saturation_frames_ = 0;
    long safety_last_kp_episode_length_ = -1;
};

REGISTER_FSM(State_RLBase)
