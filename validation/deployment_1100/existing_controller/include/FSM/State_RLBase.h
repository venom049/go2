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

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();
                if(policy_step_hook) {
                    policy_step_hook();
                }

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    std::atomic<bool> policy_thread_running{false};
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
