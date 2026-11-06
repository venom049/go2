#pragma once

#include "FSM/FSMState.h"
#include "LinearInterpolator.h"
#include <cctype>
#include <mutex>

class State_LieDown : public FSMState
{
public:
    State_LieDown(int state, std::string state_string = "LieDown")
    : FSMState(state, state_string)
    {
        const auto cfg = param::config["FSM"][state_string];
        kp_ = cfg["kp"].as<std::vector<float>>();
        kd_ = cfg["kd"].as<std::vector<float>>();
        ts_ = cfg["ts"].as<std::vector<float>>();
        qs_ = cfg["qs"].as<std::vector<std::vector<float>>>();
        if(kp_.size() != 12 || kd_.size() != 12 ||
           ts_.size() != qs_.size() || ts_.size() < 2 ||
           qs_.back().size() != 12)
        {
            throw std::runtime_error("Invalid Go2 LieDown configuration.");
        }

        registered_checks.emplace_back(
            std::make_pair(
                [this]()->bool{
                    return complete_ && keyboard && keyboard->on_pressed &&
                           normalized_key(keyboard->key()) == "x";
                },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void enter() override
    {
        {
            std::lock_guard<std::mutex> lock(lowstate->mutex_);
            q_start_.resize(12);
            for(std::size_t i = 0; i < q_start_.size(); ++i)
            {
                q_start_[i] = lowstate->msg_.motor_state()[i].q();
            }
        }

        for(std::size_t i = 0; i < kp_.size(); ++i)
        {
            auto& motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = kp_[i];
            motor.kd() = kd_[i];
            motor.dq() = 0;
            motor.tau() = 0;
        }
        qs_[0] = q_start_;
        t0_ = now_seconds();
        complete_ = false;
        if(keyboard)
        {
            keyboard->clear_events();
        }
        spdlog::info(
            "LieDown started; wait for the low pose before pressing X again."
        );
    }

    void run() override
    {
        const float t = static_cast<float>(now_seconds() - t0_);
        const auto q = linear_interpolate(t, ts_, qs_);
        for(std::size_t i = 0; i < q.size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].q() = q[i];
        }

        if(t >= ts_.back() && !complete_)
        {
            complete_ = true;
            spdlog::info("LieDown complete; press X again for Passive damping.");
        }

        if(keyboard && keyboard->on_pressed && !complete_ &&
           normalized_key(keyboard->key()) == "x")
        {
            spdlog::warn(
                "LieDown is still moving; wait until complete, then press X again."
            );
        }
    }

private:
    static std::string normalized_key(std::string key)
    {
        if(key.size() == 1)
        {
            key[0] = static_cast<char>(std::tolower(
                static_cast<unsigned char>(key[0])
            ));
        }
        return key;
    }

    static double now_seconds()
    {
        return static_cast<double>(
            unitree::common::GetCurrentTimeMillisecond()
        ) * 1.0e-3;
    }

    std::vector<float> kp_;
    std::vector<float> kd_;
    std::vector<float> ts_;
    std::vector<std::vector<float>> qs_;
    std::vector<float> q_start_;
    double t0_ = 0.0;
    bool complete_ = false;
};

REGISTER_FSM(State_LieDown)
