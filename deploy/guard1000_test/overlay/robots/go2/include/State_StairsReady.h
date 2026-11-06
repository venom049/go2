#pragma once

#include "FSM/State_FixStand.h"
#include "StairsCommandBuffer.h"
#include "StandReadiness.h"
#include <algorithm>
#include <cctype>
#include <cmath>

class State_StairsReady : public State_FixStand
{
public:
    State_StairsReady(int state, std::string state_string)
    : State_FixStand(state, std::move(state_string))
    {
        const auto times = param::config["FSM"]["FixStand"]["ts"]
            .as<std::vector<float>>();
        settle_duration_ = *std::max_element(times.begin(), times.end());
        registered_checks.emplace_back(
            std::make_pair(
                [this]()->bool{
                    if(!keyboard || !keyboard->on_pressed ||
                       !stairs_command_buffer.armed() || !stand_ready() ||
                       !selected_speed_is_fixed())
                    {
                        return false;
                    }
                    return normalized_key(keyboard->key()) == "g";
                },
                FSMStringMap.right.at("Stairs")
            )
        );
    }

    void enter() override
    {
        State_FixStand::enter();
        entered_at_ = now_seconds();
        readiness_.reset(); ready_=false; ready_reported_=false;
        stairs_command_buffer.begin_selection();
        spdlog::info(
            "First test: only [1] arms the configured {:.2f} m/s before G "
            "(current selection {:.2f} m/s).",
            configured_speed(), stairs_command_buffer.selected_speed()
        );
    }

    void run() override
    {
        State_FixStand::run();
        const auto s=FSMState::snapshot();
        double norm=0.;for(double v:s.quaternion_wxyz) norm+=v*v;
        const double x=s.quaternion_wxyz[1], y=s.quaternion_wxyz[2];
        const double tilt=std::acos(std::clamp(1.-2.*(x*x+y*y)/norm,-1.,1.));
        if(!std::isfinite(tilt) || tilt>.45) {FSMState::latch_stop("stand_tilt_limit");return;}
        for(std::size_t i=0;i<12;++i) {
            if(!std::isfinite(s.dq[i]) || std::abs(s.dq[i])>5. ||
               std::abs(s.q[i]-lowcmd->msg_.motor_cmd()[i].q())>.5) {
                FSMState::latch_stop("stand_tracking_or_speed_limit");return;
            }
        }
        ready_=readiness_.update(s,target_,now_seconds(),now_seconds()-entered_at_>=settle_duration_);
        if(ready_ && !ready_reported_) {
            spdlog::info("STAND_LOCKED: measured joint error <=0.18 rad, speed <=0.8 rad/s, "
                         "tilt <=0.25 rad and gyro <=0.35 rad/s for 0.75s. No contact guarantee.");
            ready_reported_=true;
        }
        if(now_seconds()-entered_at_>15. && !ready_) {
            FSMState::latch_stop("stand_failed_to_settle"); return;
        }
        if(!keyboard || !keyboard->on_pressed)
        {
            return;
        }

        const auto key = normalized_key(keyboard->key());
        if(key == "1")
        {
            if(!stairs_command_buffer.select("1") || !selected_speed_is_fixed()) {
                stairs_command_buffer.begin_selection();
                FSMState::latch_stop("unexpected_first_test_speed_configuration");
                return;
            }
            spdlog::info("First-test speed armed: vx={:.2f} m/s; G starts the software-limited test.", configured_speed());
        }
        else if(key == "w" || key == "s" || key == "up" || key == "down" ||
                (key.size() == 1 && key[0] >= '2' && key[0] <= '9'))
        {
            stairs_command_buffer.begin_selection();
            spdlog::warn("Speed selection cleared: only [1] can arm the configured fixed speed in this test.");
        }
        else if(key == "g" && !stairs_command_buffer.armed())
        {
            spdlog::warn("Press [1] to arm the configured fixed command before G.");
        }
        else if(key == "g" && !stand_ready())
        {
            spdlog::warn("FixStand is still settling; wait, then press G again.");
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

    static float configured_speed()
    {
        const float speed=param::config["FSM"]["Stairs"]["speed_override"].as<float>();
        if(!std::isfinite(speed) ||
           (std::abs(speed-0.45f)>1.0e-6f && std::abs(speed-0.6f)>1.0e-6f))
            throw std::runtime_error("Unexpected fixed stairs speed");
        return speed;
    }

    static bool selected_speed_is_fixed()
    {
        const float speed = stairs_command_buffer.selected_speed();
        return std::isfinite(speed) && std::abs(speed - configured_speed()) <= 1.0e-6f;
    }

    static double now_seconds() { return FSMState::steady_seconds(); }

    bool stand_ready() const
    {
        return ready_;
    }

    double entered_at_ = 0.0;
    double settle_duration_ = 0.0;
    go2_first_test::StandReadiness readiness_;
    bool ready_=false, ready_reported_=false;
};

REGISTER_FSM(State_StairsReady)
