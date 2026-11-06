#pragma once

#include "FSM/State_FixStand.h"
#include "StairsCommandBuffer.h"
#include <algorithm>
#include <cctype>

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
                       !stairs_command_buffer.armed() || !stand_ready())
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
        stairs_command_buffer.begin_selection();
        spdlog::info(
            "Select stair speed before G: [1-4] = 0.5-0.8 m/s "
            "(current selection {:.2f} m/s).",
            stairs_command_buffer.selected_speed()
        );
    }

    void run() override
    {
        State_FixStand::run();
        if(!keyboard || !keyboard->on_pressed)
        {
            return;
        }

        const auto key = normalized_key(keyboard->key());
        if(stairs_command_buffer.select(key))
        {
            spdlog::info(
                "Stairs speed armed: vx={:.2f} m/s; press G when stand is stable.",
                stairs_command_buffer.selected_speed()
            );
        }
        else if(key == "g" && !stairs_command_buffer.armed())
        {
            spdlog::warn("Select stair speed [1-4] before pressing G.");
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

    static double now_seconds()
    {
        return static_cast<double>(
            unitree::common::GetCurrentTimeMillisecond()
        ) * 1.0e-3;
    }

    bool stand_ready() const
    {
        return now_seconds() - entered_at_ >= settle_duration_;
    }

    double entered_at_ = 0.0;
    double settle_duration_ = 0.0;
};

REGISTER_FSM(State_StairsReady)
