#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"
#include <cctype>

class FSMState : public BaseState
{
public:
    FSMState(int state, std::string state_string) 
    : BaseState(state, state_string) 
    {
        spdlog::info("Initializing State_{} ...", state_string);

        auto transitions = param::config["FSM"][state_string]["transitions"];

        if(transitions)
        {
            auto transition_map = transitions.as<std::map<std::string, std::string>>();

            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn("FSM State_'{}' not found in FSMStringMap!", target_fsm);
                    continue;
                }

                int fsm_id = FSMStringMap.right.at(target_fsm);

                std::string condition = it->second;
                unitree::common::dsl::Parser p(condition);
                auto ast = p.Parse();
                auto func = unitree::common::dsl::Compile(*ast);
                registered_checks.emplace_back(
                    std::make_pair(
                        [func]()->bool{ return func(FSMState::lowstate->joystick); },
                        fsm_id
                    )
                );
            }
        }

        auto keyboard_transitions =
            param::config["FSM"][state_string]["keyboard_transitions"];
        if(keyboard_transitions && keyboard)
        {
            auto transition_map =
                keyboard_transitions.as<std::map<std::string, std::string>>();
            for(auto it = transition_map.begin(); it != transition_map.end(); ++it)
            {
                const std::string target_fsm = it->first;
                if(!FSMStringMap.right.count(target_fsm))
                {
                    spdlog::warn(
                        "FSM State_'{}' not found in FSMStringMap!", target_fsm
                    );
                    continue;
                }

                const int fsm_id = FSMStringMap.right.at(target_fsm);
                const std::string key = it->second;
                registered_checks.emplace_back(
                    std::make_pair(
                        [key]()->bool{
                            if(!keyboard || !keyboard->on_pressed) {
                                return false;
                            }
                            auto current_key = keyboard->key();
                            if(current_key.size() == 1) {
                                current_key[0] = static_cast<char>(std::tolower(
                                    static_cast<unsigned char>(current_key[0])
                                ));
                            }
                            return current_key == key;
                        },
                        fsm_id
                    )
                );
            }
        }

        // register for all states
        registered_checks.emplace_back(
            std::make_pair(
                []()->bool{ return lowstate->isTimeout(); },
                FSMStringMap.right.at("Passive")
            )
        );
    }

    void pre_run()
    {
        lowstate->update();
        if(keyboard) keyboard->update();
    }

    void post_run()
    {
        lowcmd->unlockAndPublish();
    }

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;
};
