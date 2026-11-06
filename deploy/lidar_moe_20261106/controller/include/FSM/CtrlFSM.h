// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include <unitree/common/thread/recurrent_thread.hpp>
#include "BaseState.h"
#include "FSMState.h"
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

class CtrlFSM
{
public:
    CtrlFSM(std::shared_ptr<BaseState> initstate)
    {
        // Initialize FSM states
        states.push_back(std::move(initstate));

    }

    CtrlFSM(YAML::Node cfg)
    {
        auto fsms = cfg["_"]; // enabled FSMs

        // register FSM string map; used for state transition
        for (auto it = fsms.begin(); it != fsms.end(); ++it)
        {
            std::string fsm_name = it->first.as<std::string>();
            int id = it->second["id"].as<int>();
            FSMStringMap.insert({id, fsm_name});
        }

        // Initialize FSM states
        for (auto it = fsms.begin(); it != fsms.end(); ++it)
        {
            std::string fsm_name = it->first.as<std::string>();
            int id = it->second["id"].as<int>();
            std::string fsm_type = it->second["type"] ? it->second["type"].as<std::string>() : fsm_name;
            auto fsm_class = getFsmMap().find("State_" + fsm_type);
            if (fsm_class == getFsmMap().end()) {
                throw std::runtime_error("FSM: Unknown FSM type " + fsm_type);
            }
            auto state_instance = fsm_class->second(id, fsm_name);
            add(state_instance);
        }
    }

    void start(bool start_in_hold = false)
    {
        // Start From State_Passive
        if(states.empty() || states[0]->getStateString() != "Passive")
            throw std::runtime_error("First test must start in Passive");
        currentState = states[0];
        if(start_in_hold) {
            for(const auto& s:states) if(s->getStateString()=="FixStand") currentState=s;
            if(currentState->getStateString()!="FixStand")
                throw std::runtime_error("Handoff requires FixStand");
        }
        state_entered_ = false;

        fsm_thread_ = std::make_shared<unitree::common::RecurrentThread>(
            "FSM", 0, this->dt * 1e6, &CtrlFSM::run_, this);
        spdlog::info("FSM: Start {}", currentState->getStateString());
    }

    void add(std::shared_ptr<BaseState> state)
    {
        for(auto & s : states)
        {
            if(s->isState(state->getState()))
            {
                spdlog::error("FSM: State_{} already exists", state->getStateString());
                std::exit(0);
            }
        }

        states.push_back(std::move(state));
    }
    
    ~CtrlFSM()
    {
        join_worker();
        exit_state_once();
        states.clear();
    }

    void stop_for_process_exit()
    {
        join_worker();
        FSMState::latch_stop("process_exit");
        exit_state_once();
        currentState.reset();
    }

    std::vector<std::shared_ptr<BaseState>> states;
private:
    const double dt = 0.001;

    void join_worker()
    {
        // This SDK's RecurrentThread destructor destroys its callback before
        // Thread's destructor waits. Explicit Wait first sets mQuit and waits
        // for the callback to finish, while its captures and FSM still exist.
        if(fsm_thread_) {
            fsm_thread_->Wait();
            fsm_thread_.reset();
        }
    }

    void exit_state_once()
    {
        if(currentState && state_entered_) {
            state_entered_ = false;
            currentState->exit();
        }
    }

    void run_() noexcept
    {
        if(!FSMState::begin_cycle()) return;
        try {
            if(FSMState::lidar_service) FSMState::lidar_service(
                currentState->getStateString()=="Stairs",FSMState::latched());
            if(FSMState::latched()) {
                FSMState::finish_damping_cycle();
                exit_state_once();
                return;
            }
            if(!state_entered_) {
                currentState->enter();
                state_entered_ = true;
            }
            currentState->pre_run();
            if(FSMState::latched()) {
                FSMState::finish_damping_cycle();
                exit_state_once();
                return;
            }
            if(currentState->getStateString() == "Passive") {
                const auto input = FSMState::snapshot();
                for(std::size_t i = 0; i < 12; ++i)
                    FSMState::lowcmd->msg_.motor_cmd()[i].q() = static_cast<float>(input.q[i]);
            } else {
                currentState->run();
            }

            // All state writes, including enter(), happen while the publisher
            // mutex is held and BEFORE the one checked publication boundary.
            int next_mode = 0;
            for(const auto& check : currentState->registered_checks) {
                if(check.first()) { next_mode = check.second; break; }
            }
            if(next_mode && !currentState->isState(next_mode)) {
                for(const auto& state : states) {
                    if(!state->isState(next_mode)) continue;
                    const auto next_name = state->getStateString();
                    if(next_name == "LieDown" || next_name == "Passive") {
                        FSMState::latch_stop("stop_transition_" + next_name);
                        break;
                    }
                    if(next_name != "FixStand" && next_name != "Stairs") {
                        FSMState::latch_stop("unsupported_state_transition");
                        break;
                    }
                    spdlog::info("FIRST_TEST state {} -> {}",
                                 currentState->getStateString(), next_name);
                    exit_state_once();
                    currentState = state;
                    // Mark before enter so partial setup exceptions get exit().
                    state_entered_ = true;
                    currentState->enter();
                    break;
                }
            }
            currentState->post_run();
            if(FSMState::latched()) {
                spdlog::warn("FIRST_TEST_LATCHED reason={}", FSMState::stop_reason());
                // Fixed damping has already crossed the publication boundary.
                // No policy re-entry or rearm is permitted in this process.
                exit_state_once();
            }
        } catch(const std::exception& error) {
            FSMState::latch_stop(std::string("fsm_exception: ") + error.what());
            FSMState::finish_damping_cycle();
            try { exit_state_once(); } catch(...) {}
            spdlog::critical("FIRST_TEST_LATCHED reason={}", FSMState::stop_reason());
        } catch(...) {
            FSMState::latch_stop("unknown_fsm_exception");
            FSMState::finish_damping_cycle();
            try { exit_state_once(); } catch(...) {}
        }
    }

    bool state_entered_ = false;
    std::shared_ptr<BaseState> currentState;
    unitree::common::RecurrentThreadPtr fsm_thread_;
};
