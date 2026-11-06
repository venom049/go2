#pragma once

#include "Types.h"
#include "param.h"
#include "FSM/BaseState.h"
#include "isaaclab/devices/keyboard/keyboard.h"
#include "unitree_joystick_dsl.hpp"
#include <cctype>
#include <atomic>
#include <chrono>
#include <mutex>
#include "FirstTestGuard.h"

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

        // SDK isTimeout() reads an unsynchronized internal timestamp. This
        // overlay instead checks the mutex-protected device tick at publish.
    }

    static double steady_seconds()
    {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    static go2_first_test::GuardInput snapshot()
    {
        go2_first_test::GuardInput input;
        if(!lowstate) return input;
        std::lock_guard<std::mutex> lock(lowstate->mutex_);
        const auto& msg = lowstate->msg_;
        input.state_available = true;  // Storage exists; tick advance proves liveness.
        input.tick = msg.tick();
        for(std::size_t i = 0; i < 4; ++i)
            input.quaternion_wxyz[i] = msg.imu_state().quaternion()[i];
        for(std::size_t i = 0; i < 3; ++i) {
            input.gyro[i] = msg.imu_state().gyroscope()[i];
            input.accel[i] = msg.imu_state().accelerometer()[i];
        }
        for(std::size_t i = 0; i < 12; ++i) {
            input.q[i] = msg.motor_state()[i].q();
            input.dq[i] = msg.motor_state()[i].dq();
        }
        // The SDK exposes neither a synchronized device timestamp nor a
        // race-free receive timestamp. Leave both absolute ages unavailable.
        return input;
    }

    // Policy thread may only request a stop; Guard and LowCmd stay FSM-owned.
    static void request_stop() noexcept { stop_requested.store(true); }
    static void latch_stop(const std::string& reason) { guard.latch(reason); }
    static bool latched() { return guard.latched(); }
    static const std::string& stop_reason() { return guard.reason(); }

    static bool begin_cycle()
    {
        if(!control_enabled) return false;
        if(!lowcmd || !lowcmd->trylock()) return false;
        command_locked = true;
        if(stop_requested.load()) guard.latch("asynchronous_policy_stop");
        return true;
    }

    static void write_command(const go2_first_test::GuardOutput& output)
    {
        for(std::size_t i = 0; i < 12; ++i) {
            auto& motor = lowcmd->msg_.motor_cmd()[i];
            motor.mode() = 1;
            motor.q() = static_cast<float>(output.q[i]);
            motor.dq() = static_cast<float>(output.dq[i]);
            motor.kp() = static_cast<float>(output.kp[i]);
            motor.kd() = static_cast<float>(output.kd[i]);
            motor.tau() = static_cast<float>(output.tau[i]);
        }
        // Go2 has 12 actuated joints; normalize all spare slots to no output.
        for(std::size_t i = 12; i < lowcmd->msg_.motor_cmd().size(); ++i) {
            auto& motor = lowcmd->msg_.motor_cmd()[i];
            motor.mode() = 0;
            motor.q() = motor.dq() = motor.kp() = motor.kd() = motor.tau() = 0.0f;
        }
    }

    static void finish_damping_cycle()
    {
        if(!command_locked) return;
        write_command(go2_first_test::Guard::damping());
        command_locked = false;
        lowcmd->unlockAndPublish();
    }

    static bool publish_damping_once()
    {
        if(!begin_cycle()) return false;
        finish_damping_cycle();
        return true;
    }

    static void consume_stop_keys()
    {
        if(!keyboard) return;
        std::string key;
        while(keyboard->pop_key(key)) {
            if(key.size() == 1) key[0] = static_cast<char>(std::tolower(
                static_cast<unsigned char>(key[0])));
            if(key == "x" || key == " " || key == "0")
                guard.latch("operator_stop_key");
        }
    }

    void pre_run() override
    {
        lowstate->update();
        if(keyboard) keyboard->update();
        // Drain FIFO stop events, not only the latest key which may overwrite X.
        // Normal selection/transition keys still use keyboard->key/on_pressed.
        consume_stop_keys();
    }

    void post_run() override
    {
        if(!command_locked) throw std::logic_error("publication without command lock");
        go2_first_test::GuardOutput proposed;
        for(std::size_t i = 0; i < 12; ++i) {
            const auto& motor = lowcmd->msg_.motor_cmd()[i];
            proposed.q[i] = motor.q();
            proposed.dq[i] = motor.dq();
            proposed.kp[i] = motor.kp();
            proposed.kd[i] = motor.kd();
            proposed.tau[i] = motor.tau();
        }
        consume_stop_keys();
        if(stop_requested.load()) guard.latch("asynchronous_policy_stop");
        const auto measured = snapshot();
        const auto result = guard.filter(measured, proposed, steady_seconds(),
                                         getStateString() == "Stairs");
        write_command(result.command);
        last_submitted_command = result.command;
        have_submitted_command = true;
        command_locked = false;
        lowcmd->unlockAndPublish();
        if(result.policy_allowed) ++accepted_stairs_publications;
        // One immutable rejection snapshot; main logs it after damping has
        // been submitted. No file I/O is added to the publication thread.
        if(result.latched && !rejection_ready.load(std::memory_order_relaxed)) {
            rejected_input = measured;
            rejected_proposal = proposed;
            rejection_reason = result.reason;
            rejected_state = getStateString();
            accepted_before_rejection = accepted_stairs_publications;
            rejection_ready.store(true, std::memory_order_release);
        }
    }

    static void flush_rejection_diagnostic()
    {
        // Called only by main. The writer never changes a published snapshot.
        if(rejection_reported || !rejection_ready.load(std::memory_order_acquire)) return;
        rejection_reported = true;
        spdlog::error("GUARD_REJECTION reason={} state={} tick={} accepted_stairs_publications={}",
            rejection_reason, rejected_state, rejected_input.tick, accepted_before_rejection);
        for(std::size_t i=0;i<12;++i) {
            const double estimate = rejected_proposal.kp[i]*(rejected_proposal.q[i]-rejected_input.q[i])
                + rejected_proposal.kd[i]*(rejected_proposal.dq[i]-rejected_input.dq[i])
                + rejected_proposal.tau[i];
            spdlog::error("GUARD_JOINT joint={} q={} dq={} target_q={} target_dq={} kp={} kd={} tau_ff={} estimated_nm={} limit_nm={} dq_limit_rad_s={} q_lower_rad={} q_upper_rad={}",
                go2_first_test::kSdkJointNames[i], rejected_input.q[i], rejected_input.dq[i],
                rejected_proposal.q[i], rejected_proposal.dq[i], rejected_proposal.kp[i],
                rejected_proposal.kd[i], rejected_proposal.tau[i], estimate,
                go2_first_test::kEffortUpper[i], go2_joint_limits::kVelocityUpper[i],
                go2_first_test::kTargetLower[i], go2_first_test::kTargetUpper[i]);
        }
        spdlog::default_logger()->flush();
    }

    inline static std::atomic<bool> rejection_ready{false};
    inline static bool rejection_reported = false; // main thread only
    inline static go2_first_test::GuardInput rejected_input;
    inline static go2_first_test::GuardOutput rejected_proposal;
    inline static std::string rejection_reason, rejected_state;
    inline static std::uint64_t accepted_stairs_publications = 0; // FSM only
    inline static std::uint64_t accepted_before_rejection = 0;

    // This records submitted SDK targets, not acknowledged mechanical execution.
    // Only the FSM owner reads/writes this; never expose a background policy writer.
    inline static go2_first_test::GuardOutput last_submitted_command;
    inline static bool have_submitted_command = false;
    inline static go2_first_test::Guard guard;
    inline static std::atomic<bool> stop_requested{false};
    inline static bool command_locked = false;  // FSM owner; shutdown joins first.
    inline static bool control_enabled = false; // Enabled once, only after handoff.

    static std::unique_ptr<LowCmd_t> lowcmd;
    static std::shared_ptr<LowState_t> lowstate;
    static std::shared_ptr<Keyboard> keyboard;
};
