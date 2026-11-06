// MoE calf30 profile regression suite; adapted from test_first_test_guard.cpp.
// Offline C++17 tests: no SDK, ROS, sockets, threads or robot access.
/* Build with g++ -std=c++17 -Wall -Wextra -Werror -pedantic -O2
   -Ideploy/first_test_overlay/include tests/test_first_test_guard.cpp
   -o /tmp/test_first_test_guard ; run /tmp/test_first_test_guard separately. */
#include "FirstTestGuard.h"

#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

using namespace go2_first_test;

#define CHECK(condition) do { if (!(condition)) { \
    std::cerr << "FAILED line " << __LINE__ << ": " << #condition << '\n'; \
    std::exit(1); } } while (false)

GuardInput input() {
    GuardInput value;
    value.state_available = true;
    value.quaternion_wxyz = {1., 0., 0., 0.};
    value.accel = {0., 0., 9.81};
    value.q = {0., .8, -1.5, 0., .8, -1.5, 0., 1., -1.5, 0., 1., -1.5};
    value.tick = 100;
    return value;
}

GuardOutput command() {
    GuardOutput value;
    value.q = input().q;
    value.kp.fill(40.);
    value.kd.fill(1.);
    return value;
}

void is_damping(const GuardResult& result, bool latched) {
    CHECK(result.latched == latched);
    CHECK(!result.policy_allowed && !result.command_allowed);
    for (std::size_t i = 0; i < kJointCount; ++i) {
        CHECK(result.command.q[i] == 0. && result.command.dq[i] == 0.);
        CHECK(result.command.kp[i] == 0. && result.command.kd[i] == 3.);
        CHECK(result.command.tau[i] == 0.);
    }
}

void bootstrap(Guard& guard, GuardInput& state) {
    auto result = guard.filter(state, command(), 0., false);
    is_damping(result, false);
    CHECK(result.reason == "waiting_for_source_tick_advance");
    ++state.tick;
}

void freshness_and_duplicates() {
    Guard guard;
    auto state = input();
    is_damping(guard.filter(state, command(), 10., false), false);
    is_damping(guard.filter(state, command(), 10.001, false), false);
    ++state.tick;
    CHECK(guard.filter(state, command(), 10.002, true).policy_allowed);
    // A 1 kHz publisher can reuse the latest 500 Hz source sample.
    CHECK(guard.filter(state, command(), 10.003, true).policy_allowed);
    CHECK(guard.filter(state, command(), 10.09, true).policy_allowed);
    auto stalled = guard.filter(state, command(), 10.103, true);
    is_damping(stalled, true);
    CHECK(stalled.reason == "source_tick_stalled");
    ++state.tick;
    is_damping(guard.filter(state, command(), 10.104, false), true);
    CHECK(guard.reason() == stalled.reason);
}

void first_sample_cannot_authorize() {
    Guard guard;
    auto state = input();
    is_damping(guard.filter(state, command(), 0., true), false);
    auto result = guard.filter(state, command(), .101, true);
    is_damping(result, true);
    CHECK(result.reason == "source_tick_stalled");
    Guard late;
    is_damping(late.filter(state, command(), 0., false), false);
    ++state.tick;  // New data after a long unobserved gap does not auto-recover.
    CHECK(late.filter(state, command(), .101, true).latched);
}

void tick_wrap_and_regression() {
    Guard wrap;
    auto state = input();
    state.tick = UINT32_C(0xfffffffe);
    is_damping(wrap.filter(state, command(), 0., false), false);
    state.tick = 1;
    CHECK(wrap.filter(state, command(), .002, true).policy_allowed);
    state.tick = 0;
    auto result = wrap.filter(state, command(), .003, true);
    is_damping(result, true);
    CHECK(result.reason == "source_tick_regressed");
}

void stale_source_and_receipt_ages() {
    const std::vector<double> invalid = {
        -.001, .100001, std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::infinity()};
    for (bool source : {false, true}) for (double age : invalid) {
        Guard guard;
        auto state = input();
        if (source) state.source_age_s = age; else state.receipt_age_s = age;
        is_damping(guard.filter(state, command(), 0., false), true);
    }
    Guard valid;
    auto state = input();
    state.source_age_s = .1;
    state.receipt_age_s = .1;
    bootstrap(valid, state);
    CHECK(valid.filter(state, command(), .001, true).policy_allowed);
    // Missing absolute age is explicit nullopt; tick progress still required.
    CHECK(!input().source_age_s && !input().receipt_age_s);
}

void nonfinite_state_every_array() {
    using Mutator = std::function<void(GuardInput&, double)>;
    const std::vector<Mutator> mutate = {
        [](GuardInput& s, double n) { s.quaternion_wxyz[3] = n; },
        [](GuardInput& s, double n) { s.gyro[2] = n; },
        [](GuardInput& s, double n) { s.accel[0] = n; },
        [](GuardInput& s, double n) { s.q[11] = n; },
        [](GuardInput& s, double n) { s.dq[6] = n; }};
    for (auto& change : mutate) for (double bad : {
        std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        Guard guard;
        auto state = input();
        change(state, bad);
        auto result = guard.filter(state, Guard::damping(), 0., false);
        is_damping(result, true);  // Passive never hides a NaN input.
        CHECK(result.reason == "nonfinite_state");
    }
    Guard quaternion;
    auto state = input();
    state.quaternion_wxyz = {0., 0., 0., 0.};
    CHECK(quaternion.filter(state, command(), 0., false).reason == "invalid_imu_quaternion_norm");
    Guard missing;
    state = input(); state.state_available = false;
    CHECK(missing.filter(state, command(), 0., false).reason == "state_missing");
}

void nonfinite_commands_every_array() {
    using Mutator = std::function<void(GuardOutput&, double)>;
    const std::vector<Mutator> mutate = {
        [](GuardOutput& c, double n) { c.q[0] = n; },
        [](GuardOutput& c, double n) { c.dq[1] = n; },
        [](GuardOutput& c, double n) { c.kp[2] = n; },
        [](GuardOutput& c, double n) { c.kd[10] = n; },
        [](GuardOutput& c, double n) { c.tau[11] = n; }};
    for (auto& change : mutate) for (double bad : {
        std::numeric_limits<double>::quiet_NaN(), -std::numeric_limits<double>::infinity()}) {
        Guard guard;
        auto proposed = Guard::damping();
        change(proposed, bad);
        auto result = guard.filter(input(), proposed, 0., false);
        is_damping(result, true);
        CHECK(result.reason == "nonfinite_command");
    }
}

void target_limits_and_sdk_order() {
    for (std::size_t i = 0; i < kJointCount; ++i) for (bool upper : {false, true}) {
        Guard at_boundary;
        auto state = input(); bootstrap(at_boundary, state);
        auto proposed = command();
        proposed.q[i] = upper ? kTargetUpper[i] : kTargetLower[i];
        state.q[i] = proposed.q[i];  // Isolate position boundary from PD-error bound.
        CHECK(at_boundary.filter(state, proposed, .001, true).policy_allowed);
        Guard outside;
        state = input(); bootstrap(outside, state);
        proposed.q[i] += upper ? 1e-7 : -1e-7;
        const double original = proposed.q[i];
        auto result = outside.filter(state, proposed, .001, true);
        is_damping(result, true);
        CHECK(result.reason == "target_position_outside_urdf_range");
        CHECK(proposed.q[i] == original);  // No mutation or silent clipping.

        Guard measured_outside;
        state = input(); bootstrap(measured_outside, state);
        proposed = command();
        state.q[i] = (upper ? kTargetUpper[i] : kTargetLower[i])
                     + (upper ? 1e-7 : -1e-7);
        auto measured_result = measured_outside.filter(state, proposed, .001, true);
        is_damping(measured_result, true);
        CHECK(measured_result.reason == "initial_position_outside_software_range");
    }
    Guard front, rear;
    auto state = input(); bootstrap(front, state);
    auto proposed = command(); proposed.q[1] = 4.0;  // FR thigh: invalid.
    CHECK(front.filter(state, proposed, .001, false).latched);
    state = input(); bootstrap(rear, state);
    proposed = command(); proposed.q[7] = 4.0;  // RR thigh: within URDF.
    state.q[7] = 4.0;
    CHECK(rear.filter(state, proposed, .001, false).command_allowed);
    CHECK(std::string(kSdkJointNames[0]) == "FR_hip");
    CHECK(std::string(kSdkJointNames[3]) == "FL_hip");
    CHECK(std::string(kSdkJointNames[6]) == "RR_hip");
    CHECK(std::string(kSdkJointNames[9]) == "RL_hip");
    // The support document's hip range is narrower than the URDF's +/-60 deg.
    const double hip_48_degrees = static_cast<double>(static_cast<float>(48. * std::acos(-1.) / 180.));
    for (std::size_t i : {0u, 3u, 6u, 9u}) {
        CHECK(std::abs(kTargetLower[i] + hip_48_degrees) < 1e-12);
        CHECK(std::abs(kTargetUpper[i] - hip_48_degrees) < 1e-12);
        Guard old_hip_range;
        auto hip_state = input(); bootstrap(old_hip_range, hip_state);
        auto hip_command = command(); hip_command.q[i] = .9;
        CHECK(old_hip_range.filter(hip_state, hip_command, .001, true).reason
              == "target_position_outside_urdf_range");
    }
}

void folded_passive_is_not_position_control() {
    Guard guard;
    auto state = input();
    state.q[2] = -2.8184; state.q[5] = -2.8053;
    state.q[8] = -2.7550; state.q[11] = -2.7860;
    auto passive = Guard::damping();
    passive.q.fill(1e30);  // Finite placeholder not used by kp=0 damping.
    is_damping(guard.filter(state, passive, 0., false), false);
    ++state.tick;
    auto result = guard.filter(state, passive, .002, false);
    is_damping(result, false);
    CHECK(result.reason == "passive_fixed_damping");
    ++state.tick;
    result = guard.filter(state, command(), .004, false);  // FixStand also guarded.
    is_damping(result, true);
    CHECK(result.reason == "initial_position_outside_software_range");
    CHECK(state.q[2] == -2.8184);
}

void finite_but_excessive_outputs() {
    using Mutator = std::function<void(GuardOutput&)>;
    const std::vector<Mutator> mutate = {
        [](GuardOutput& c) { c.kp[0] = -1.; },
        [](GuardOutput& c) { c.kp[0] = 80.01; },
        [](GuardOutput& c) { c.kd[1] = -1.; },
        [](GuardOutput& c) { c.kd[1] = 5.01; },
        [](GuardOutput& c) { c.dq[2] = 30.0001; },
        [](GuardOutput& c) { c.dq[3] = -30.1001; },
        [](GuardOutput& c) { c.tau[0] = 23.701; },
        [](GuardOutput& c) { c.tau[2] = -45.431; }};
    for (auto& change : mutate) {
        Guard guard;
        auto proposed = command(); change(proposed);
        is_damping(guard.filter(input(), proposed, 0., false), true);
    }
}

void measured_speed_and_estimated_pd_effort() {
    for (double speed : {30.1001, -30.1001, 1e300, -1e300}) {
        Guard guard;
        auto state = input(); bootstrap(guard, state);
        state.dq[7] = speed;  // Finite, but not a valid active control sample.
        auto result = guard.filter(state, command(), .001, true);
        is_damping(result, true);
        CHECK(result.reason == "measured_speed_outside_urdf_range");
    }
    for (double target : {-.8, .8}) {
        Guard jump;
        auto state = input(); bootstrap(jump, state);
        auto proposed = command();
        proposed.q[0] = target;  // Within +/-48 deg, but |40*.8| > 23.7.
        auto result = jump.filter(state, proposed, .001, true);
        is_damping(result, true);
        CHECK(result.reason == "estimated_pd_effort_outside_urdf_range");
        CHECK(proposed.q[0] == target);
        ++state.tick;
        is_damping(jump.filter(state, command(), .002, false), true);
    }
    Guard combined;
    auto state = input(); bootstrap(combined, state);
    auto proposed = command();
    proposed.q[0] = .25;      // P = 10, D = 10, FF = 5 -> 25 > 23.7.
    state.dq[0] = -10.;
    proposed.tau[0] = 5.;
    CHECK(combined.filter(state, proposed, .001, false).reason
          == "estimated_pd_effort_outside_urdf_range");

    Guard boundary;
    state = input(); bootstrap(boundary, state);
    proposed = command();
    proposed.q[0] = .5; proposed.kp[0] = 47.4;
    CHECK(boundary.filter(state, proposed, .001, true).policy_allowed);
    ++state.tick;
    proposed.kp[0] += .00001;
    CHECK(boundary.filter(state, proposed, .002, true).latched);

    Guard passive;
    state = input(); state.dq[0] = 31.;
    is_damping(passive.filter(state, Guard::damping(), 0., false), false);
    ++state.tick;
    is_damping(passive.filter(state, Guard::damping(), .001, false), false);
}

void every_joint_speed_boundary() {
    for (std::size_t i = 0; i < kJointCount; ++i)
        for (double sign : {-1., 1.})
            for (bool measured : {false, true}) {
                const double expected_limit = static_cast<double>(static_cast<float>(i % 3 == 2 ? 30.0 : 30.1));
                CHECK(kVelocityUpper[i] == expected_limit);
                for (bool exceeded : {false, true}) {
                    Guard guard;
                    auto state = input(); bootstrap(guard, state);
                    auto proposed = command();
                    proposed.kd.fill(0.);  // Isolate speed from estimated D torque.
                    const double speed = sign * (expected_limit + (exceeded ? 1e-6 : 0.));
                    if (measured) state.dq[i] = speed; else proposed.dq[i] = speed;
                    const auto result = guard.filter(state, proposed, .001, true);
                    if (exceeded) {
                        is_damping(result, true);
                        CHECK(result.reason == (measured ? "measured_speed_outside_urdf_range"
                                                         : "target_speed_outside_urdf_range"));
                        ++state.tick;
                        state.dq.fill(0.); proposed.dq.fill(0.);
                        const auto recovered = guard.filter(state, proposed, .002, true);
                        is_damping(recovered, true);
                        CHECK(recovered.reason == result.reason);  // No automatic rearm.
                    } else {
                        CHECK(result.command_allowed && result.policy_allowed && !result.latched);
                        CHECK(result.command.dq[i] == proposed.dq[i]);  // No hidden clipping.
                    }
                }
            }
}

void heterogeneous_speeds_do_not_hide_calf_violation() {
    for (bool measured : {false, true}) {
        Guard guard;
        auto state = input(); bootstrap(guard, state);
        auto proposed = command(); proposed.kd.fill(0.);
        auto& speeds = measured ? state.dq : proposed.dq;
        speeds[0] = 20.;  // The largest absolute speed is legal at the hip.
        speeds[11] = -30.01;  // A smaller absolute speed is illegal at the calf.
        auto result = guard.filter(state, proposed, .001, true);
        is_damping(result, true);
        CHECK(result.reason == (measured ? "measured_speed_outside_urdf_range"
                                         : "target_speed_outside_urdf_range"));
    }
    Guard historical;
    auto state = input(); bootstrap(historical, state);
    state.dq[11] = -13.312929153442383;  // Previously stopped at scalar 13 rad/s.
    auto proposed = command(); proposed.kd.fill(0.);
    auto result = historical.filter(state, proposed, .001, true);
    CHECK(result.command_allowed && result.policy_allowed && !result.latched);
}

void every_joint_float_wire_speed_boundary() {
    for (std::size_t i = 0; i < kJointCount; ++i)
        for (double sign : {-1., 1.})
            for (bool measured : {false, true})
                for (bool exceeded : {false, true}) {
                    Guard guard;
                    auto state = input(); bootstrap(guard, state);
                    auto proposed = command(); proposed.kd.fill(0.);
                    const float wire_limit = static_cast<float>(i % 3 == 2 ? 30.0 : 30.1);
                    const double wire_speed = sign * static_cast<double>(exceeded
                        ? std::nextafter(wire_limit, std::numeric_limits<float>::infinity())
                        : wire_limit);
                    if (measured) state.dq[i] = wire_speed; else proposed.dq[i] = wire_speed;
                    const auto result = guard.filter(state, proposed, .001, true);
                    if (exceeded) {
                        is_damping(result, true);
                        CHECK(result.reason == (measured ? "measured_speed_outside_urdf_range"
                                                         : "target_speed_outside_urdf_range"));
                    } else {
                        CHECK(result.command_allowed && result.policy_allowed && !result.latched);
                    }
                }
}

void every_joint_effort_limits() {
    for (std::size_t i = 0; i < kJointCount; ++i) for (double sign : {-1., 1.}) {
        const double expected_limit = i % 3 == 2 ? 45.43 : 23.7;
        CHECK(kEffortUpper[i] == expected_limit);
        Guard boundary;
        auto state = input(); bootstrap(boundary, state);
        auto proposed = command(); proposed.kd.fill(0.);
        proposed.tau[i] = sign * expected_limit;
        CHECK(boundary.filter(state, proposed, .001, true).policy_allowed);

        Guard feedforward_outside;
        state = input(); bootstrap(feedforward_outside, state);
        proposed.tau[i] = sign * (expected_limit + 1e-6);
        auto result = feedforward_outside.filter(state, proposed, .001, true);
        is_damping(result, true);
        CHECK(result.reason == "feedforward_torque_outside_urdf_range");

        Guard combined_outside;
        state = input(); bootstrap(combined_outside, state);
        proposed = command(); proposed.kd.fill(0.);
        proposed.q[i] += sign * .1;
        proposed.tau[i] = sign * (expected_limit - 1.);
        // Individually legal feedforward plus 4 Nm P term exceeds the limit.
        result = combined_outside.filter(state, proposed, .001, true);
        is_damping(result, true);
        CHECK(result.reason == "estimated_pd_effort_outside_urdf_range");
    }
}

void two_seconds_never_rearmed() {
    Guard guard;
    auto state = input();
    is_damping(guard.filter(state, command(), .999, false), false);
    ++state.tick;
    CHECK(guard.filter(state, command(), 1., true).policy_allowed);
    for (int i = 1; i < 40; ++i) {
        ++state.tick;
        // Exit and re-enter Stairs; the first activation deadline remains 3.0.
        const bool active = i < 10 || i >= 20;
        CHECK(guard.filter(state, command(), 1. + i*.05, active).command_allowed);
    }
    ++state.tick;
    auto finished = guard.filter(state, command(), 3., true);
    is_damping(finished, true);
    CHECK(finished.reason == "two_second_test_complete");
    ++state.tick;
    is_damping(guard.filter(state, command(), 3.001, false), true);
}

void latch_time_and_missing_state() {
    Guard manual;
    manual.latch("operator_stop");
    manual.latch("later_exception");
    CHECK(manual.reason() == "operator_stop");
    is_damping(manual.filter(GuardInput{}, GuardOutput{}, 0., false), true);
    Guard clock;
    auto state = input();
    is_damping(clock.filter(state, command(), 1., false), false);
    auto result = clock.filter(state, command(), .99, false);
    CHECK(result.reason == "steady_time_regressed");
    is_damping(result, true);
    for (double now : {-1., std::numeric_limits<double>::quiet_NaN(),
                       std::numeric_limits<double>::infinity()}) {
        Guard bad_clock;
        is_damping(bad_clock.filter(input(), command(), now, false), true);
    }
}

int main() {
    freshness_and_duplicates();
    first_sample_cannot_authorize();
    tick_wrap_and_regression();
    stale_source_and_receipt_ages();
    nonfinite_state_every_array();
    nonfinite_commands_every_array();
    target_limits_and_sdk_order();
    folded_passive_is_not_position_control();
    finite_but_excessive_outputs();
    measured_speed_and_estimated_pd_effort();
    every_joint_speed_boundary();
    heterogeneous_speeds_do_not_hide_calf_violation();
    every_joint_float_wire_speed_boundary();
    every_joint_effort_limits();
    two_seconds_never_rearmed();
    latch_time_and_missing_state();
    std::cout << "PASS: 16 offline guard test groups; no hardware or networking.\n";
}
