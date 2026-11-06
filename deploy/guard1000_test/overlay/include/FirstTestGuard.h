#pragma once

// Independent publication-boundary guard. No SDK, ROS, networking or actuator API.
// Published official limits are centralized in Go2JointLimits.h. They do not
// establish calibrated mechanical travel, safe contact, or hardware readiness.
// SDK order is FR, FL, RR, RL, each hip/thigh/calf. Never silently clamp and keep
// executing a policy after a violation. A latch requires a new Guard/process;
// there is deliberately no reset/rearm method.

#include "Go2JointLimits.h"
#include <array>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace go2_first_test {

constexpr std::size_t kJointCount = go2_joint_limits::kJointCount;
using JointArray = go2_joint_limits::JointArray;
inline constexpr auto kSdkJointNames = go2_joint_limits::kSdkJointNames;
inline constexpr auto kTargetLower = go2_joint_limits::kPositionLower;
inline constexpr auto kTargetUpper = go2_joint_limits::kPositionUpper;
inline constexpr auto kEffortUpper = go2_joint_limits::kEffortUpper;
inline constexpr auto kVelocityUpper = go2_joint_limits::kVelocityUpper;
inline constexpr double kMaximumAgeSeconds = 0.1;
inline constexpr double kMaximumActiveSeconds = 2.0;
// Gain ceilings preserve the existing FixStand 60/80 and 5/4 presets. These are
// software ceilings, not experimentally established safety recommendations.
inline constexpr double kMaximumKp = 80.0;
inline constexpr double kMaximumKd = 5.0;

struct GuardInput {
    bool state_available = false;
    std::array<double, 4> quaternion_wxyz{};
    std::array<double, 3> gyro{};
    std::array<double, 3> accel{};
    JointArray q{};
    JointArray dq{};
    std::uint32_t tick = 0;
    // Use nullopt when unavailable, especially for an unsynchronized device
    // clock. Never substitute host-now minus device tick as an absolute age.
    // A receipt age must originate at the actual receive callback, not polling.
    std::optional<double> source_age_s;
    std::optional<double> receipt_age_s;
};

struct GuardOutput {
    JointArray q{};
    JointArray dq{};
    JointArray kp{};
    JointArray kd{};
    JointArray tau{};
};

struct GuardResult {
    GuardOutput command;
    bool latched = false;
    bool policy_allowed = false;   // True only for accepted active Stairs output.
    bool command_allowed = false;  // True for accepted non-passive output.
    std::string reason;
};

class Guard {
public:
    Guard() = default;
    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;

    // Configure once, before any filter call. Never lengthen the established
    // 2-second ceiling, and never permit changing/rearming a started test.
    bool configure_active_seconds(double seconds) noexcept {
        if (deadline_configured_ || last_call_s_ || active_started_s_ || latched_ ||
            !(seconds == 0.5 || seconds == 1.0 || seconds == kMaximumActiveSeconds))
            return false;
        maximum_active_seconds_ = seconds;
        deadline_configured_ = true;
        return true;
    }

    static GuardOutput damping() noexcept {
        GuardOutput output;
        output.kd.fill(3.0);
        return output;  // Fixed q=0, dq=0, kp=0, kd=3, tau=0; no measured hold.
    }

    bool latched() const noexcept { return latched_; }
    const std::string& reason() const noexcept { return reason_; }

    void latch(std::string reason) {
        if (!latched_) {
            latched_ = true;
            reason_ = reason.empty() ? "external_stop" : std::move(reason);
        }
    }

    // Call immediately before each publication with the same input/output
    // snapshot. now_s is steady/monotonic seconds. active means Stairs state,
    // not FixStand. 1 kHz publication may reuse a 500 Hz state for <=100 ms.
    GuardResult filter(const GuardInput& input, const GuardOutput& proposed,
                       double now_s, bool active) {
        if (latched_) return fallback();
        if (!std::isfinite(now_s) || now_s < 0.0) return fail("invalid_steady_time");
        if (last_call_s_ && now_s < *last_call_s_) return fail("steady_time_regressed");
        last_call_s_ = now_s;

        if (active && !active_started_s_) active_started_s_ = now_s;
        // This deadline cannot be restarted by toggling states. Once armed, it
        // continues through any state exit and ends this entire first test.
        if (active_started_s_ && now_s >= *active_started_s_ + maximum_active_seconds_)
            return fail(maximum_active_seconds_ == 2.0 ? "two_second_test_complete" :
                        "bounded_test_complete");

        if (!input.state_available) return fail("state_missing");
        if (!finite(input.quaternion_wxyz) || !finite(input.gyro) || !finite(input.accel)
            || !finite(input.q) || !finite(input.dq)) return fail("nonfinite_state");
        double norm = 0.0;
        for (double value : input.quaternion_wxyz) norm = std::hypot(norm, value);
        if (!std::isfinite(norm) || std::abs(norm - 1.0) > 0.05)
            return fail("invalid_imu_quaternion_norm");
        if (input.source_age_s && !valid_age(*input.source_age_s))
            return fail("invalid_or_stale_source_age");
        if (input.receipt_age_s && !valid_age(*input.receipt_age_s))
            return fail("invalid_or_stale_receipt_age");

        if (!last_tick_) {
            last_tick_ = input.tick;
            last_tick_advance_s_ = now_s;
        } else {
            const std::uint32_t delta = input.tick - *last_tick_;
            if (delta >= UINT32_C(0x80000000)) return fail("source_tick_regressed");
            if (delta != 0) {
                // Check staleness BEFORE accepting a late resumed source.
                if (now_s - last_tick_advance_s_ > kMaximumAgeSeconds)
                    return fail("source_tick_stalled");
                last_tick_ = input.tick;
                last_tick_advance_s_ = now_s;
                observed_tick_advance_ = true;
            }
        }
        if (now_s - last_tick_advance_s_ > kMaximumAgeSeconds)
            return fail("source_tick_stalled");

        if (!finite(proposed.q) || !finite(proposed.dq) || !finite(proposed.kp)
            || !finite(proposed.kd) || !finite(proposed.tau)) return fail("nonfinite_command");
        const auto target_speed = go2_joint_limits::check_speed(proposed.dq);
        if (!target_speed.valid || target_speed.exceeded)
            return fail("target_speed_outside_urdf_range");
        bool passive = true;
        for (std::size_t i = 0; i < kJointCount; ++i) {
            if (proposed.kp[i] < 0.0 || proposed.kp[i] > kMaximumKp
                || proposed.kd[i] < 0.0 || proposed.kd[i] > kMaximumKd)
                return fail("gain_outside_software_range");
            if (std::abs(proposed.tau[i]) > kEffortUpper[i])
                return fail("feedforward_torque_outside_urdf_range");
            passive = passive && proposed.kp[i] == 0.0
                      && proposed.dq[i] == 0.0 && proposed.tau[i] == 0.0;
        }
        if (!passive) {
            const auto measured_speed = go2_joint_limits::check_speed(input.dq);
            if (!measured_speed.valid || measured_speed.exceeded)
                return fail("measured_speed_outside_urdf_range");
            for (std::size_t i = 0; i < kJointCount; ++i) {
                if (proposed.q[i] < kTargetLower[i] || proposed.q[i] > kTargetUpper[i])
                    return fail("target_position_outside_urdf_range");
                // Folded measurements may legitimately exceed this model's
                // limits. Passive does not diagnose that as hardware failure;
                // requesting position control from such a pose is refused.
                if (input.q[i] < kTargetLower[i] || input.q[i] > kTargetUpper[i])
                    return fail("initial_position_outside_software_range");
                // Snapshot-based software estimate only. This does not measure
                // or guarantee actual total motor torque: delays, drive-side
                // dynamics, contact, calibration and control implementation may
                // differ. Reject the whole command; do not clip and continue.
                const double estimated_torque =
                    proposed.kp[i] * (proposed.q[i] - input.q[i])
                    + proposed.kd[i] * (proposed.dq[i] - input.dq[i])
                    + proposed.tau[i];
                if (!std::isfinite(estimated_torque)
                    || std::abs(estimated_torque) > kEffortUpper[i])
                    return fail("estimated_pd_effort_outside_urdf_range");
            }
        }
        if (!observed_tick_advance_) return fallback("waiting_for_source_tick_advance");
        if (passive) return fallback("passive_fixed_damping");
        return {proposed, false, active, true, "accepted"};
    }

private:
    template <std::size_t N> static bool finite(const std::array<double, N>& values) noexcept {
        for (double value : values) if (!std::isfinite(value)) return false;
        return true;
    }
    static bool valid_age(double age) noexcept {
        return std::isfinite(age) && age >= 0.0 && age <= kMaximumAgeSeconds;
    }
    GuardResult fallback(const char* waiting_reason = "") const {
        return {damping(), latched_, false, false, latched_ ? reason_ : waiting_reason};
    }
    GuardResult fail(const char* reason) { latch(reason); return fallback(); }

    bool latched_ = false;
    bool deadline_configured_ = false;
    double maximum_active_seconds_ = kMaximumActiveSeconds;
    bool observed_tick_advance_ = false;
    std::string reason_;
    std::optional<double> last_call_s_;
    std::optional<double> active_started_s_;
    std::optional<std::uint32_t> last_tick_;
    double last_tick_advance_s_ = 0.0;
};

}  // namespace go2_first_test
