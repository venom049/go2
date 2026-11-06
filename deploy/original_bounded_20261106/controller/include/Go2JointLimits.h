#pragma once

// Official published specifications, not firmware readback or permission to move.
// SDK order: FR, FL, RR, RL; each hip, thigh, calf.
// Speed, effort and non-hip position bounds are pinned to Unitree's URDF:
// https://raw.githubusercontent.com/unitreerobotics/unitree_ros/2ded279cfb3207d410b1ade88303e1a8971b0ac3/robots/go2_description/urdf/go2_description.urdf
// SHA256: 7d19fe48e2e689ee1a032ab99f2a4a8b671d87e73de48d3e65811682a5b48b9e
// Hip position uses the more restrictive +/-48 degree published specification:
// https://support.unitree.com/home/zh/developer/about_Go2
// The document's anatomical angle/sign conventions must not be copied onto SDK
// motor q blindly: calf SDK/URDF q is negative; front/rear thigh bounds differ.
// Therefore retain the pinned URDF's SDK-coordinate thigh/calf bounds; intersect
// the symmetric hip range with the narrower documented range. Application
// tracking-error, stand-readiness, gains and watchdogs are separate protections.

#include <array>
#include <cmath>
#include <cstddef>

namespace go2_joint_limits {
inline constexpr std::size_t kJointCount = 12;
using JointArray = std::array<double, kJointCount>;
inline constexpr const char* kProfileId = "unitree_go2_2ded279_hip48";
inline constexpr const char* kUrdfCommit = "2ded279cfb3207d410b1ade88303e1a8971b0ac3";
inline constexpr std::array<const char*, kJointCount> kSdkJointNames = {
    "FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"};
inline constexpr JointArray kVelocityUpperNominal = {
    30.1, 30.1, 15.7, 30.1, 30.1, 15.7,
    30.1, 30.1, 15.7, 30.1, 30.1, 15.7};
inline constexpr double kHipPositionMagnitude = 0.8377580409572781;
inline constexpr JointArray kPositionLowerNominal = {
    -kHipPositionMagnitude, -1.5708, -2.7227,
    -kHipPositionMagnitude, -1.5708, -2.7227,
    -kHipPositionMagnitude, -0.5236, -2.7227,
    -kHipPositionMagnitude, -0.5236, -2.7227};
inline constexpr JointArray kPositionUpperNominal = {
    kHipPositionMagnitude, 3.4907, -0.83776,
    kHipPositionMagnitude, 3.4907, -0.83776,
    kHipPositionMagnitude, 4.5379, -0.83776,
    kHipPositionMagnitude, 4.5379, -0.83776};
// LowState and LowCmd serialize q/dq as float32. Compare against the same
// representable boundary even after a snapshot promotes the value to double.
// Keep declared decimal values separately; there is no tunable tolerance.
constexpr JointArray protocol_limits(const JointArray& declared) {
    JointArray result{};
    for (std::size_t i = 0; i < kJointCount; ++i)
        result[i] = static_cast<double>(static_cast<float>(declared[i]));
    return result;
}
inline constexpr auto kVelocityUpper = protocol_limits(kVelocityUpperNominal);
inline constexpr auto kPositionLower = protocol_limits(kPositionLowerNominal);
inline constexpr auto kPositionUpper = protocol_limits(kPositionUpperNominal);
inline constexpr JointArray kEffortUpper = {
    23.7, 23.7, 45.43, 23.7, 23.7, 45.43,
    23.7, 23.7, 45.43, 23.7, 23.7, 45.43};

template<class Container> bool valid_sdk_mapping(const Container& mapping) {
    if (static_cast<std::size_t>(mapping.size()) != kJointCount) return false;
    for (std::size_t i = 0; i < kJointCount; ++i) {
        const double value = static_cast<double>(mapping[i]);
        if (!std::isfinite(value) || value != static_cast<double>(i)) return false;
    }
    return true;
}

struct SpeedCheck {
    bool valid = false;
    bool exceeded = false;
    std::size_t joint_index = kJointCount;
    double abs_speed = 0.0;
    double limit = 0.0;
    double ratio = 0.0;
    const char* reason = "invalid_joint_count";
};

// Use the largest ratio, not the largest raw speed: a legal hip at 20 rad/s
// must never hide a calf at 16 rad/s. Reject malformed or nonfinite input even
// if a previously inspected joint has an otherwise valid maximum.
template<class Container> SpeedCheck check_speed(const Container& speeds) {
    SpeedCheck result;
    if (static_cast<std::size_t>(speeds.size()) != kJointCount) return result;
    for (std::size_t i = 0; i < kJointCount; ++i) {
        const double speed = static_cast<double>(speeds[i]);
        if (!std::isfinite(speed)) {
            result.valid = false;
            result.exceeded = false;
            result.joint_index = i;
            result.abs_speed = std::abs(speed);
            result.limit = kVelocityUpper[i];
            result.ratio = std::abs(speed) / result.limit;
            result.reason = "nonfinite_joint_speed";
            return result;
        }
        const double magnitude = std::abs(speed);
        const double ratio = magnitude / kVelocityUpper[i];
        if (i == 0 || ratio > result.ratio) {
            result.joint_index = i;
            result.abs_speed = magnitude;
            result.limit = kVelocityUpper[i];
            result.ratio = ratio;
        }
    }
    result.valid = true;
    result.exceeded = result.ratio > 1.0;
    result.reason = result.exceeded ? "joint_speed_limit_exceeded" : "accepted";
    return result;
}
}  // namespace go2_joint_limits
