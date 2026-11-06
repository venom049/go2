// Offline C++17 tests: no SDK, ROS, sockets, threads or robot access.
// Compile with the include directory of each deploy overlay to verify its copy.
#include "Go2JointLimits.h"

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

using namespace go2_joint_limits;

#define CHECK(condition) do { if (!(condition)) { \
    std::cerr << "FAILED line " << __LINE__ << ": " << #condition << '\n'; \
    std::exit(1); } } while (false)

void documented_profile_and_sdk_order() {
    const std::array<const char*, 12> expected_names = {
        "FR_hip", "FR_thigh", "FR_calf", "FL_hip", "FL_thigh", "FL_calf",
        "RR_hip", "RR_thigh", "RR_calf", "RL_hip", "RL_thigh", "RL_calf"};
    const double hip_limit = 48. * std::acos(-1.) / 180.;
    CHECK(kJointCount == 12);
    CHECK(std::string(kProfileId) == "unitree_go2_2ded279_hip48");
    for (std::size_t i = 0; i < kJointCount; ++i) {
        CHECK(std::string(kSdkJointNames[i]) == expected_names[i]);
        const double speed_nominal = i % 3 == 2 ? 15.7 : 30.1;
        CHECK(kVelocityUpperNominal[i] == speed_nominal);
        CHECK(kVelocityUpper[i] == static_cast<double>(static_cast<float>(speed_nominal)));
        CHECK(kEffortUpper[i] == (i % 3 == 2 ? 45.43 : 23.7));
        if (i % 3 == 0) {
            CHECK(std::abs(kPositionLowerNominal[i] + hip_limit) < 1e-12);
            CHECK(std::abs(kPositionUpperNominal[i] - hip_limit) < 1e-12);
        } else if (i % 3 == 1) {
            // SDK +Y right-hand q convention; front and rear ranges differ.
            CHECK(kPositionLowerNominal[i] == (i < 6 ? -1.5708 : -.5236));
            CHECK(kPositionUpperNominal[i] == (i < 6 ? 3.4907 : 4.5379));
        } else {
            CHECK(kPositionLowerNominal[i] == -2.7227);
            CHECK(kPositionUpperNominal[i] == -.83776);
        }
        CHECK(kPositionLower[i] == static_cast<double>(static_cast<float>(kPositionLowerNominal[i])));
        CHECK(kPositionUpper[i] == static_cast<double>(static_cast<float>(kPositionUpperNominal[i])));
    }
}

void every_positive_and_negative_speed_boundary() {
    JointArray stationary{};
    auto zero = check_speed(stationary);
    CHECK(zero.valid && !zero.exceeded);
    CHECK(zero.abs_speed == 0. && zero.ratio == 0.);
    for (std::size_t i = 0; i < kJointCount; ++i) for (double sign : {-1., 1.}) {
        JointArray speeds{};
        const double expected_limit = static_cast<double>(static_cast<float>(i % 3 == 2 ? 15.7 : 30.1));
        for (double epsilon : {-1e-6, 0., 1e-6}) {
            speeds[i] = sign * (expected_limit + epsilon);
            const auto result = check_speed(speeds);
            CHECK(result.valid);
            CHECK(result.exceeded == (epsilon > 0.));
            CHECK(result.joint_index == i);
            CHECK(result.limit == expected_limit);
            CHECK(result.abs_speed == std::abs(speeds[i]));
            CHECK(result.ratio == std::abs(speeds[i]) / expected_limit);
            CHECK(speeds[i] == sign * (expected_limit + epsilon));
        }
    }
}

void select_by_normalized_ratio_not_absolute_speed() {
    JointArray speeds{};
    speeds[0] = 20.;
    speeds[11] = -16.;
    const auto result = check_speed(speeds);
    CHECK(result.valid && result.exceeded);
    CHECK(result.joint_index == 11);
    CHECK(result.abs_speed == 16. && result.limit == static_cast<double>(15.7f));
    CHECK(result.ratio > 1.);
    speeds.fill(0.);
    speeds[11] = -13.312929153442383;
    const auto historical = check_speed(speeds);
    CHECK(historical.valid && !historical.exceeded);
    CHECK(historical.joint_index == 11);
}

void float_wire_speed_boundaries() {
    for (std::size_t i = 0; i < kJointCount; ++i) for (double sign : {-1., 1.}) {
        const float wire_limit = static_cast<float>(i % 3 == 2 ? 15.7 : 30.1);
        JointArray speeds{};
        speeds[i] = sign * static_cast<double>(wire_limit);
        const auto at_wire_boundary = check_speed(speeds);
        CHECK(at_wire_boundary.valid && !at_wire_boundary.exceeded);
        speeds[i] = sign * static_cast<double>(
            std::nextafter(wire_limit, std::numeric_limits<float>::infinity()));
        const auto beyond_wire_boundary = check_speed(speeds);
        CHECK(beyond_wire_boundary.valid && beyond_wire_boundary.exceeded);
        CHECK(beyond_wire_boundary.joint_index == i);
    }
}

void invalid_input_never_counts_as_within_limits() {
    for (std::size_t count : {0u, 1u, 11u, 13u, 20u}) {
        const auto result = check_speed(std::vector<double>(count, 0.));
        CHECK(!result.valid);
        CHECK(std::string(result.reason) == "invalid_joint_count");
    }
    const auto sized = check_speed(std::vector<double>(12, 0.));
    CHECK(sized.valid && !sized.exceeded);
    for (std::size_t i = 0; i < kJointCount; ++i)
        for (double invalid : {std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity(),
                               -std::numeric_limits<double>::infinity()}) {
            JointArray speeds{};
            speeds[0] = 31.;  // An earlier speed violation must not hide invalid data.
            speeds[i] = invalid;
            const auto result = check_speed(speeds);
            CHECK(!result.valid);
            CHECK(std::string(result.reason) == "nonfinite_joint_speed");
        }
    JointArray large{};
    large[8] = -1e300;
    auto finite_large = check_speed(large);
    CHECK(finite_large.valid && finite_large.exceeded);
    CHECK(finite_large.joint_index == 8);
}

void only_exact_sdk_mapping_is_accepted() {
    std::array<int, 12> sdk_order{};
    for (std::size_t i = 0; i < sdk_order.size(); ++i) sdk_order[i] = static_cast<int>(i);
    CHECK(valid_sdk_mapping(sdk_order));
    for (std::size_t count : {0u, 11u, 13u, 20u})
        CHECK(!valid_sdk_mapping(std::vector<int>(count, 0)));
    for (std::size_t i = 0; i < kJointCount; ++i) {
        JointArray mapping{};
        for (std::size_t j = 0; j < mapping.size(); ++j) mapping[j] = static_cast<double>(j);
        for (double invalid : {-1., 12., static_cast<double>(i) + .5,
                               std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity()}) {
            mapping[i] = invalid;
            CHECK(!valid_sdk_mapping(mapping));
        }
    }
    auto swapped = sdk_order;
    std::swap(swapped[0], swapped[3]);  // Policy/URDF leg order is not SDK order.
    CHECK(!valid_sdk_mapping(swapped));
    auto duplicate = sdk_order;
    duplicate[11] = duplicate[8];
    CHECK(!valid_sdk_mapping(duplicate));
}

int main() {
    documented_profile_and_sdk_order();
    every_positive_and_negative_speed_boundary();
    select_by_normalized_ratio_not_absolute_speed();
    float_wire_speed_boundaries();
    invalid_input_never_counts_as_within_limits();
    only_exact_sdk_mapping_is_accepted();
    std::cout << "PASS: official Go2 joint profile and all 12 speed checks; offline only.\n";
}
