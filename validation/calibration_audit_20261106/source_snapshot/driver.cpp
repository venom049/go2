#include "rclcpp/rclcpp.hpp"
// 发布里程计消息的头文件
#include "nav_msgs/msg/odometry.hpp"
// 发布坐标变换的头文件
#include "geometry_msgs/msg/transform_stamped.hpp"
#include "tf2_ros/transform_broadcaster.h"
#include "tf2_ros/static_transform_broadcaster.h"
#include "tf2/LinearMath/Quaternion.h"
#include "tf2/utils.h"
// 发布关节状态信息的头文件
#include "sensor_msgs/msg/joint_state.hpp"
#include "unitree_go/msg/low_state.hpp"
#include <cmath>
#include <optional>

#include "go2_driver/session_frame.hpp"

using namespace std::placeholders;

// 自定义节点类
class Driver : public rclcpp::Node
{
public:
  Driver()
  : Node("driver")
  {
    RCLCPP_INFO(this->get_logger(), "Driver节点创建, 用于发布里程计消息，坐标变换和关节状态信息");

    // 声明参数
    // this->declare_parameter("publish_tf", true);

    // 获取参数
    // publish_tf = this->get_parameter("publish_tf").as_bool();

    //坐标变换广播器
    tf_bro_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);
    static_tf_bro_ = std::make_unique<tf2_ros::StaticTransformBroadcaster>(*this);

    odom_frame_ = this->declare_parameter<std::string>("odom_frame", "odom");
    native_odom_frame_ =
      this->declare_parameter<std::string>("native_odom_frame", "utlidar_odom");
    base_frame_ = this->declare_parameter<std::string>("base_frame", "base_footprint");

    odom_pub_ = this->create_publisher<nav_msgs::msg::Odometry>("odom", 10);
    // The onboard Unitree lidar publishes the persistent robot pose as
    // nav_msgs/Odometry on /utlidar/robot_odom.  The old subscription to
    // geometry_msgs/PoseStamped /utlidar/robot_pose has no publisher on the
    // real robot, so it only produced one stale TF at startup.
    robot_pose_sub_ = this->create_subscription<nav_msgs::msg::Odometry>(
      "/utlidar/robot_odom", 10,
      std::bind(&Driver::pose_callback, this, std::placeholders::_1));


    //关节状态发布
    joint_state_pub_ = this->create_publisher<sensor_msgs::msg::JointState>("joint_states", 10);
    low_state_sub_ = this->create_subscription<unitree_go::msg::LowState>(
      "/lf/lowstate", 10, std::bind(
        &Driver::low_state_cb, this,
        std::placeholders::_1));
  }

private:
  // 发布关节状态信息
  rclcpp::Publisher<sensor_msgs::msg::JointState>::SharedPtr joint_state_pub_;
  // 订阅低层状态
  rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr low_state_sub_;
  // 广播坐标变换
  std::unique_ptr<tf2_ros::TransformBroadcaster> tf_bro_;
  std::unique_ptr<tf2_ros::StaticTransformBroadcaster> static_tf_bro_;
  // 订阅go2的状态
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr robot_pose_sub_;
  //发布里程计
  rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odom_pub_;
  std::optional<rclcpp::Time> previous_stamp_;
  std::optional<double> initial_base_x_;
  double initial_base_y_{0.0};
  double initial_yaw_{0.0};
  double previous_x_{0.0};
  double previous_y_{0.0};
  double previous_yaw_{0.0};
  std::string odom_frame_;
  std::string native_odom_frame_;
  std::string base_frame_;

  //订阅低层信息获取关节状态， 组织消息并发布
  void low_state_cb(const unitree_go::msg::LowState::SharedPtr low_state)
  {
    sensor_msgs::msg::JointState joint_state;

    // Unitree's motor order is FR, FL, RR, RL. Keep the joint names in the
    // same order as motor_state instead of assigning one leg's data to another.
    joint_state.header.stamp = this->now();
    joint_state.name = {
      "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
      "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
      "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
      "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"
    };

    constexpr size_t kLegJointCount = 12;
    constexpr double kMaxSaneJointPosition = 10.0;
    joint_state.position.reserve(kLegJointCount);

    for (size_t i = 0; i < kLegJointCount; ++i) {
      const double position = low_state->motor_state[i].q;
      if (!std::isfinite(position) || std::abs(position) > kMaxSaneJointPosition) {
        RCLCPP_WARN_THROTTLE(
          this->get_logger(), *this->get_clock(), 2000,
          "Ignoring invalid LowState joint frame: motor_state[%zu].q=%f", i, position);
        return;
      }

      joint_state.position.push_back(position);
    }

    joint_state_pub_->publish(joint_state);
  }

  void pose_callback(const nav_msgs::msg::Odometry::SharedPtr msg)
  {
    // The native /utlidar/robot_odom header can be delayed by more than a
    // minute on the real robot, while /trans_cloud is stamped with current
    // ROS time. Reusing that stale stamp makes TF2 reject every cloud as
    // extrapolation into the past and prevents SLAM from producing /map.
    // Navigation TF and odometry must share the current ROS clock with the
    // cloud pipeline, so timestamp this derived message at receipt time.
    const rclcpp::Time stamp = this->now();

    tf2::Quaternion orientation(
      msg->pose.pose.orientation.x, msg->pose.pose.orientation.y,
      msg->pose.pose.orientation.z, msg->pose.pose.orientation.w);
    if (orientation.length2() < 1e-6) {
      orientation.setRPY(0.0, 0.0, 0.0);
    } else {
      orientation.normalize();
    }
    const double yaw = tf2::getYaw(orientation);
    tf2::Quaternion planar_orientation;
    planar_orientation.setRPY(0.0, 0.0, yaw);

    // Unitree publishes both robot_pose and deskewed clouds in its persistent
    // world frame, named "odom" on the wire. Keep that native frame distinct
    // from the fresh navigation-session odom frame.
    constexpr double lidar_offset_x = 0.28945;
    const go2_driver::PlanarPose native_base = go2_driver::base_pose_from_lidar(
      {msg->pose.pose.position.x, msg->pose.pose.position.y, yaw}, lidar_offset_x);

    if (!initial_base_x_.has_value()) {
      initial_base_x_ = native_base.x;
      initial_base_y_ = native_base.y;
      initial_yaw_ = native_base.yaw;

      const auto native_frame = go2_driver::native_frame_in_session(native_base);
      geometry_msgs::msg::TransformStamped session_to_native;
      session_to_native.header.stamp = stamp;
      session_to_native.header.frame_id = odom_frame_;
      session_to_native.child_frame_id = native_odom_frame_;
      session_to_native.transform.translation.x = native_frame.x;
      session_to_native.transform.translation.y = native_frame.y;
      session_to_native.transform.translation.z = 0.0;
      tf2::Quaternion session_to_native_orientation;
      session_to_native_orientation.setRPY(0.0, 0.0, native_frame.yaw);
      session_to_native.transform.rotation.x = session_to_native_orientation.x();
      session_to_native.transform.rotation.y = session_to_native_orientation.y();
      session_to_native.transform.rotation.z = session_to_native_orientation.z();
      session_to_native.transform.rotation.w = session_to_native_orientation.w();
      static_tf_bro_->sendTransform(session_to_native);

      RCLCPP_INFO(
        this->get_logger(),
        "Started navigation odom from native lidar frame: base=(%.3f, %.3f, %.3f rad)",
        native_base.x, native_base.y, native_base.yaw);
    }

    const go2_driver::PlanarPose initial_native_base = {
      initial_base_x_.value(), initial_base_y_, initial_yaw_};
    const auto session_base =
      go2_driver::native_pose_in_session(native_base, initial_native_base);
    const double base_x = session_base.x;
    const double base_y = session_base.y;
    const double relative_yaw = session_base.yaw;
    planar_orientation.setRPY(0.0, 0.0, relative_yaw);

    // Publish the native pose, not a second direct odom->base edge. The TF
    // chain then transforms native world-frame clouds and robot motion through
    // the same session transform: odom -> utlidar_odom -> base_footprint.
    geometry_msgs::msg::TransformStamped transform;
    transform.header.stamp = stamp;
    transform.header.frame_id = native_odom_frame_;
    transform.child_frame_id = base_frame_;
    transform.transform.translation.x = native_base.x;
    transform.transform.translation.y = native_base.y;
    transform.transform.translation.z = 0.0;
    tf2::Quaternion native_orientation;
    native_orientation.setRPY(0.0, 0.0, native_base.yaw);
    transform.transform.rotation.x = native_orientation.x();
    transform.transform.rotation.y = native_orientation.y();
    transform.transform.rotation.z = native_orientation.z();
    transform.transform.rotation.w = native_orientation.w();
    tf_bro_->sendTransform(transform);

    nav_msgs::msg::Odometry odom;
    odom.header.stamp = stamp;
    odom.header.frame_id = odom_frame_;
    odom.child_frame_id = base_frame_;
    odom.pose.pose.position.x = base_x;
    odom.pose.pose.position.y = base_y;
    odom.pose.pose.position.z = 0.0;
    odom.pose.pose.orientation.x = planar_orientation.x();
    odom.pose.pose.orientation.y = planar_orientation.y();
    odom.pose.pose.orientation.z = planar_orientation.z();
    odom.pose.pose.orientation.w = planar_orientation.w();
    odom.pose.covariance[0] = 0.02;
    odom.pose.covariance[7] = 0.02;
    odom.pose.covariance[35] = 0.04;

    if (previous_stamp_.has_value()) {
      const double dt = (stamp - previous_stamp_.value()).seconds();
      if (dt > 0.001 && dt < 1.0) {
        const double vx_odom = (base_x - previous_x_) / dt;
        const double vy_odom = (base_y - previous_y_) / dt;
        odom.twist.twist.linear.x =
          std::cos(relative_yaw) * vx_odom + std::sin(relative_yaw) * vy_odom;
        odom.twist.twist.linear.y =
          -std::sin(relative_yaw) * vx_odom + std::cos(relative_yaw) * vy_odom;
        odom.twist.twist.angular.z =
          std::remainder(relative_yaw - previous_yaw_, 2.0 * M_PI) / dt;
      }
    }
    odom.twist.covariance[0] = 0.05;
    odom.twist.covariance[7] = 0.05;
    odom.twist.covariance[35] = 0.08;
    odom_pub_->publish(odom);

    previous_stamp_ = stamp;
    previous_x_ = base_x;
    previous_y_ = base_y;
    previous_yaw_ = relative_yaw;
  }
};

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<Driver>());
  rclcpp::shutdown();
  return 0;
}
