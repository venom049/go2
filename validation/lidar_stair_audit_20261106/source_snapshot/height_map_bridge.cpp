#include "go2_mapping3d/map_utils.hpp"

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <unitree_go/msg/height_map.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

namespace go2_mapping3d
{

class HeightMapBridge : public rclcpp::Node
{
public:
  HeightMapBridge()
  : Node("height_map_bridge")
  {
    input_topic_ = declare_parameter<std::string>(
      "input_topic", "/utlidar/height_map_array");
    output_topic_ = declare_parameter<std::string>(
      "output_topic", "/go2/terrain/local_elevation");
    output_frame_ = declare_parameter<std::string>("output_frame", "utlidar_odom");
    use_message_stamp_ = declare_parameter<bool>("use_message_stamp", true);
    min_accepted_height_ = declare_parameter<double>("min_accepted_height", -3.0);
    max_accepted_height_ = declare_parameter<double>("max_accepted_height", 5.0);

    publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      output_topic_, rclcpp::SensorDataQoS());
    subscription_ = create_subscription<unitree_go::msg::HeightMap>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&HeightMapBridge::height_map_callback, this, std::placeholders::_1));

    RCLCPP_INFO(
      get_logger(), "Bridging %s to %s in frame %s", input_topic_.c_str(),
      output_topic_.c_str(), output_frame_.c_str());
  }

private:
  void height_map_callback(const unitree_go::msg::HeightMap::SharedPtr msg)
  {
    const std::size_t expected =
      static_cast<std::size_t>(msg->width) * static_cast<std::size_t>(msg->height);
    if (msg->resolution <= 0.0F || msg->width == 0 || msg->height == 0 ||
      msg->data.size() != expected)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "Invalid height map: resolution=%.4f width=%u height=%u data=%zu expected=%zu",
        msg->resolution, msg->width, msg->height, msg->data.size(), expected);
      return;
    }

    std::vector<PointXYZI> points;
    points.reserve(expected);
    std::vector<float> heights;
    heights.reserve(expected);
    std::size_t nan_count = 0;
    std::size_t inf_count = 0;
    std::size_t range_rejected_count = 0;
    std::size_t zero_count = 0;

    for (std::uint32_t iy = 0; iy < msg->height; ++iy) {
      for (std::uint32_t ix = 0; ix < msg->width; ++ix) {
        const auto index = static_cast<std::size_t>(msg->width) * iy + ix;
        const float z = msg->data[index];
        if (std::isnan(z)) {
          ++nan_count;
          continue;
        }
        if (std::isinf(z)) {
          ++inf_count;
          continue;
        }
        if (z < min_accepted_height_ || z > max_accepted_height_) {
          ++range_rejected_count;
          continue;
        }
        if (z == 0.0F) {
          ++zero_count;
        }
        points.push_back(
          {
            msg->origin[0] + static_cast<float>(ix) * msg->resolution,
            msg->origin[1] + static_cast<float>(iy) * msg->resolution,
            z, 0.0F});
        heights.push_back(z);
      }
    }

    auto stamp = use_message_stamp_ ? stamp_from_seconds(msg->stamp) : time_message(now());
    if (stamp.sec == 0 && stamp.nanosec == 0) {
      stamp = time_message(now());
    }
    publisher_->publish(make_cloud(points, output_frame_, stamp));

    if (!heights.empty()) {
      const auto minmax = std::minmax_element(heights.begin(), heights.end());
      const double mean = std::accumulate(heights.begin(), heights.end(), 0.0) /
        static_cast<double>(heights.size());
      auto middle = heights.begin() + static_cast<std::ptrdiff_t>(heights.size() / 2);
      std::nth_element(heights.begin(), middle, heights.end());
      const double median = *middle;
      RCLCPP_INFO_THROTTLE(
        get_logger(), *get_clock(), 5000,
        "HeightMap %ux%u res=%.3f valid=%zu/%zu (%.1f%%) nan=%zu inf=%zu zero=%zu "
        "range_rejected=%zu min=%.3f max=%.3f mean=%.3f median=%.3f source_frame=%s",
        msg->width, msg->height, msg->resolution, points.size(), expected,
        100.0 * static_cast<double>(points.size()) / static_cast<double>(expected),
        nan_count, inf_count, zero_count, range_rejected_count, *minmax.first, *minmax.second,
        mean, median, msg->frame_id.c_str());
    }
  }

  std::string input_topic_;
  std::string output_topic_;
  std::string output_frame_;
  bool use_message_stamp_;
  double min_accepted_height_;
  double max_accepted_height_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
  rclcpp::Subscription<unitree_go::msg::HeightMap>::SharedPtr subscription_;
};

}  // namespace go2_mapping3d

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<go2_mapping3d::HeightMapBridge>());
  rclcpp::shutdown();
  return 0;
}
