#include "go2_mapping3d/map_utils.hpp"

#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/executors/multi_threaded_executor.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <std_srvs/srv/empty.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <unitree_go/msg/height_map.hpp>
#include <visualization_msgs/msg/marker.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace go2_mapping3d
{

struct GridKey
{
  std::int64_t x;
  std::int64_t y;

  bool operator==(const GridKey & other) const
  {
    return x == other.x && y == other.y;
  }
};

struct GridKeyHash
{
  std::size_t operator()(const GridKey & key) const
  {
    const auto hx = std::hash<std::int64_t>{} (key.x);
    const auto hy = std::hash<std::int64_t>{} (key.y);
    return hx ^ (hy + 0x9e3779b9U + (hx << 6U) + (hx >> 2U));
  }
};

struct ElevationCell
{
  double last_z{0.0};
  double mean_z{0.0};
  double min_z{std::numeric_limits<double>::infinity()};
  double max_z{-std::numeric_limits<double>::infinity()};
  double m2{0.0};
  std::uint64_t observation_count{0};
  double last_seen{0.0};

  void observe(double z, double stamp)
  {
    ++observation_count;
    const double delta = z - mean_z;
    mean_z += delta / static_cast<double>(observation_count);
    m2 += delta * (z - mean_z);
    last_z = z;
    min_z = std::min(min_z, z);
    max_z = std::max(max_z, z);
    last_seen = stamp;
  }

  double variance() const
  {
    return observation_count > 1 ? m2 / static_cast<double>(observation_count - 1) : 0.0;
  }
};

enum class TerrainClass
{
  kFlat,
  kStepCandidate,
  kRisk
};

struct TerrainMetrics
{
  TerrainClass terrain_class{TerrainClass::kRisk};
  double max_neighbor_step{0.0};
  double robust_height_span{0.0};
  std::size_t neighbor_count{0};
  bool persistent_evidence{false};
};

struct SemanticMemoryCell
{
  TerrainClass terrain_class{TerrainClass::kFlat};
  std::uint64_t last_observation_count{0};
  std::uint32_t consecutive_step_observations{0};
  std::uint32_t consecutive_danger_observations{0};
  bool confirmed{false};
  double confirmed_at{0.0};
};

const char * terrain_class_name(TerrainClass terrain_class)
{
  switch (terrain_class) {
    case TerrainClass::kFlat:
      return "flat";
    case TerrainClass::kStepCandidate:
      return "step_candidate";
    case TerrainClass::kRisk:
    default:
      return "risk";
  }
}

PointXYZRGB terrain_point(float x, float y, float z, TerrainClass terrain_class)
{
  switch (terrain_class) {
    case TerrainClass::kFlat:
      return {x, y, z, 238, 242, 245};
    case TerrainClass::kStepCandidate:
      return {x, y, z, 30, 200, 235};
    case TerrainClass::kRisk:
    default:
      return {x, y, z, 235, 55, 55};
  }
}

class GlobalElevationMapper : public rclcpp::Node
{
public:
  GlobalElevationMapper()
  : Node("global_elevation_mapper")
  {
    input_topic_ = declare_parameter<std::string>(
      "input_topic", "/utlidar/height_map_array");
    surface_cloud_topic_ = declare_parameter<std::string>(
      "surface_cloud_topic", "/utlidar/cloud_deskewed");
    surface_cloud_frame_alias_ = declare_parameter<std::string>(
      "surface_cloud_frame_alias", "");
    map_frame_ = declare_parameter<std::string>("map_frame", "utlidar_odom");
    global_resolution_ = declare_parameter<double>("global_resolution", 0.05);
    use_input_resolution_ = declare_parameter<bool>("use_input_resolution", true);
    integration_period_ = declare_parameter<double>("integration_period", 0.08);
    publish_period_ = declare_parameter<double>("publish_period", 0.20);
    minimum_observations_ = declare_parameter<int>("minimum_observations", 1);
    ambiguous_height_span_ = declare_parameter<double>("ambiguous_height_span", 0.15);
    min_accepted_height_ = declare_parameter<double>("min_accepted_height", -3.0);
    max_accepted_height_ = declare_parameter<double>("max_accepted_height", 5.0);
    terrain_min_neighbors_ = declare_parameter<int>("terrain_min_neighbors", 2);
    step_candidates_enabled_ = declare_parameter<bool>("step_candidates_enabled", false);
    terrain_neighborhood_radius_ = declare_parameter<int>("terrain_neighborhood_radius", 2);
    terrain_trim_quantile_ = declare_parameter<double>("terrain_trim_quantile", 0.05);
    ground_height_tolerance_ = declare_parameter<double>("ground_height_tolerance", 0.025);
    obstacle_min_cluster_cells_ = declare_parameter<int>("obstacle_min_cluster_cells", 2);
    obstacle_min_height_ = declare_parameter<double>("obstacle_min_height", 0.035);
    obstacle_max_height_ = declare_parameter<double>("obstacle_max_height", 0.30);
    semantic_memory_enabled_ = declare_parameter<bool>("semantic_memory_enabled", true);
    step_confirmation_observations_ =
      declare_parameter<int>("step_confirmation_observations", 3);
    danger_confirmation_observations_ =
      declare_parameter<int>("danger_confirmation_observations", 2);
    surface_smoothing_radius_ = declare_parameter<int>("surface_smoothing_radius", 1);
    surface_smoothing_height_tolerance_ =
      declare_parameter<double>("surface_smoothing_height_tolerance", 0.025);
    surface_max_triangle_height_delta_ =
      declare_parameter<double>("surface_max_triangle_height_delta", 0.035);
    surface_require_cloud_support_ =
      declare_parameter<bool>("surface_require_cloud_support", true);
    surface_cloud_height_tolerance_ =
      declare_parameter<double>("surface_cloud_height_tolerance", 0.15);
    surface_cloud_search_radius_cells_ =
      declare_parameter<int>("surface_cloud_search_radius_cells", 1);
    surface_cloud_cluster_tolerance_ =
      declare_parameter<double>("surface_cloud_cluster_tolerance", 0.025);
    surface_cloud_min_cluster_points_ =
      declare_parameter<int>("surface_cloud_min_cluster_points", 2);
    save_directory_ = declare_parameter<std::string>("save_directory", "/workspace/go2/maps");

    if (terrain_min_neighbors_ < 1 || terrain_neighborhood_radius_ < 1 ||
      terrain_trim_quantile_ < 0.0 || terrain_trim_quantile_ >= 0.5 ||
      ground_height_tolerance_ <= 0.0 || obstacle_min_cluster_cells_ < 1 ||
      obstacle_min_height_ < 0.0 || obstacle_max_height_ <= obstacle_min_height_ ||
      step_confirmation_observations_ < 1 ||
      danger_confirmation_observations_ < 1 || surface_smoothing_radius_ < 0 ||
      surface_smoothing_height_tolerance_ <= 0.0 ||
      surface_max_triangle_height_delta_ <= 0.0 ||
      surface_cloud_height_tolerance_ <= 0.0 || surface_cloud_search_radius_cells_ < 0 ||
      surface_cloud_cluster_tolerance_ <= 0.0 || surface_cloud_min_cluster_points_ < 1)
    {
      throw std::invalid_argument("invalid terrain classification parameters");
    }

    height_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    cloud_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    publish_callback_group_ = create_callback_group(rclcpp::CallbackGroupType::MutuallyExclusive);
    rclcpp::SubscriptionOptions height_subscription_options;
    height_subscription_options.callback_group = height_callback_group_;
    rclcpp::SubscriptionOptions cloud_subscription_options;
    cloud_subscription_options.callback_group = cloud_callback_group_;

    global_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/go2/terrain/elevation", rclcpp::QoS(1).transient_local());
    ambiguous_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/go2/terrain/ambiguous", rclcpp::QoS(1).transient_local());
    terrain_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/go2/terrain/classified", rclcpp::QoS(1).transient_local());
    hazard_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/go2/terrain/hazards", rclcpp::QoS(1).transient_local());
    stair_publisher_ = create_publisher<sensor_msgs::msg::PointCloud2>(
      "/go2/terrain/stairs", rclcpp::QoS(1).transient_local());
    surface_publisher_ = create_publisher<visualization_msgs::msg::Marker>(
      "/go2/terrain/surface", rclcpp::QoS(1).transient_local());
    subscription_ = create_subscription<unitree_go::msg::HeightMap>(
      input_topic_, rclcpp::SensorDataQoS(),
      std::bind(&GlobalElevationMapper::height_map_callback, this, std::placeholders::_1),
      height_subscription_options);
    surface_cloud_subscription_ = create_subscription<sensor_msgs::msg::PointCloud2>(
      surface_cloud_topic_, rclcpp::SensorDataQoS(),
      std::bind(&GlobalElevationMapper::surface_cloud_callback, this, std::placeholders::_1),
      cloud_subscription_options);
    publish_timer_ = create_wall_timer(
      std::chrono::duration<double>(std::max(0.1, publish_period_)),
      std::bind(&GlobalElevationMapper::publish_maps, this), publish_callback_group_);
    save_service_ = create_service<std_srvs::srv::Trigger>(
      "~/save", std::bind(
        &GlobalElevationMapper::save_callback, this, std::placeholders::_1,
        std::placeholders::_2));
    clear_service_ = create_service<std_srvs::srv::Empty>(
      "~/clear", std::bind(
        &GlobalElevationMapper::clear_callback, this, std::placeholders::_1,
        std::placeholders::_2));
  }

private:
  using CellMap = std::unordered_map<GridKey, ElevationCell, GridKeyHash>;

  static std::int64_t grid_index(double coordinate, double resolution)
  {
    const double scaled = coordinate / resolution;
    const double nearest_integer = std::round(scaled);
    if (std::abs(scaled - nearest_integer) < 1.0e-4) {
      return static_cast<std::int64_t>(nearest_integer);
    }
    return static_cast<std::int64_t>(std::floor(scaled));
  }

  TerrainMetrics classify_terrain(
    const GridKey & key, const ElevationCell & cell, const CellMap & cells,
    double resolution) const
  {
    TerrainMetrics metrics;
    struct TerrainSample
    {
      double x;
      double y;
      double z;
    };
    std::vector<TerrainSample> samples;
    std::vector<double> heights;
    for (std::int64_t dy = -terrain_neighborhood_radius_;
      dy <= terrain_neighborhood_radius_; ++dy)
    {
      for (std::int64_t dx = -terrain_neighborhood_radius_;
        dx <= terrain_neighborhood_radius_; ++dx)
      {
        if (dx == 0 && dy == 0) {
          samples.push_back({0.0, 0.0, cell.mean_z});
          heights.push_back(cell.mean_z);
          continue;
        }
        const auto neighbor = cells.find({key.x + dx, key.y + dy});
        if (neighbor == cells.end() ||
          neighbor->second.observation_count < static_cast<std::uint64_t>(minimum_observations_))
        {
          continue;
        }
        const double height_delta = std::abs(cell.mean_z - neighbor->second.mean_z);
        metrics.max_neighbor_step = std::max(metrics.max_neighbor_step, height_delta);
        samples.push_back(
          {static_cast<double>(dx) * resolution, static_cast<double>(dy) * resolution,
            neighbor->second.mean_z});
        heights.push_back(neighbor->second.mean_z);
        ++metrics.neighbor_count;
      }
    }

    if (metrics.neighbor_count < static_cast<std::size_t>(terrain_min_neighbors_)) {
      metrics.terrain_class = TerrainClass::kRisk;
      return metrics;
    }

    std::sort(heights.begin(), heights.end());
    const auto quantile = [&heights](double fraction) {
        const auto index = static_cast<std::size_t>(
          std::round(fraction * static_cast<double>(heights.size() - 1)));
        return heights[index];
      };
    const double low_height = quantile(terrain_trim_quantile_);
    const double high_height = quantile(1.0 - terrain_trim_quantile_);
    metrics.robust_height_span = high_height - low_height;

    if (metrics.robust_height_span > obstacle_max_height_) {
      metrics.terrain_class = TerrainClass::kRisk;
      metrics.persistent_evidence = true;
      return metrics;
    }
    std::size_t low_cluster_count = 0;
    std::size_t high_cluster_count = 0;
    bool adjacent_low = false;
    bool adjacent_high = false;
    for (const auto & sample : samples) {
      const bool low = std::abs(sample.z - low_height) <= ground_height_tolerance_;
      const bool high = std::abs(sample.z - high_height) <= ground_height_tolerance_;
      low_cluster_count += low ? 1U : 0U;
      high_cluster_count += high ? 1U : 0U;
      const bool direct_neighbor =
        (std::abs(sample.x) <= resolution * 1.01) &&
        (std::abs(sample.y) <= resolution * 1.01) &&
        (std::abs(sample.x) > 1.0e-9 || std::abs(sample.y) > 1.0e-9);
      adjacent_low = adjacent_low || (direct_neighbor && low);
      adjacent_high = adjacent_high || (direct_neighbor && high);
    }
    const bool center_low = std::abs(cell.mean_z - low_height) <= ground_height_tolerance_;
    const bool center_high = std::abs(cell.mean_z - high_height) <= ground_height_tolerance_;
    const bool supported_obstacle_edge =
      low_cluster_count >= static_cast<std::size_t>(obstacle_min_cluster_cells_) &&
      high_cluster_count >= static_cast<std::size_t>(obstacle_min_cluster_cells_) &&
      ((center_low && adjacent_high) || (center_high && adjacent_low));
    if (step_candidates_enabled_ && metrics.robust_height_span >= obstacle_min_height_ &&
      supported_obstacle_edge)
    {
      metrics.terrain_class = TerrainClass::kStepCandidate;
      metrics.persistent_evidence = true;
      return metrics;
    }
    if (metrics.robust_height_span >= obstacle_min_height_) {
      metrics.terrain_class = TerrainClass::kRisk;
      // Unsupported single-cell height changes remain visible as risk, but only a
      // spatially supported upper/lower cluster is retained in the costmap feed.
      metrics.persistent_evidence = supported_obstacle_edge;
      return metrics;
    }
    metrics.terrain_class = TerrainClass::kFlat;
    return metrics;
  }

  void height_map_callback(const unitree_go::msg::HeightMap::SharedPtr msg)
  {
    const std::size_t expected =
      static_cast<std::size_t>(msg->width) * static_cast<std::size_t>(msg->height);
    if (msg->resolution <= 0.0F || msg->width == 0 || msg->height == 0 ||
      msg->data.size() != expected)
    {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000, "Ignoring invalid HeightMap");
      return;
    }

    const auto callback_time = now();
    const double resolution = use_input_resolution_ ? msg->resolution : global_resolution_;
    if (!(resolution > 0.0)) {
      return;
    }
    const double stamp = std::isfinite(msg->stamp) ? msg->stamp : callback_time.seconds();

    std::lock_guard<std::mutex> lock(map_mutex_);
    if (last_integration_time_.nanoseconds() != 0 &&
      (callback_time - last_integration_time_).seconds() < integration_period_)
    {
      return;
    }
    last_integration_time_ = callback_time;
    if (use_input_resolution_ && active_resolution_ == 0.0) {
      active_resolution_ = resolution;
    }
    const double cell_resolution = use_input_resolution_ ? active_resolution_ : global_resolution_;
    for (std::uint32_t iy = 0; iy < msg->height; ++iy) {
      for (std::uint32_t ix = 0; ix < msg->width; ++ix) {
        const auto index = static_cast<std::size_t>(msg->width) * iy + ix;
        const double z = msg->data[index];
        if (!std::isfinite(z) || z < min_accepted_height_ || z > max_accepted_height_) {
          continue;
        }
        const double x = msg->origin[0] + static_cast<double>(ix) * msg->resolution;
        const double y = msg->origin[1] + static_cast<double>(iy) * msg->resolution;
        const GridKey key{
          grid_index(x, cell_resolution), grid_index(y, cell_resolution)};
        cells_[key].observe(z, stamp);
      }
    }
  }

  struct MapSnapshot
  {
    std::vector<PointXYZI> global;
    std::vector<PointXYZI> ambiguous;
    std::vector<PointXYZRGB> terrain;
    std::vector<PointXYZRGB> hazards;
    std::vector<PointXYZRGB> stairs;
    std::vector<PointXYZRGB> mesh_vertices;
    double resolution{0.05};
  };

  TerrainClass update_semantic_memory(
    const GridKey & key, const ElevationCell & cell, const TerrainMetrics & metrics)
  {
    if (!semantic_memory_enabled_) {
      return metrics.terrain_class;
    }

    std::lock_guard<std::mutex> lock(semantic_memory_mutex_);
    auto & memory = semantic_memory_[key];
    if (cell.observation_count > memory.last_observation_count) {
      memory.last_observation_count = cell.observation_count;
      if (metrics.persistent_evidence && metrics.terrain_class == TerrainClass::kRisk) {
        ++memory.consecutive_danger_observations;
        memory.consecutive_step_observations = 0;
        if (memory.consecutive_danger_observations >=
          static_cast<std::uint32_t>(danger_confirmation_observations_))
        {
          memory.terrain_class = TerrainClass::kRisk;
          memory.confirmed = true;
          memory.confirmed_at = cell.last_seen;
        }
      } else if (
        metrics.persistent_evidence && metrics.terrain_class == TerrainClass::kStepCandidate)
      {
        ++memory.consecutive_step_observations;
        memory.consecutive_danger_observations = 0;
        if (!memory.confirmed && memory.consecutive_step_observations >=
          static_cast<std::uint32_t>(step_confirmation_observations_))
        {
          memory.terrain_class = TerrainClass::kStepCandidate;
          memory.confirmed = true;
          memory.confirmed_at = cell.last_seen;
        }
      } else {
        memory.consecutive_step_observations = 0;
        memory.consecutive_danger_observations = 0;
      }
    }

    if (!memory.confirmed) {
      return metrics.terrain_class;
    }
    if (memory.terrain_class == TerrainClass::kRisk ||
      (metrics.persistent_evidence && metrics.terrain_class == TerrainClass::kRisk))
    {
      return TerrainClass::kRisk;
    }
    return memory.terrain_class;
  }

  TerrainClass remembered_class(
    const GridKey & key, const TerrainMetrics & current, bool & confirmed) const
  {
    confirmed = false;
    if (!semantic_memory_enabled_) {
      return current.terrain_class;
    }
    std::lock_guard<std::mutex> lock(semantic_memory_mutex_);
    const auto memory = semantic_memory_.find(key);
    if (memory == semantic_memory_.end() || !memory->second.confirmed) {
      return current.terrain_class;
    }
    confirmed = true;
    if (memory->second.terrain_class == TerrainClass::kRisk ||
      (current.persistent_evidence && current.terrain_class == TerrainClass::kRisk))
    {
      return TerrainClass::kRisk;
    }
    return memory->second.terrain_class;
  }

  CellMap make_surface_cells(
    const CellMap & source, const std::vector<PointXYZI> & cloud_points,
    double resolution) const
  {
    std::unordered_map<GridKey, std::vector<float>, GridKeyHash> cloud_columns;
    cloud_columns.reserve(cloud_points.size() / 2U + 1U);
    for (const auto & point : cloud_points) {
      cloud_columns[
        {grid_index(point.x, resolution), grid_index(point.y, resolution)}].push_back(point.z);
    }

    CellMap cloud_refined;
    cloud_refined.reserve(source.size());
    for (const auto & item : source) {
      const auto & key = item.first;
      const auto & cell = item.second;
      if (cell.observation_count < static_cast<std::uint64_t>(minimum_observations_)) {
        continue;
      }
      std::vector<float> supported_heights;
      for (std::int64_t dy = -surface_cloud_search_radius_cells_;
        dy <= surface_cloud_search_radius_cells_; ++dy)
      {
        for (std::int64_t dx = -surface_cloud_search_radius_cells_;
          dx <= surface_cloud_search_radius_cells_; ++dx)
        {
          const auto column = cloud_columns.find({key.x + dx, key.y + dy});
          if (column == cloud_columns.end()) {
            continue;
          }
          for (const float z : column->second) {
            if (std::abs(static_cast<double>(z) - cell.mean_z) <=
              surface_cloud_height_tolerance_)
            {
              supported_heights.push_back(z);
            }
          }
        }
      }
      if (supported_heights.empty() && surface_require_cloud_support_) {
        continue;
      }
      auto refined = cell;
      if (!supported_heights.empty()) {
        const auto clustered_height = select_clustered_surface_height(supported_heights);
        if (!clustered_height.has_value() && surface_require_cloud_support_) {
          continue;
        }
        if (clustered_height.has_value()) {
          refined.mean_z = clustered_height.value();
        }
      }
      cloud_refined.emplace(key, refined);
    }

    CellMap surface;
    surface.reserve(cloud_refined.size());
    for (const auto & item : cloud_refined) {
      const auto & key = item.first;
      const auto & cell = item.second;
      double weighted_height = 2.0 * cell.mean_z;
      double total_weight = 2.0;
      for (std::int64_t dy = -surface_smoothing_radius_;
        dy <= surface_smoothing_radius_; ++dy)
      {
        for (std::int64_t dx = -surface_smoothing_radius_;
          dx <= surface_smoothing_radius_; ++dx)
        {
          if (dx == 0 && dy == 0) {
            continue;
          }
          const auto neighbor = cloud_refined.find({key.x + dx, key.y + dy});
          if (neighbor == cloud_refined.end() ||
            neighbor->second.observation_count <
            static_cast<std::uint64_t>(minimum_observations_) ||
            std::abs(neighbor->second.mean_z - cell.mean_z) >
            surface_smoothing_height_tolerance_)
          {
            continue;
          }
          const double weight = (dx == 0 || dy == 0) ? 1.0 : 0.70710678118;
          weighted_height += weight * neighbor->second.mean_z;
          total_weight += weight;
        }
      }
      auto smoothed = cell;
      smoothed.mean_z = weighted_height / total_weight;
      surface.emplace(key, smoothed);
    }
    return surface;
  }

  std::optional<double> select_clustered_surface_height(std::vector<float> heights) const
  {
    if (heights.empty()) {
      return std::nullopt;
    }
    std::sort(heights.begin(), heights.end());

    std::vector<double> clusters;
    std::size_t cluster_begin = 0;
    const auto finish_cluster = [&heights, &clusters, this](std::size_t begin, std::size_t end) {
        const std::size_t point_count = end - begin;
        if (point_count >= static_cast<std::size_t>(surface_cloud_min_cluster_points_)) {
          clusters.push_back(heights[begin + point_count / 2U]);
        }
      };
    for (std::size_t index = 1; index < heights.size(); ++index) {
      if (static_cast<double>(heights[index] - heights[index - 1]) >
        surface_cloud_cluster_tolerance_)
      {
        finish_cluster(cluster_begin, index);
        cluster_begin = index;
      }
    }
    finish_cluster(cluster_begin, heights.size());
    if (clusters.empty()) {
      return std::nullopt;
    }

    // The lowest spatially supported layer is the local ground reference. A
    // second supported layer 3.5 cm or more above it is retained as obstacle
    // surface instead of being averaged back into the ground.
    const double ground_height = clusters.front();
    double selected_height = ground_height;
    for (const double cluster_height : clusters) {
      const double relative_height = cluster_height - ground_height;
      if (relative_height >= obstacle_min_height_) {
        selected_height = cluster_height;
      }
    }
    return selected_height;
  }

  void surface_cloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
  {
    if (!msg->header.frame_id.empty() && msg->header.frame_id != map_frame_ &&
      msg->header.frame_id != surface_cloud_frame_alias_)
    {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "Ignoring surface cloud in frame '%s'; expected '%s'%s",
        msg->header.frame_id.c_str(), map_frame_.c_str(),
        surface_cloud_frame_alias_.empty() ? "" : " or the configured numeric-frame alias");
      return;
    }
    std::vector<PointXYZI> points;
    points.reserve(static_cast<std::size_t>(msg->width) * msg->height);
    try {
      sensor_msgs::PointCloud2ConstIterator<float> x(*msg, "x");
      sensor_msgs::PointCloud2ConstIterator<float> y(*msg, "y");
      sensor_msgs::PointCloud2ConstIterator<float> z(*msg, "z");
      for (; x != x.end(); ++x, ++y, ++z) {
        if (std::isfinite(*x) && std::isfinite(*y) && std::isfinite(*z) &&
          *z >= min_accepted_height_ && *z <= max_accepted_height_)
        {
          points.push_back({*x, *y, *z, 0.0F});
        }
      }
    } catch (const std::runtime_error & error) {
      RCLCPP_WARN_THROTTLE(
        get_logger(), *get_clock(), 5000, "Invalid surface cloud: %s", error.what());
      return;
    }
    std::lock_guard<std::mutex> lock(map_mutex_);
    surface_cloud_points_ = std::move(points);
  }

  MapSnapshot snapshot_points()
  {
    MapSnapshot snapshot;
    CellMap cell_snapshot;
    std::vector<PointXYZI> cloud_snapshot;
    double resolution;
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      cell_snapshot = cells_;
      cloud_snapshot = surface_cloud_points_;
      resolution = active_resolution_ > 0.0 ? active_resolution_ : global_resolution_;
    }
    const auto surface_cells = make_surface_cells(cell_snapshot, cloud_snapshot, resolution);
    snapshot.resolution = resolution;
    snapshot.global.reserve(surface_cells.size());
    snapshot.terrain.reserve(surface_cells.size());
    std::unordered_map<GridKey, TerrainClass, GridKeyHash> terrain_classes;
    terrain_classes.reserve(surface_cells.size());
    for (const auto & item : surface_cells) {
      const auto & key = item.first;
      const auto & cell = item.second;
      if (cell.observation_count < static_cast<std::uint64_t>(minimum_observations_)) {
        continue;
      }
      PointXYZI point{
        static_cast<float>((static_cast<double>(key.x) + 0.5) * resolution),
        static_cast<float>((static_cast<double>(key.y) + 0.5) * resolution),
        static_cast<float>(cell.mean_z), static_cast<float>(cell.observation_count)};
      snapshot.global.push_back(point);
      if (cell.max_z - cell.min_z > ambiguous_height_span_) {
        snapshot.ambiguous.push_back(point);
      }
      const auto metrics = classify_terrain(key, cell, surface_cells, resolution);
      const auto terrain_class = update_semantic_memory(key, cell, metrics);
      terrain_classes.emplace(key, terrain_class);
      snapshot.terrain.push_back(terrain_point(point.x, point.y, point.z, terrain_class));
    }

    if (semantic_memory_enabled_) {
      std::lock_guard<std::mutex> lock(semantic_memory_mutex_);
      snapshot.hazards.reserve(semantic_memory_.size());
      snapshot.stairs.reserve(semantic_memory_.size());
      for (const auto & item : semantic_memory_) {
        if (!item.second.confirmed) {
          continue;
        }
        const auto surface_cell = surface_cells.find(item.first);
        if (surface_cell == surface_cells.end()) {
          continue;
        }
        auto point = terrain_point(
          static_cast<float>((static_cast<double>(item.first.x) + 0.5) * resolution),
          static_cast<float>((static_cast<double>(item.first.y) + 0.5) * resolution),
          static_cast<float>(surface_cell->second.mean_z), item.second.terrain_class);
        if (item.second.terrain_class == TerrainClass::kStepCandidate) {
          snapshot.stairs.push_back(point);
        } else if (item.second.terrain_class == TerrainClass::kRisk) {
          snapshot.hazards.push_back(point);
        }
      }
    }

    snapshot.mesh_vertices.reserve(surface_cells.size() * 6U);
    for (const auto & item : surface_cells) {
      const auto & key = item.first;
      const auto right = surface_cells.find({key.x + 1, key.y});
      const auto upper = surface_cells.find({key.x, key.y + 1});
      const auto diagonal = surface_cells.find({key.x + 1, key.y + 1});
      if (right == surface_cells.end() || upper == surface_cells.end() ||
        diagonal == surface_cells.end())
      {
        continue;
      }
      const double min_z = std::min(
        {item.second.mean_z, right->second.mean_z, upper->second.mean_z,
          diagonal->second.mean_z});
      const double max_z = std::max(
        {item.second.mean_z, right->second.mean_z, upper->second.mean_z,
          diagonal->second.mean_z});
      if (max_z - min_z > surface_max_triangle_height_delta_) {
        continue;
      }
      const auto vertex = [&](
        const GridKey & vertex_key, const ElevationCell & vertex_cell,
        TerrainClass terrain_class) {
          return terrain_point(
            static_cast<float>((static_cast<double>(vertex_key.x) + 0.5) * resolution),
            static_cast<float>((static_cast<double>(vertex_key.y) + 0.5) * resolution),
            static_cast<float>(vertex_cell.mean_z), terrain_class);
        };
      const auto lower_left_class = terrain_classes.at(key);
      const auto lower_right_class = terrain_classes.at({key.x + 1, key.y});
      const auto upper_left_class = terrain_classes.at({key.x, key.y + 1});
      const auto upper_right_class = terrain_classes.at({key.x + 1, key.y + 1});
      const auto triangle_class = [](TerrainClass first, TerrainClass second, TerrainClass third) {
          if (first == TerrainClass::kStepCandidate ||
            second == TerrainClass::kStepCandidate || third == TerrainClass::kStepCandidate)
          {
            return TerrainClass::kStepCandidate;
          }
          return first == TerrainClass::kFlat && second == TerrainClass::kFlat &&
                 third == TerrainClass::kFlat ? TerrainClass::kFlat : TerrainClass::kRisk;
        };
      const auto lower_triangle_class = triangle_class(
        lower_left_class, lower_right_class, upper_right_class);
      const auto upper_triangle_class = triangle_class(
        lower_left_class, upper_right_class, upper_left_class);
      snapshot.mesh_vertices.insert(
        snapshot.mesh_vertices.end(),
        {
          vertex(key, item.second, lower_triangle_class),
          vertex({key.x + 1, key.y}, right->second, lower_triangle_class),
          vertex({key.x + 1, key.y + 1}, diagonal->second, lower_triangle_class),
          vertex(key, item.second, upper_triangle_class),
          vertex({key.x + 1, key.y + 1}, diagonal->second, upper_triangle_class),
          vertex({key.x, key.y + 1}, upper->second, upper_triangle_class)
        });
    }
    return snapshot;
  }

  void publish_maps()
  {
    const auto points = snapshot_points();
    const auto stamp = time_message(now());
    global_publisher_->publish(make_cloud(points.global, map_frame_, stamp, true));
    ambiguous_publisher_->publish(make_cloud(points.ambiguous, map_frame_, stamp, true));
    terrain_publisher_->publish(make_rgb_cloud(points.terrain, map_frame_, stamp));
    hazard_publisher_->publish(make_rgb_cloud(points.hazards, map_frame_, stamp));
    stair_publisher_->publish(make_rgb_cloud(points.stairs, map_frame_, stamp));

    visualization_msgs::msg::Marker surface;
    surface.header.frame_id = map_frame_;
    surface.header.stamp = stamp;
    surface.ns = "terrain_surface";
    surface.id = 0;
    surface.type = visualization_msgs::msg::Marker::TRIANGLE_LIST;
    surface.action = visualization_msgs::msg::Marker::ADD;
    surface.pose.orientation.w = 1.0;
    surface.scale.x = 1.0;
    surface.scale.y = 1.0;
    surface.scale.z = 1.0;
    surface.points.reserve(points.mesh_vertices.size());
    surface.colors.reserve(points.mesh_vertices.size());
    for (const auto & terrain : points.mesh_vertices) {
      geometry_msgs::msg::Point point;
      point.x = terrain.x;
      point.y = terrain.y;
      point.z = terrain.z;
      surface.points.push_back(point);

      std_msgs::msg::ColorRGBA color;
      color.r = static_cast<float>(terrain.r) / 255.0F;
      color.g = static_cast<float>(terrain.g) / 255.0F;
      color.b = static_cast<float>(terrain.b) / 255.0F;
      color.a = 1.0F;
      surface.colors.push_back(color);
    }
    surface_publisher_->publish(surface);
  }

  void save_callback(
    const std::shared_ptr<std_srvs::srv::Trigger::Request>,
    std::shared_ptr<std_srvs::srv::Trigger::Response> response)
  {
    std::string error;
    if (!ensure_directory(save_directory_, error)) {
      response->success = false;
      response->message = error;
      return;
    }

    CellMap snapshot;
    std::vector<PointXYZI> surface_cloud_snapshot;
    double resolution;
    {
      std::lock_guard<std::mutex> lock(map_mutex_);
      snapshot = cells_;
      surface_cloud_snapshot = surface_cloud_points_;
      resolution = active_resolution_ > 0.0 ? active_resolution_ : global_resolution_;
    }
    const auto surface_snapshot = make_surface_cells(
      snapshot, surface_cloud_snapshot, resolution);
    if (surface_snapshot.empty()) {
      response->success = false;
      response->message = "elevation map is empty; nothing to save";
      return;
    }

    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.reserve(surface_snapshot.size());
    std::ofstream csv(save_directory_ + "/elevation.csv");
    csv << "x,y,mean_z,surface_z,min_z,max_z,variance,observation_count,last_seen,ambiguous,"
      "current_terrain_class,terrain_class,semantic_memory_confirmed,max_neighbor_step,"
      "robust_height_span,neighbor_count\n";
    double min_x = std::numeric_limits<double>::infinity();
    double max_x = -std::numeric_limits<double>::infinity();
    double min_y = std::numeric_limits<double>::infinity();
    double max_y = -std::numeric_limits<double>::infinity();
    std::size_t ambiguous_count = 0;
    std::size_t flat_count = 0;
    std::size_t step_count = 0;
    std::size_t risk_count = 0;
    for (const auto & item : surface_snapshot) {
      const auto & key = item.first;
      const auto & surface_cell = item.second;
      const auto & cell = snapshot.at(key);
      const double x = (static_cast<double>(key.x) + 0.5) * resolution;
      const double y = (static_cast<double>(key.y) + 0.5) * resolution;
      const bool ambiguous = cell.max_z - cell.min_z > ambiguous_height_span_;
      const auto terrain = classify_terrain(key, surface_cell, surface_snapshot, resolution);
      bool semantic_memory_confirmed = false;
      const auto terrain_class = remembered_class(
        key, terrain, semantic_memory_confirmed);
      cloud.push_back(pcl::PointXYZ(x, y, surface_cell.mean_z));
      csv << x << ',' << y << ',' << cell.mean_z << ',' << surface_cell.mean_z << ',' <<
        cell.min_z << ',' << cell.max_z << ',' <<
        cell.variance() << ',' << cell.observation_count << ',' << cell.last_seen << ',' <<
      (ambiguous ? 1 : 0) << ',' << terrain_class_name(terrain.terrain_class) << ',' <<
        terrain_class_name(terrain_class) << ',' << (semantic_memory_confirmed ? 1 : 0) << ',' <<
        terrain.max_neighbor_step << ',' << terrain.robust_height_span << ',' <<
        terrain.neighbor_count << '\n';
      min_x = std::min(min_x, x);
      max_x = std::max(max_x, x);
      min_y = std::min(min_y, y);
      max_y = std::max(max_y, y);
      ambiguous_count += ambiguous ? 1U : 0U;
      flat_count += terrain_class == TerrainClass::kFlat ? 1U : 0U;
      step_count += terrain_class == TerrainClass::kStepCandidate ? 1U : 0U;
      risk_count += terrain_class == TerrainClass::kRisk ? 1U : 0U;
    }
    cloud.width = static_cast<std::uint32_t>(cloud.size());
    cloud.height = 1;
    cloud.is_dense = true;

    const std::string pcd_path = save_directory_ + "/elevation.pcd";
    int pcd_result = -1;
    try {
      pcd_result = pcl::io::savePCDFileBinaryCompressed(pcd_path, cloud);
    } catch (const pcl::IOException & error) {
      response->success = false;
      response->message = std::string("failed to save elevation map: ") + error.what();
      return;
    }
    if (pcd_result != 0 || !csv.good()) {
      response->success = false;
      response->message = "failed to write elevation map files";
      return;
    }
    std::ofstream metadata(save_directory_ + "/elevation_metadata.yaml");
    metadata << "frame_id: " << map_frame_ << '\n' <<
      "resolution: " << resolution << '\n' <<
      "cell_count: " << cloud.size() << '\n' <<
      "ambiguous_cell_count: " << ambiguous_count << '\n' <<
      "flat_cell_count: " << flat_count << '\n' <<
      "step_candidate_cell_count: " << step_count << '\n' <<
      "risk_cell_count: " << risk_count << '\n' <<
      "min_x: " << (cloud.empty() ? 0.0 : min_x) << '\n' <<
      "max_x: " << (cloud.empty() ? 0.0 : max_x) << '\n' <<
      "min_y: " << (cloud.empty() ? 0.0 : min_y) << '\n' <<
      "max_y: " << (cloud.empty() ? 0.0 : max_y) << '\n' <<
      "created_at: " << now().seconds() << '\n';
    response->success = metadata.good();
    response->message = response->success ?
      "saved elevation.pcd, elevation.csv and elevation_metadata.yaml" :
      "failed to write elevation metadata";
  }

  void clear_callback(
    const std::shared_ptr<std_srvs::srv::Empty::Request>,
    std::shared_ptr<std_srvs::srv::Empty::Response>)
  {
    std::lock_guard<std::mutex> lock(map_mutex_);
    cells_.clear();
    surface_cloud_points_.clear();
    {
      std::lock_guard<std::mutex> semantic_lock(semantic_memory_mutex_);
      semantic_memory_.clear();
    }
    active_resolution_ = 0.0;
    last_integration_time_ = rclcpp::Time(0, 0, get_clock()->get_clock_type());
    visualization_msgs::msg::Marker surface;
    surface.header.frame_id = map_frame_;
    surface.header.stamp = time_message(now());
    surface.ns = "terrain_surface";
    surface.id = 0;
    surface.action = visualization_msgs::msg::Marker::DELETE;
    surface_publisher_->publish(surface);
    RCLCPP_INFO(get_logger(), "Cleared global elevation map");
  }

  std::string input_topic_;
  std::string surface_cloud_topic_;
  std::string surface_cloud_frame_alias_;
  std::string map_frame_;
  std::string save_directory_;
  double global_resolution_;
  bool use_input_resolution_;
  double integration_period_;
  double publish_period_;
  int minimum_observations_;
  double ambiguous_height_span_;
  double min_accepted_height_;
  double max_accepted_height_;
  int terrain_min_neighbors_;
  bool step_candidates_enabled_;
  int terrain_neighborhood_radius_;
  double terrain_trim_quantile_;
  double ground_height_tolerance_;
  int obstacle_min_cluster_cells_;
  double obstacle_min_height_;
  double obstacle_max_height_;
  bool semantic_memory_enabled_;
  int step_confirmation_observations_;
  int danger_confirmation_observations_;
  int surface_smoothing_radius_;
  double surface_smoothing_height_tolerance_;
  double surface_max_triangle_height_delta_;
  bool surface_require_cloud_support_;
  double surface_cloud_height_tolerance_;
  int surface_cloud_search_radius_cells_;
  double surface_cloud_cluster_tolerance_;
  int surface_cloud_min_cluster_points_;
  double active_resolution_{0.0};
  rclcpp::Time last_integration_time_{0, 0, RCL_ROS_TIME};

  mutable std::mutex map_mutex_;
  mutable std::mutex semantic_memory_mutex_;
  CellMap cells_;
  std::unordered_map<GridKey, SemanticMemoryCell, GridKeyHash> semantic_memory_;
  std::vector<PointXYZI> surface_cloud_points_;
  rclcpp::Subscription<unitree_go::msg::HeightMap>::SharedPtr subscription_;
  rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr surface_cloud_subscription_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr global_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr ambiguous_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr terrain_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr hazard_publisher_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr stair_publisher_;
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr surface_publisher_;
  rclcpp::TimerBase::SharedPtr publish_timer_;
  rclcpp::CallbackGroup::SharedPtr height_callback_group_;
  rclcpp::CallbackGroup::SharedPtr cloud_callback_group_;
  rclcpp::CallbackGroup::SharedPtr publish_callback_group_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_service_;
  rclcpp::Service<std_srvs::srv::Empty>::SharedPtr clear_service_;
};

}  // namespace go2_mapping3d

int main(int argc, char ** argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<go2_mapping3d::GlobalElevationMapper>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 3U);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
