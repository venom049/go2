// Receive-only LowState -> IMU. DDS SOURCE time is retained; never receipt-retimed.
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <unitree_go/msg/low_state.hpp>
#include <std_msgs/msg/string.hpp>
#include <cmath>
#include <sstream>
#include <iomanip>
class BodyImuBridge : public rclcpp::Node {
public:
 BodyImuBridge():Node("moe_body_imu_bridge") {
  pub_=create_publisher<sensor_msgs::msg::Imu>("/moe_lio/body_imu",rclcpp::SensorDataQoS());
  health_=create_publisher<std_msgs::msg::String>("/moe_lio/imu_health",10);
  sub_=create_subscription<unitree_go::msg::LowState>("/lowstate",rclcpp::SensorDataQoS(),
   [this](unitree_go::msg::LowState::SharedPtr m,const rclcpp::MessageInfo& info){receive(*m,info);});
 }
 ~BodyImuBridge(){RCLCPP_INFO(get_logger(),"IMU bridge exit accepted=%lu rejected=%lu; no motor publisher",accepted_,rejected_);}
private:
 void receive(const unitree_go::msg::LowState& m,const rclcpp::MessageInfo& info){
  const auto source=info.get_rmw_message_info().source_timestamp;
  const double age=(get_clock()->now().nanoseconds()-source)*1e-9;
  if(m.tick==last_tick_)return;
  double norm=0.;for(float x:m.imu_state.quaternion)norm+=double(x)*x;
  bool valid=source>0 && source>last_stamp_ && age>=-.03 && age<=.25 && std::abs(norm-1.)<.02;
  for(float x:m.imu_state.gyroscope)valid=valid && std::isfinite(x);
  for(float x:m.imu_state.accelerometer)valid=valid && std::isfinite(x);
  if(last_stamp_ && static_cast<std::uint32_t>(m.tick-last_tick_)>1000)valid=false;
  if(!valid){++rejected_;if(rejected_%250==1)RCLCPP_WARN(get_logger(),"IMU source rejected source_ns=%ld age_s=%.6f",source,age);return;}
  last_tick_=m.tick;last_stamp_=source;++accepted_;
  sensor_msgs::msg::Imu out;out.header.stamp=rclcpp::Time(source,RCL_SYSTEM_TIME);out.header.frame_id="imu";
  out.orientation.w=m.imu_state.quaternion[0];out.orientation.x=m.imu_state.quaternion[1];out.orientation.y=m.imu_state.quaternion[2];out.orientation.z=m.imu_state.quaternion[3];
  out.angular_velocity.x=m.imu_state.gyroscope[0];out.angular_velocity.y=m.imu_state.gyroscope[1];out.angular_velocity.z=m.imu_state.gyroscope[2];
  out.linear_acceleration.x=m.imu_state.accelerometer[0];out.linear_acceleration.y=m.imu_state.accelerometer[1];out.linear_acceleration.z=m.imu_state.accelerometer[2];pub_->publish(out);
  if(accepted_%50==1){std_msgs::msg::String h;std::ostringstream s;s<<std::setprecision(17)<<"{\"source_stamp\":"<<source*1e-9<<",\"age_s\":"<<age<<",\"tick\":"<<m.tick<<",\"accepted\":"<<accepted_<<",\"rejected\":"<<rejected_<<"}";h.data=s.str();health_->publish(h);}
 }
 std::uint32_t last_tick_=0;std::int64_t last_stamp_=0;std::uint64_t accepted_=0,rejected_=0;
 rclcpp::Publisher<sensor_msgs::msg::Imu>::SharedPtr pub_;rclcpp::Publisher<std_msgs::msg::String>::SharedPtr health_;rclcpp::Subscription<unitree_go::msg::LowState>::SharedPtr sub_;
};
int main(int argc,char**argv){rclcpp::init(argc,argv);rclcpp::spin(std::make_shared<BodyImuBridge>());rclcpp::shutdown();return 0;}
