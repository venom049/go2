#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_StairsReady.h"
#include "State_LieDown.h"
#include "unitree_articulation.h"
#include <unitree/robot/go2/robot_state/robot_state_client.hpp>
#include <unitree/robot/go2/robot_state/robot_state_api.hpp>
#include <csignal>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <thread>

namespace
{
volatile std::sig_atomic_t keep_running = 1;

class RawRobotStateClient : public unitree::robot::Client
{
public:
    RawRobotStateClient()
        : unitree::robot::Client(unitree::robot::go2::ROBOT_STATE_SERVICE_NAME)
    {
        SetApiVersion(unitree::robot::go2::ROBOT_STATE_API_VERSION);
    }

    void Init() override
    {
        RegistApi(unitree::robot::go2::ROBOT_STATE_API_ID_SERVICE_LIST, 0);
    }

    int32_t ServiceListRaw(std::string& data)
    {
        return Call(unitree::robot::go2::ROBOT_STATE_API_ID_SERVICE_LIST, "{}", data);
    }
};

enum class InitResult
{
    Ready,
    LowCmdBusy,
    CheckComplete,
};

void handle_signal(int)
{
    keep_running = 0;
}

int release_motion_mode()
{
    unitree::robot::b2::MotionSwitcherClient motion_switcher;
    motion_switcher.SetTimeout(3.0f);
    motion_switcher.Init();

    std::string robot_form;
    std::string motion_name;
    int32_t result = motion_switcher.CheckMode(robot_form, motion_name);
    if(result != 0)
    {
        spdlog::error("MOTION_MODE_QUERY_FAILED={}", result);
        return 3;
    }
    if(motion_name.empty())
    {
        spdlog::info("MOTION_MODE=RELEASED (already released)");
        spdlog::info("No LowCmd or motor commands were published.");
        return 0;
    }

    spdlog::warn(
        "Releasing motion service: form={} name={}", robot_form, motion_name
    );
    result = motion_switcher.ReleaseMode();
    if(result != 0)
    {
        spdlog::error("MOTION_MODE_RELEASE_FAILED={}", result);
        return 4;
    }

    usleep(500 * 1000);
    robot_form.clear();
    motion_name.clear();
    result = motion_switcher.CheckMode(robot_form, motion_name);
    if(result != 0)
    {
        spdlog::error("MOTION_MODE_VERIFY_FAILED={}", result);
        return 5;
    }
    if(!motion_name.empty())
    {
        spdlog::error(
            "MOTION_MODE_RELEASE_INCOMPLETE form={} name={}",
            robot_form,
            motion_name
        );
        return 6;
    }

    spdlog::info("MOTION_MODE=RELEASED");
    spdlog::info("No LowCmd or motor commands were published.");
    return 0;
}

void override_stairs_speed(YAML::Node cfg, float speed)
{
    if(speed <= 0.0f)
    {
        throw std::runtime_error("--stairs-speed must be positive.");
    }
    auto obs = cfg["observations"]["keyboard_stairs_velocity_commands"];
    if(!obs)
    {
        throw std::runtime_error(
            "Policy deploy.yaml has no keyboard_stairs_velocity_commands observation."
        );
    }
    obs["params"]["min_forward_speed"] = speed;
    obs["params"]["max_forward_speed"] = speed;
    obs["params"]["initial_forward_speed"] = speed;
}

int robot_print(const boost::program_options::variables_map& vm)
{
    const int steps = vm["robot-print-steps"].as<int>();
    if(steps <= 0)
    {
        spdlog::error("--robot-print-steps must be positive.");
        return 1;
    }

    auto cfg = param::config["FSM"]["Stairs"];
    auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());
    auto deploy_cfg = YAML::LoadFile(policy_dir / "params" / "deploy.yaml");
    if(vm.count("stairs-speed") > 0)
    {
        override_stairs_speed(deploy_cfg, vm["stairs-speed"].as<float>());
    }

    auto lowstate = std::make_shared<LowState_t>();
    lowstate->wait_for_connection();
    lowstate->update();
    auto env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
        deploy_cfg,
        std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(
            lowstate
        )
    );
    env->alg = std::make_unique<isaaclab::OrtRunner>(
        policy_dir / "exported" / "policy.onnx"
    );

    env->reset();
    spdlog::info(
        "ROBOT_PRINT started: steps={} speed={:.2f} m/s. No LowCmd publisher was created.",
        steps,
        vm.count("stairs-speed") > 0 ? vm["stairs-speed"].as<float>() : 0.5f
    );

    using clock = std::chrono::steady_clock;
    auto next = clock::now();
    for(int step = 0; step < steps; ++step)
    {
        lowstate->update();
        env->step();
        const auto raw = env->action_manager->raw_actions();
        const auto processed = env->action_manager->processed_actions();
        const auto& data = env->robot->data;
        if(raw.size() != 16 || processed.size() != 36)
        {
            spdlog::error(
                "Unexpected policy output sizes: raw={} processed={}",
                raw.size(),
                processed.size()
            );
            return 2;
        }

        float max_abs_raw = 0.0f;
        for(float value : raw) {
            max_abs_raw = std::max(max_abs_raw, std::abs(value));
        }
        float min_kp = processed[12];
        float max_kp = processed[12];
        for(std::size_t i = 12; i < 24; ++i)
        {
            min_kp = std::min(min_kp, processed[i]);
            max_kp = std::max(max_kp, processed[i]);
        }

        if(step == 0 || step == steps - 1 || step % 25 == 0)
        {
            spdlog::info(
                "ROBOT_PRINT step={} gravity=[{:.3f},{:.3f},{:.3f}] "
                "gyro=[{:.3f},{:.3f},{:.3f}] raw_max_abs={:.3f} kp=[{:.1f},{:.1f}]",
                step,
                data.projected_gravity_b[0],
                data.projected_gravity_b[1],
                data.projected_gravity_b[2],
                data.root_ang_vel_b[0],
                data.root_ang_vel_b[1],
                data.root_ang_vel_b[2],
                max_abs_raw,
                min_kp,
                max_kp
            );
        }

        next += std::chrono::duration_cast<clock::duration>(
            std::chrono::duration<double>(env->step_dt)
        );
        std::this_thread::sleep_until(next);
    }

    spdlog::info("ROBOT_PRINT complete. No LowCmd or motor commands were published.");
    return 0;
}

int lowcmd_monitor(float seconds)
{
    if(seconds <= 0.0f)
    {
        spdlog::error("--lowcmd-monitor-seconds must be positive.");
        return 1;
    }

    std::atomic<uint64_t> sample_count{0};
    unitree::robot::SubscriptionBase<unitree_go::msg::dds_::LowCmd_> lowcmd_sub(
        "rt/lowcmd",
        [&sample_count](const void*) {
            sample_count.fetch_add(1, std::memory_order_relaxed);
        }
    );

    spdlog::info(
        "LOWCMD_MONITOR started for {:.2f}s. No LowCmd publisher was created.",
        seconds
    );
    const auto start = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration<double>(seconds);
    uint64_t last_count = 0;
    auto last_time = start;
    while(std::chrono::steady_clock::now() - start < duration)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        const auto now = std::chrono::steady_clock::now();
        const auto count = sample_count.load(std::memory_order_relaxed);
        const double dt = std::chrono::duration<double>(now - last_time).count();
        const double hz = dt > 0.0 ? static_cast<double>(count - last_count) / dt : 0.0;
        spdlog::info(
            "LOWCMD_MONITOR interval_count={} interval_hz={:.1f}",
            count - last_count,
            hz
        );
        last_count = count;
        last_time = now;
    }

    const auto end = std::chrono::steady_clock::now();
    const double elapsed = std::chrono::duration<double>(end - start).count();
    const auto total = sample_count.load(std::memory_order_relaxed);
    const double hz = elapsed > 0.0 ? static_cast<double>(total) / elapsed : 0.0;
    spdlog::info(
        "LOWCMD_MONITOR total_count={} elapsed={:.2f}s average_hz={:.1f}",
        total,
        elapsed,
        hz
    );
    spdlog::info("LOWCMD_MONITOR complete. No commands were published.");
    return 0;
}

int service_list()
{
    RawRobotStateClient raw_client;
    raw_client.SetTimeout(5.0f);
    raw_client.Init();
    std::string raw_data;
    const int32_t raw_result = raw_client.ServiceListRaw(raw_data);
    if(raw_result == 0)
    {
        spdlog::info("SERVICE_LIST_RAW data={}", raw_data);
        spdlog::info("SERVICE_LIST complete. No LowCmd or motor commands were published.");
        return 0;
    }
    spdlog::warn("SERVICE_LIST_RAW_FAILED={}; trying typed client", raw_result);

    unitree::robot::go2::RobotStateClient client;
    client.SetTimeout(5.0f);
    client.Init();

    std::vector<unitree::robot::go2::ServiceState> services;
    const int32_t result = client.ServiceList(services);
    if(result != 0)
    {
        spdlog::error("SERVICE_LIST_FAILED={}", result);
        return 1;
    }

    spdlog::info("SERVICE_LIST count={}", services.size());
    for(const auto& service : services)
    {
        spdlog::info(
            "SERVICE name={} status={} protect={}",
            service.name,
            service.status,
            service.protect
        );
    }
    spdlog::info("SERVICE_LIST complete. No LowCmd or motor commands were published.");
    return 0;
}
}

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

InitResult init_fsm_state(bool check_only)
{
    auto lowcmd_sub = std::make_shared<unitree::robot::go2::subscription::LowCmd>();
    lowcmd_sub->set_timeout_ms(500);
    usleep(1200 * 1000);
    const bool lowcmd_active = !lowcmd_sub->isTimeout();

    if(check_only)
    {
        if(lowcmd_active) {
            spdlog::warn("LOWCMD_STATUS=ACTIVE");
        } else {
            spdlog::info("LOWCMD_STATUS=IDLE");
        }

        spdlog::info("Read-only check complete; no commands were published.");
        return InitResult::CheckComplete;
    }

    if(lowcmd_active)
    {
        spdlog::critical(
            "The lowcmd channel is already active. No commands were published."
        );
        spdlog::critical(
            "Place Go2 in zero-torque mode, press L2 + R2 for Debug Mode, "
            "and make sure no other low-level controller is running."
        );
        return InitResult::LowCmdBusy;
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
    return InitResult::Ready;
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv, true);

    if(vm.count("stairs-policy") > 0)
    {
        param::config["FSM"]["Stairs"]["policy_dir"] =
            vm["stairs-policy"].as<std::string>();
    }
    if(vm.count("stairs-trace") > 0)
    {
        param::config["FSM"]["Stairs"]["trace_file"] =
            vm["stairs-trace"].as<std::string>();
    }
    if(vm.count("stairs-speed") > 0)
    {
        const float speed = vm["stairs-speed"].as<float>();
        if(speed <= 0.0f)
        {
            spdlog::error("--stairs-speed must be positive.");
            return 1;
        }
        param::config["FSM"]["Stairs"]["speed_override"] = speed;
    }
    if(vm.count("stairs-auto-lie-seconds") > 0)
    {
        const float seconds = vm["stairs-auto-lie-seconds"].as<float>();
        if(seconds <= 0.0f)
        {
            spdlog::error("--stairs-auto-lie-seconds must be positive.");
            return 1;
        }
        param::config["FSM"]["Stairs"]["auto_lie_seconds"] = seconds;
    }
    if(vm.count("stairs-auto-zero-seconds") > 0)
    {
        const float seconds = vm["stairs-auto-zero-seconds"].as<float>();
        if(seconds <= 0.0f)
        {
            spdlog::error("--stairs-auto-zero-seconds must be positive.");
            return 1;
        }
        param::config["FSM"]["Stairs"]["auto_zero_seconds"] = seconds;
    }
    if(vm.count("stairs-safety-disable") > 0)
    {
        param::config["FSM"]["Stairs"]["safety_enabled"] = false;
    }
    if(vm.count("stairs-safety-q-error") > 0)
    {
        const float value = vm["stairs-safety-q-error"].as<float>();
        if(value <= 0.0f)
        {
            spdlog::error("--stairs-safety-q-error must be positive.");
            return 1;
        }
        param::config["FSM"]["Stairs"]["safety_q_error_limit"] = value;
    }
    if(vm.count("stairs-safety-dq") > 0)
    {
        const float value = vm["stairs-safety-dq"].as<float>();
        if(value <= 0.0f)
        {
            spdlog::error("--stairs-safety-dq must be positive.");
            return 1;
        }
        param::config["FSM"]["Stairs"]["safety_dq_limit"] = value;
    }
    if(vm.count("stairs-safety-raw-action") > 0)
    {
        const float value = vm["stairs-safety-raw-action"].as<float>();
        if(value <= 0.0f)
        {
            spdlog::error("--stairs-safety-raw-action must be positive.");
            return 1;
        }
        param::config["FSM"]["Stairs"]["safety_raw_action_limit"] = value;
    }
    if(vm.count("stairs-safety-kp") > 0)
    {
        const float value = vm["stairs-safety-kp"].as<float>();
        if(value <= 0.0f)
        {
            spdlog::error("--stairs-safety-kp must be positive.");
            return 1;
        }
        param::config["FSM"]["Stairs"]["safety_kp_saturation_limit"] = value;
    }
    if(vm.count("stairs-safety-kp-frames") > 0)
    {
        const int value = vm["stairs-safety-kp-frames"].as<int>();
        if(value <= 0)
        {
            spdlog::error("--stairs-safety-kp-frames must be positive.");
            return 1;
        }
        param::config["FSM"]["Stairs"]["safety_kp_saturation_frames"] = value;
    }
    if(vm.count("stairs-safety-orientation-rad") > 0)
    {
        const float value = vm["stairs-safety-orientation-rad"].as<float>();
        if(value <= 0.0f)
        {
            spdlog::error("--stairs-safety-orientation-rad must be positive.");
            return 1;
        }
        param::config["FSM"]["Stairs"]["bad_orientation_limit"] = value;
    }

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     Go2 Controller \n";

    // Unitree DDS Config
    unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());

    const int exclusive_modes =
        (vm.count("check-lowcmd") > 0 ? 1 : 0) +
        (vm.count("lowcmd-monitor-seconds") > 0 ? 1 : 0) +
        (vm.count("service-list") > 0 ? 1 : 0) +
        (vm.count("release-motion") > 0 ? 1 : 0) +
        (vm.count("robot-print") > 0 ? 1 : 0);
    if(exclusive_modes > 1)
    {
        spdlog::error(
            "Use only one of --check-lowcmd, --lowcmd-monitor-seconds, "
            "--service-list, --release-motion, or --robot-print."
        );
        return 1;
    }
    if(vm.count("lowcmd-monitor-seconds") > 0)
    {
        return lowcmd_monitor(vm["lowcmd-monitor-seconds"].as<float>());
    }
    if(vm.count("service-list") > 0)
    {
        return service_list();
    }
    if(vm.count("release-motion") > 0)
    {
        return release_motion_mode();
    }
    if(vm.count("robot-print") > 0)
    {
        return robot_print(vm);
    }

    const auto init_result = init_fsm_state(vm.count("check-lowcmd") > 0);
    if(init_result == InitResult::CheckComplete)
    {
        return 0;
    }
    if(init_result == InitResult::LowCmdBusy)
    {
        return 2;
    }

    FSMState::keyboard = std::make_shared<Keyboard>();
    if(!FSMState::keyboard->available())
    {
        spdlog::critical(
            "Keyboard control requires an interactive terminal. "
            "When connecting over SSH, allocate a TTY with ssh -tt."
        );
        return 1;
    }

    std::signal(SIGINT, handle_signal);
    std::signal(SIGTERM, handle_signal);

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [F] for FixStand, preselect speed, then [G] to start.\n";
    std::cout << "Before [G]: [W/Up] faster, [S/Down] slower, "
                 "[1-4] select 0.5-0.8 m/s. In Stairs: [Space/0] stop.\n";
    std::cout << "Press [X] to lie down, then [X] again for Passive damping. "
                 "Remote L2+Up enters FixStand; L2+B starts LieDown.\n";

    while (keep_running)
    {
        sleep(1);
    }

    if(fsm)
    {
        fsm->stop_for_process_exit();
        (void)fsm.release();
    }
    FSMState::keyboard.reset();

    std::cout.flush();
    std::cerr.flush();
    spdlog::shutdown();

    // Unitree SDK2 / CycloneDDS on this Go2 has shown intermittent heap
    // corruption during process-wide/static object destruction after Ctrl-C.
    // Stop live threads first, but intentionally skip destroying the FSM state
    // graph because it owns SDK/DDS/ONNX objects whose destructors are fragile
    // on the robot.
    std::_Exit(0);
}
