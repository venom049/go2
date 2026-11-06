#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "State_StairsReady.h"
#include "State_LieDown.h"
#include "unitree_articulation.h"
#include <csignal>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cmath>
#include <thread>
#include <net/if.h>
#include "StandReadiness.h"
#include "HandoffHold.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <cstring>

namespace
{
volatile std::sig_atomic_t keep_running = 1;

void handle_signal(int)
{
    keep_running = 0;
}

void override_stairs_speed(YAML::Node cfg, float speed)
{
    if(!std::isfinite(speed) || std::abs(speed - 0.6f) > 1.0e-6f)
    {
        throw std::runtime_error("This first-test build accepts only --stairs-speed 0.6.");
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
    if(steps <= 0 || steps > 3000)
    {
        spdlog::error("--robot-print-steps must be in [1, 3000].");
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
    spdlog::warn("ROBOT_PRINT has no submitted motor target: applied history is unavailable; "
                 "this is a shape/runtime diagnostic, not a closed-loop policy validation.");
    spdlog::info(
        "ROBOT_PRINT started: steps={} speed={:.2f} m/s. No LowCmd publisher was created.",
        steps,
        0.6f
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
    if(!std::isfinite(seconds) || seconds <= 0.0f || seconds > 60.0f)
    {
        spdlog::error("--lowcmd-monitor-seconds must be in (0, 60].");
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


}

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

// This is deliberately a new executable in an isolated copied deploy tree.
// Its default path never initializes DDS and never creates any publisher.
namespace {
constexpr const char* kCandidateDirectory =
    "/home/unitree/go2-stairs-addon-1100-20261103/legacy_candidate/v0";

bool candidate_selected(const boost::program_options::variables_map& vm)
{
    if(vm.count("stairs-policy") == 0) return false;
    const std::filesystem::path requested(vm["stairs-policy"].as<std::string>());
    if(!requested.is_absolute() || requested.lexically_normal() != kCandidateDirectory)
        return false;
    if(!std::filesystem::is_regular_file(requested / "params/deploy.yaml") ||
       !std::filesystem::is_regular_file(requested / "exported/policy.onnx"))
        return false;
    return true;  // Model/config hashes must separately match the deployment record.
}

bool check_lowcmd_active()
{
    // Avoid SubscriptionBase::isTimeout(), whose receive timestamp is not atomic.
    std::atomic<std::int64_t> last_receive_ns{0};
    unitree::robot::SubscriptionBase<unitree_go::msg::dds_::LowCmd_> subscriber(
        "rt/lowcmd", [&last_receive_ns](const void*) {
            last_receive_ns.store(std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count());
        });
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    // Any observed competing traffic aborts. This is a preflight observation,
    // not a mutual-exclusion lease or proof that another publisher cannot start.
    return last_receive_ns.load() != 0;
}

bool preflight_position_control()
{
    FSMState::lowstate = std::make_shared<LowState_t>();
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
    go2_first_test::Guard probe;
    bool probe_started = false;
    int accepted_samples = 0;
    while(keep_running && std::chrono::steady_clock::now() < deadline) {
        const auto input = FSMState::snapshot();
        double norm_squared = 0.0;
        for(double v : input.quaternion_wxyz) norm_squared += v * v;
        // Default/no-message storage is not considered a received measurement.
        if(!probe_started && (!std::isfinite(norm_squared) ||
                             std::abs(norm_squared - 1.0) > 0.1)) {
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
            continue;
        }
        probe_started = true;
        auto proposed = go2_first_test::Guard::damping();
        proposed.q = input.q;
        proposed.kp.fill(1.0);  // Check position-control prerequisites, never send.
        const auto result = probe.filter(input, proposed, FSMState::steady_seconds(), false);
        if(result.latched) {
            spdlog::error("FIRST_TEST_PREFLIGHT_REJECTED reason={} (no publisher)", result.reason);
            return false;
        }
        if(result.command_allowed && ++accepted_samples >= 20) {
            spdlog::info("Preflight: finite in-range pose and advancing tick; "
                         "this is not permission for unprotected locomotion.");
            return true;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
    spdlog::error("FIRST_TEST_PREFLIGHT_REJECTED: no fresh valid state in 3 s (no publisher)");
    return false;
}

void damping_before_exit(CtrlFSM* fsm)
{
    if(!FSMState::lowcmd || !FSMState::control_enabled) return;
    FSMState::request_stop();
    if(fsm) fsm->stop_for_process_exit();  // Joins the FSM before main takes ownership.
    FSMState::latch_stop("process_exit");
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
    unsigned requested = 0;
    while(std::chrono::steady_clock::now() < deadline) {
        if(FSMState::publish_damping_once()) ++requested;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    // Allow asynchronous publisher to service its last finite damping request.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    spdlog::warn("FIRST_TEST_EXIT damping_requests={} over 250ms; "
                 "DDS delivery/motor response is not acknowledged; no factory-mode restore.",
                 requested);
}

struct HandoffObservedCommand {
    std::mutex mutex;
    go2_first_test::GuardOutput command;
    double received_at=0.;
};

YAML::Node bridge_request(const char* operation)
{
    const int fd=socket(AF_UNIX,SOCK_STREAM,0);
    if(fd<0) throw std::runtime_error("Cannot open local motion RPC bridge socket");
    timeval timeout{4,0};
    setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));
    setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));
    sockaddr_un addr{};addr.sun_family=AF_UNIX;
    std::strcpy(addr.sun_path,"/home/unitree/go2-stairs-handoff-20261103/validation/motion_rpc.sock");
    if(connect(fd,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))!=0) {
        close(fd);throw std::runtime_error("Local Python motion bridge is unavailable");
    }
    const std::string request=std::string(operation)+"\n";
    if(send(fd,request.data(),request.size(),MSG_NOSIGNAL)!=static_cast<ssize_t>(request.size())) {
        close(fd);throw std::runtime_error("Motion bridge request failed");
    }
    std::string response;char buf[512];
    while(response.find('\n')==std::string::npos && response.size()<4096) {
        const auto n=recv(fd,buf,sizeof(buf),0);
        if(n<=0) {close(fd);throw std::runtime_error("Motion bridge result unavailable; do not retry release");}
        response.append(buf,n);
    }
    close(fd);
    const auto result=YAML::Load(response);
    spdlog::info("MOTION_BRIDGE {}",response);
    if(result["operation"].as<std::string>()!=operation || !result["ok"].as<bool>())
        throw std::runtime_error("Python bridge did not verify requested mode transition");
    return result;
}

// The SDK writer is pre-created but never unlocked for publication until the
// checked handoff succeeds. Constructor/warmup failures send no damping either.
void protected_handoff(std::unique_ptr<CtrlFSM>& fsm)
{
    bridge_request("check");
    auto observed=std::make_shared<HandoffObservedCommand>();
    unitree::robot::SubscriptionBase<unitree_go::msg::dds_::LowCmd_> monitor(
        "rt/lowcmd",[observed](const void* ptr) {
            const auto& msg=*static_cast<const unitree_go::msg::dds_::LowCmd_*>(ptr);
            std::lock_guard<std::mutex> lock(observed->mutex);
            for(std::size_t i=0;i<12;++i) {
                const auto& m=msg.motor_cmd()[i];
                observed->command.q[i]=m.q(); observed->command.dq[i]=m.dq();
                observed->command.kp[i]=m.kp(); observed->command.kd[i]=m.kd();
                observed->command.tau[i]=m.tau();
            }
            observed->received_at=FSMState::steady_seconds();
        });
    if(!preflight_position_control()) throw std::runtime_error("Pose rejected before release; original mode unchanged");
    FSMState::lowcmd=std::make_unique<LowCmd_t>();
    fsm=std::make_unique<CtrlFSM>(param::config["FSM"]);
    for(const auto& state:fsm->states) {
        if(auto* rl=dynamic_cast<State_RLBase*>(state.get())) rl->warmup_without_commands();
    }
    spdlog::info("HANDOFF_PRELOADED: no motor commands sent. H requests one mcf release and immediate hold; "
                 "X/space/0 cancels. 45s deadline. Fall support remains required.");
    const double deadline=FSMState::steady_seconds()+45.;
    go2_first_test::Guard probe;
    go2_first_test::StandReadiness stability;
    go2_first_test::GuardOutput initial;
    bool authorized=false;
    while(keep_running && FSMState::steady_seconds()<deadline) {
        const double now=FSMState::steady_seconds();
        double received=0.;
        go2_first_test::GuardOutput factory;
        {std::lock_guard<std::mutex> lock(observed->mutex);factory=observed->command;received=observed->received_at;}
        const auto s=FSMState::snapshot();
        if(!go2_first_test::make_handoff_hold(s,factory,initial))
            throw std::runtime_error("Factory stand cannot be converted to bounded 40/1 hold; no release");
        const auto checked=probe.filter(s,initial,now,false);
        if(checked.latched) throw std::runtime_error("Factory command/pose rejected: "+checked.reason);
        bool supporting=true;
        for(double kp:initial.kp) supporting=supporting && kp>0.;
        const bool ready=stability.update(s,initial.q,now,checked.command_allowed && supporting &&
                                         received>0. && now-received<.02);
        FSMState::keyboard->update();
        std::string key;
        while(FSMState::keyboard->pop_key(key)) {
            if(key=="x" || key=="X" || key==" " || key=="0")
                throw std::runtime_error("Handoff cancelled before release");
            if(key=="h" || key=="H") {
                if(ready) authorized=true;
                else spdlog::warn("Handoff not ready: waiting for stable supported factory stand");
            }
        }
        if(authorized) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
    if(!authorized || !keep_running) throw std::runtime_error("Handoff not requested; original mode unchanged");
    // One bounded request, no retry or service manipulation. No policy is armed.
    const double release_start=FSMState::steady_seconds();
    const auto released=bridge_request("release");
    if(!released["release_attempted"].as<bool>() || released["release_code"].as<int>()!=0 ||
       released["check_code"].as<int>()!=0 || released["mode"]["name"].as<std::string>()!="")
        throw std::runtime_error("Released mode not verified empty; no custom output");
    spdlog::info("HANDOFF_RELEASE elapsed_ms={:.1f}",1000.*(FSMState::steady_seconds()-release_start));
    const double quiet_deadline=FSMState::steady_seconds()+.1;
    bool quiet=false;
    while(keep_running && FSMState::steady_seconds()<quiet_deadline) {
        double received=0.;
        {std::lock_guard<std::mutex> lock(observed->mutex);received=observed->received_at;}
        if(FSMState::steady_seconds()-received>=.02) {quiet=true;break;}
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    if(!quiet || !keep_running) throw std::runtime_error("Competing traffic after release; no custom output");
    // Prime the same publication guard using fresh states. Its first unprimed
    // fallback is checked only in memory and never sent to the standing robot.
    auto checked=FSMState::guard.filter(FSMState::snapshot(),initial,FSMState::steady_seconds(),false);
    const double prime_deadline=FSMState::steady_seconds()+.02;
    while(!checked.command_allowed && !checked.latched && keep_running && FSMState::steady_seconds()<prime_deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        checked=FSMState::guard.filter(FSMState::snapshot(),initial,FSMState::steady_seconds(),false);
    }
    if(!checked.command_allowed || !keep_running)
        throw std::runtime_error("Fresh hold check rejected after release: "+checked.reason);
    FSMState::lowcmd->lock();
    FSMState::write_command(initial);
    FSMState::lowcmd->unlock(); // No publication yet.
    FSMState::control_enabled=true;
    fsm->start(true);
    spdlog::info("HANDOFF_HOLD_STARTED elapsed_since_release_ms={:.1f}; awaiting measured STAND_LOCKED. "
                 "Software timing is not a hard watchdog or delivery acknowledgment.",
                 1000.*(FSMState::steady_seconds()-release_start));
}
}

int main(int argc, char** argv)
{
    bool actuate = false;
    bool preflight_only = false;
    bool handoff=false;
    bool rpc_check=false;
    std::vector<char*> filtered_argv{argv[0]};
    for(int i = 1; i < argc; ++i) {
        const std::string argument(argv[i]);
        if(argument == "--protected-handoff") {
            if(actuate) { std::cerr << "Duplicate actuation mode\n"; return 1; }
            actuate=true; handoff=true;
        } else if(argument == "--handoff-rpc-check") {
            rpc_check=true;
        } else if(argument == "--first-test-actuate") {
            if(actuate) { std::cerr << "Duplicate --first-test-actuate\n"; return 1; }
            actuate = true;
        } else if(argument == "--first-test-preflight") {
            if(preflight_only) { std::cerr << "Duplicate --first-test-preflight\n"; return 1; }
            preflight_only = true;
        } else {
            // These switches are present in the copied parser but unusable in
            // this executable, including --name=value spellings.
            if(argument.rfind("--service-list", 0) == 0 ||
               argument.rfind("--release-motion", 0) == 0 ||
               argument.rfind("--stairs-safety-", 0) == 0 ||
               argument.rfind("--stairs-auto-", 0) == 0) {
                std::cerr << "Disabled in first-test build: " << argument << '\n';
                return 1;
            }
            filtered_argv.push_back(argv[i]);
        }
    }
    const int filtered_argc = static_cast<int>(filtered_argv.size());
    filtered_argv.push_back(nullptr);
    std::unique_ptr<CtrlFSM> fsm;
    int exit_code = 0;
    try {
        auto vm = param::helper(filtered_argc, filtered_argv.data(), true);
        const int read_modes = (vm.count("check-lowcmd") ? 1 : 0) +
            (vm.count("lowcmd-monitor-seconds") ? 1 : 0) +
            (vm.count("service-list") ? 1 : 0) + (vm.count("robot-print") ? 1 : 0);
        if(read_modes > 1 || (rpc_check && (actuate || preflight_only || read_modes)) || (actuate && (read_modes || preflight_only)) ||
           (preflight_only && read_modes))
            throw std::runtime_error("Choose one read-only mode, first-test preflight, OR explicit actuation.");
        // Reject disabled abbreviations accepted by Boost's default parser.
        for(const auto& entry : vm) {
            if(entry.first == "service-list" || entry.first == "release-motion" || entry.first.rfind("stairs-safety-", 0) == 0 ||
               entry.first.rfind("stairs-auto-", 0) == 0)
                throw std::runtime_error("Disabled first-test option: " + entry.first);
        }
        if(vm.count("stairs-speed")) {
            const float speed = vm["stairs-speed"].as<float>();
            if(!std::isfinite(speed) || std::abs(speed - 0.6f) > 1.0e-6f)
                throw std::runtime_error("First-test stairs command is fixed at 0.6 m/s.");
        }
        if(!actuate && !preflight_only && read_modes == 0 && !rpc_check) {
            std::cout << "FIRST_TEST_DRY_RUN: no DDS, no publisher, no motor commands.\n"
                         "--first-test-actuate is required for protected 2-second Stairs output.\n"
                         "Read-only: --first-test-preflight / --check-lowcmd / --robot-print / --lowcmd-monitor-seconds.\n"
                         "Do not actuate until the exact candidate hash, stable supported pose,\n"
                         "exclusive control handoff and hardware stop are verified.\n";
            return 0;
        }
        if((actuate || preflight_only || vm.count("robot-print")) && !candidate_selected(vm))
            throw std::runtime_error("Explicit --stairs-policy must select the installed cumulative-1100 candidate v0.");
        if(vm.count("stairs-policy"))
            param::config["FSM"]["Stairs"]["policy_dir"] = vm["stairs-policy"].as<std::string>();
        if(vm.count("stairs-trace"))
            param::config["FSM"]["Stairs"]["trace_file"] = vm["stairs-trace"].as<std::string>();
        if(actuate || preflight_only) {
            const auto deploy = YAML::LoadFile(std::string(kCandidateDirectory) + "/params/deploy.yaml");
            const auto target = deploy["default_joint_pos"].as<std::vector<float>>();
            if(target.size() != 12) throw std::runtime_error("Candidate must have 12 stand offsets.");
            for(std::size_t i = 0; i < 12; ++i) {
                if(!std::isfinite(target[i]) || target[i] < go2_first_test::kTargetLower[i] ||
                   target[i] > go2_first_test::kTargetUpper[i])
                    throw std::runtime_error("Candidate stand offset outside software bounds.");
            }
            // New process only: interpolate from the current in-range pose
            // directly to candidate stand offsets, never via old deep crouch.
            param::config["FSM"]["FixStand"]["ts"] = std::vector<float>{0.0f, 2.0f};
            param::config["FSM"]["FixStand"]["qs"] =
                std::vector<std::vector<float>>{{}, target};
        }
        auto stairs = param::config["FSM"]["Stairs"];
        stairs["speed_override"] = 0.6f;
        stairs["safety_enabled"] = true;
        stairs["auto_lie_seconds"] = 0.0;
        stairs["auto_zero_seconds"] = 0.0;
        if(actuate) {
            const auto interface = vm["network"].as<std::string>();
            if(interface.empty() || if_nametoindex(interface.c_str()) == 0)
                throw std::runtime_error("Explicit existing --network interface is required for actuation.");
        }
        std::signal(SIGINT, handle_signal);
        std::signal(SIGTERM, handle_signal);
        std::signal(SIGHUP, handle_signal);
        unitree::robot::ChannelFactory::Instance()->Init(0, vm["network"].as<std::string>());
        if(rpc_check) {
            bridge_request("check");
            spdlog::info("HANDOFF_RPC_CHECK read_only=true");
            return 0;
        }
        if(vm.count("lowcmd-monitor-seconds"))
            return lowcmd_monitor(vm["lowcmd-monitor-seconds"].as<float>());
        if(vm.count("robot-print")) return robot_print(vm);
        if(vm.count("check-lowcmd")) {
            const bool active = check_lowcmd_active();
            spdlog::info("LOWCMD_STATUS={} (read-only, no publisher)", active ? "ACTIVE" : "IDLE");
            return active ? 2 : 0;
        }
        if(preflight_only) {
            const bool busy = check_lowcmd_active();
            spdlog::info("FIRST_TEST_PREFLIGHT LOWCMD_STATUS={}", busy ? "ACTIVE" : "IDLE");
            const bool pose_ok = preflight_position_control();
            spdlog::info("FIRST_TEST_PREFLIGHT_RESULT lowcmd_idle={} pose_probe_passed={} "
                         "publisher_created=false control_authorized=false",
                         !busy, pose_ok);
            // Even a successful preflight never creates a publisher.
            return (!busy && pose_ok) ? 0 : 3;
        }
        // From this point only the explicit actuate path can proceed.
        FSMState::keyboard = std::make_shared<Keyboard>();
        if(!FSMState::keyboard->available())
            throw std::runtime_error("First test requires an attended interactive TTY.");
        if(handoff) {
            protected_handoff(fsm);
        } else {
        if(check_lowcmd_active()) {
            spdlog::error("FIRST_TEST_REJECTED: rt/lowcmd already active; no takeover or publisher.");
            return 2;
        }
        if(!preflight_position_control()) return 3;
        // A second traffic check reduces (but cannot eliminate) the handoff race.
        if(check_lowcmd_active() || !keep_running) return 2;
        FSMState::lowcmd = std::make_unique<LowCmd_t>();
        fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
        FSMState::control_enabled=true;
        fsm->start();
        }
        std::cout << "FIRST_TEST: F=FixStand; after settling press 1 to select 0.6, then G.\n"
                     "Stairs has a 2-second software deadline checked at each publication boundary.\n"
                     "X / Space / 0 or remote stop transition requests latched fixed damping.\n"
                     "Native blocking/network failure defeats software timing; hardware stop remains essential.\n"
                     "No automatic LieDown, rearm or factory-mode restoration. Ctrl-C exits after damping.\n";
        while(keep_running) {
            FSMState::flush_rejection_diagnostic();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    } catch(const std::exception& error) {
        spdlog::critical("FIRST_TEST_ERROR {}", error.what());
        exit_code = 1;
    } catch(...) {
        spdlog::critical("FIRST_TEST_ERROR unknown exception");
        exit_code = 1;
    }
    damping_before_exit(fsm.get());
    FSMState::flush_rejection_diagnostic();
    (void)fsm.release();
    FSMState::keyboard.reset();
    std::cout.flush();
    std::cerr.flush();
    spdlog::shutdown();
    // Preserve the original binary's workaround for known DDS destructor heap
    // corruption, after explicit bounded damping requests and worker shutdown.
    // SIGKILL, power loss, hung native code and lost network cannot be handled here.
    std::_Exit(exit_code);
}
