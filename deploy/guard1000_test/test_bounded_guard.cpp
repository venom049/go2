// Offline-only; no robot SDK or sockets are linked.
#include "FirstTestGuard.h"
#include "CandidateIdentity.h"
#include "StairsCommandBuffer.h"
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>

#define CHECK(x) do { if(!(x)) { std::cerr << "FAILED " << __LINE__ << ": " << #x << '\n'; std::exit(1); } } while(false)
using namespace go2_first_test;

int main(int argc, char** argv) {
    CHECK(argc == 2);
    GuardInput state;
    state.state_available=true;
    state.quaternion_wxyz={1.,0.,0.,0.};
    state.q={0.,.8,-1.5,0.,.8,-1.5,0.,1.,-1.5,0.,1.,-1.5};
    GuardOutput proposed;
    proposed.q=state.q; proposed.kp.fill(40.); proposed.kd.fill(1.);
    for(double duration:{.5,1.,2.}) {
        Guard guard;
        CHECK(guard.configure_active_seconds(duration));
        CHECK(!guard.configure_active_seconds(2.));
        CHECK(!guard.filter(state,proposed,0.,false).command_allowed);
        ++state.tick;
        CHECK(guard.filter(state,proposed,.001,true).policy_allowed);
        // Exit/re-entry cannot pause or restart the timer. Maintain fresh data.
        for(int i=1; i<static_cast<int>(duration*1000); ++i) {
            ++state.tick;
            CHECK(guard.filter(state,proposed,.001+i*.001,i%3!=0).command_allowed);
        }
        ++state.tick;
        auto result=guard.filter(state,proposed,.001+duration,true);
        CHECK(result.latched && !result.command_allowed && !result.policy_allowed);
        CHECK(result.command.kp[0]==0. && result.command.kd[0]==3.);
        CHECK(!guard.configure_active_seconds(2.));
        ++state.tick;
        CHECK(guard.filter(state,proposed,.002+duration,false).latched);
    }
    for(double invalid:{0.,-.5,.6,1.5,2.001,std::numeric_limits<double>::quiet_NaN(),
                        std::numeric_limits<double>::infinity()}) {
        Guard guard;
        CHECK(!guard.configure_active_seconds(invalid));
    }
    Guard started;
    started.filter(state,proposed,0.,false);
    CHECK(!started.configure_active_seconds(.5));
    for(float speed:{.45f,.6f}) {
        StairsCommandBuffer buffer;
        buffer.configure(speed,speed,.1f,speed);
        CHECK(!buffer.armed());
        CHECK(buffer.select("1"));
        CHECK(buffer.armed() && std::abs(buffer.selected_speed()-speed)<1e-6f);
        CHECK(!buffer.select("2"));
        buffer.begin_selection();
        CHECK(!buffer.armed());
    }
    const std::filesystem::path candidate=argv[1];
    CHECK(go2_guard1000::sha256_file(candidate/"exported/policy.onnx")==go2_guard1000::kOnnxSha256);
    CHECK(go2_guard1000::sha256_file(candidate/"params/deploy.yaml")==go2_guard1000::kConfigSha256);
    CHECK(!go2_guard1000::verified_candidate(candidate)); // Local export cannot impersonate onboard path.
    bool missing_failed=false;
    try { go2_guard1000::sha256_file(candidate/"does_not_exist"); }
    catch(const std::exception&) { missing_failed=true; }
    CHECK(missing_failed);
    char temporary[]="/tmp/go2-guard1000-identity-XXXXXX";
    const auto created=mkdtemp(temporary);
    CHECK(created != nullptr);
    const std::filesystem::path fixture_dir=created;
    const auto fixture=fixture_dir/"literal $(false) with spaces";
    {std::ofstream stream(fixture);stream<<"abc";}
    CHECK(go2_guard1000::sha256_file(fixture)==
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    {std::ofstream stream(fixture);stream<<"abcd";}
    CHECK(go2_guard1000::sha256_file(fixture)!=
          "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    const auto link=fixture_dir/"link";
    std::filesystem::create_symlink(fixture,link);
    bool symlink_failed=false;
    try {go2_guard1000::sha256_file(link);}
    catch(const std::exception&) {symlink_failed=true;}
    CHECK(symlink_failed);
    std::filesystem::remove(link);
    std::filesystem::remove(fixture);
    std::filesystem::remove(fixture_dir);
    std::cout << "PASS: bounded deadline, no rearm, invalid inputs, exact identity hashes; no robot access.\n";
}
