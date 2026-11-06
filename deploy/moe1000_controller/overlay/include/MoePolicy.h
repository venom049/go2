#pragma once
// Pure policy contract: no DDS, filesystem or motor publisher.
#include "FirstTestGuard.h"
#include <algorithm>
#include <array>
#include <stdexcept>

namespace go2_moe {
inline constexpr std::array<int,12> kMap{3,4,5,0,1,2,9,10,11,6,7,8};
inline constexpr std::array<float,12> kOffset{
    .1f,.8f,-1.5f,-.1f,.8f,-1.5f,.1f,1.f,-1.5f,-.1f,1.f,-1.5f};
using Action = std::array<float,12>;
using Observation = std::array<float,225>;
using Frame = std::array<float,45>;
inline void finite(double v) {
    if(!std::isfinite(v)) throw std::runtime_error("MoE nonfinite input/output");
}
inline go2_first_test::JointArray stand_sdk() {
    go2_first_test::JointArray result{};
    for(size_t i=0;i<12;++i) result[kMap[i]]=kOffset[i];
    return result;
}
inline Action inverse_submitted(const go2_first_test::GuardOutput& cmd) {
    Action result{};
    for(size_t i=0;i<12;++i) {
        finite(cmd.q[kMap[i]]);
        result[i]=(static_cast<float>(cmd.q[kMap[i]])-kOffset[i])/.25f;
    }
    return result;
}
inline Frame frame(const go2_first_test::GuardInput& s, float vx, const Action& previous) {
    if(!s.state_available) throw std::runtime_error("MoE state unavailable");
    finite(vx);
    if(vx!=0.f && vx!=.6f) throw std::runtime_error("MoE fixed test command required");
    double norm=0.;
    for(double v:s.quaternion_wxyz) {finite(v);norm+=v*v;}
    if(std::abs(std::sqrt(norm)-1.)>.01) throw std::runtime_error("MoE invalid quaternion");
    for(double v:s.q) finite(v);
    for(double v:s.dq) finite(v);
    for(double v:s.gyro) finite(v);
    for(float v:previous) finite(v);
    const auto& q=s.quaternion_wxyz;
    // Inverse quaternion rotation of world down; same convention as training.
    const std::array<double,3> gravity{
        2.*(q[0]*q[2]-q[1]*q[3]),
        -2.*(q[0]*q[1]+q[2]*q[3]),
        -1.+2.*(q[1]*q[1]+q[2]*q[2])};
    Frame f{};
    auto scaled=[](double value,double scale) {
        finite(value);return static_cast<float>(std::clamp(value,-100.,100.)*scale);
    };
    for(size_t i=0;i<3;++i) {f[i]=scaled(s.gyro[i],.25);f[3+i]=scaled(gravity[i],1.);}
    f[6]=scaled(vx,2.);
    for(size_t i=0;i<12;++i) {
        f[9+i]=scaled(s.q[kMap[i]]-kOffset[i],1.);
        f[21+i]=scaled(s.dq[kMap[i]],.05);
        f[33+i]=scaled(previous[i],1.);
    }
    return f;
}
class History {
public:
    void clear() {initialized_=false;}
    Observation append(const Frame& f) {
        for(float v:f) finite(v);
        if(!initialized_) {frames_.fill(f);initialized_=true;}
        else {for(size_t i=0;i<4;++i) frames_[i]=frames_[i+1];frames_[4]=f;}
        Observation obs{};size_t out=0,start=0;
        for(size_t width:{3,3,3,12,12,12}) {
            for(const auto& history_frame:frames_)
                for(size_t j=0;j<width;++j) obs[out++]=history_frame[start+j];
            start+=width;
        }
        return obs;
    }
private:
    bool initialized_=false;
    std::array<Frame,5> frames_{};
};
inline go2_first_test::GuardOutput decode(const Action& action) {
    go2_first_test::GuardOutput result;
    for(size_t i=0;i<12;++i) {
        finite(action[i]);
        // Training clips the scaled target to [-100,100], not the raw action.
        // Physical target/tracking/effort rejection remains at publication.
        const float target=action[i]*.25f+kOffset[i];finite(target);
        const auto sdk=kMap[i];result.q[sdk]=std::clamp(target,-100.f,100.f);
        result.kp[sdk]=20.;result.kd[sdk]=.5;
    }
    return result;
}
} // namespace go2_moe
