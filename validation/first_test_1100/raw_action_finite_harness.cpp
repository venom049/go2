#include <algorithm>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <limits>
#include <cassert>
#include <iostream>
struct Decoder {
static constexpr std::size_t policy_action_dim=16,joint_action_dim=12,stiffness_action_dim=4;
std::vector<float> raw=std::vector<float>(16,123.0f),processed=std::vector<float>(36),offsets=std::vector<float>(12);
float action_clip=4.5f,position_scale=.25f,kp_default=40,kp_action_scale=0,kp_min=40,kp_max=40,kd_sqrt_scale=.158113883f;
void process_actions(std::vector<float> actions) {
        if(actions.size() != policy_action_dim)
        {
            throw std::runtime_error("Go2 stair policy must output 16 actions.");
        }

        // Validate the entire original vector BEFORE clipping or mutating raw.
        // std::clamp(+/-Inf) is finite, which would otherwise hide a bad model
        // output from both the legacy safety check and publication guard.
        for(float value : actions) {
            if(!std::isfinite(value))
                throw std::runtime_error("Nonfinite raw policy output before clipping");
        }
        for(std::size_t i = 0; i < policy_action_dim; ++i)
        {
            raw[i] = std::clamp(actions[i], -action_clip, action_clip);
        }

        for(std::size_t i = 0; i < joint_action_dim; ++i)
        {
            processed[i] = offsets[i] + position_scale * raw[i];
        }

        for(std::size_t leg = 0; leg < stiffness_action_dim; ++leg)
        {
            const float kp = std::clamp(
                kp_default + kp_action_scale * raw[joint_action_dim + leg],
                kp_min,
                kp_max
            );
            const float kd = kd_sqrt_scale * std::sqrt(kp);
            for(std::size_t joint = 0; joint < 3; ++joint)
            {
                const std::size_t index = leg * 3 + joint;
                processed[joint_action_dim + index] = kp;
                processed[2 * joint_action_dim + index] = kd;
            }
        }
    }
};
int main(){
for(float invalid:{std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN()}) {
for(int i=0;i<16;++i){Decoder decoder;std::vector<float> actions(16,0.0f);actions[i]=invalid;bool threw=false;try{decoder.process_actions(actions);}catch(const std::runtime_error&){threw=true;}assert(threw);for(float raw:decoder.raw)assert(raw==123.0f);}}
Decoder finite;std::vector<float> actions(16,5.0f);finite.process_actions(actions);for(float raw:finite.raw)assert(raw==4.5f);
std::cout<<"PASS: actual extracted process_actions body rejects NaN/+Inf/-Inf in all 16 slots before raw mutation; finite clip preserved\n";}
