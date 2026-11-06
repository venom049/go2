// Cross-language test driver: consumes recorded doubles, outputs observations.
#include "MoePolicy.h"
#include <iostream>
int main() {
    go2_moe::History history;std::array<double,45> row{};
    while(std::cin.read(reinterpret_cast<char*>(row.data()),sizeof(row))) {
        go2_first_test::GuardInput s;s.state_available=true;size_t n=0;
        for(auto& v:s.q) v=row[n++];for(auto& v:s.dq) v=row[n++];
        for(auto& v:s.quaternion_wxyz) v=row[n++];for(auto& v:s.gyro) v=row[n++];
        go2_moe::Action previous{};for(auto& v:previous) v=static_cast<float>(row[n++]);
        const float vx=static_cast<float>(row[n++]);
        if(row[n]!=0.) history.clear();
        auto obs=history.append(go2_moe::frame(s,vx,previous));
        std::cout.write(reinterpret_cast<const char*>(obs.data()),sizeof(obs));
    }
}
