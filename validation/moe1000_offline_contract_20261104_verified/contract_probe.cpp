
#include "MoePolicy.h"
#include <iostream>
#include <iomanip>
int main() {
 std::cout<<std::setprecision(17); char mode;
 go2_moe::History history;
 while(std::cin>>mode) {
  if(mode=='a') {
   go2_moe::Action a{}; for(auto& x:a)std::cin>>x;
   auto c=go2_moe::decode(a); for(auto x:c.q)std::cout<<x<<' '; std::cout<<'\n';
  } else if(mode=='o') {
   go2_first_test::GuardInput s; s.state_available=true;
   for(auto& x:s.q)std::cin>>x; for(auto& x:s.dq)std::cin>>x;
   for(auto& x:s.quaternion_wxyz)std::cin>>x; for(auto& x:s.gyro)std::cin>>x;
   go2_moe::Action a;for(auto& x:a)std::cin>>x;float vx;std::cin>>vx;
   auto obs=history.append(go2_moe::frame(s,vx,a));
   for(auto x:obs)std::cout<<x<<' ';std::cout<<'\n';
  } else if(mode=='n') {
   int rejected=0;
   for(float bad:{NAN,INFINITY,-INFINITY}) {
    go2_moe::Action a{};a[0]=bad;
    try{go2_moe::decode(a);}catch(const std::exception&){++rejected;}
   }
   std::cout<<rejected<<'\n';
  }
 }
}
