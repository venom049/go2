#include "MoePolicy.h"
#include "PerceptionCommand.h"
#include <cassert>
#include <iostream>
#include <random>
#include <limits>
#include <sys/stat.h>
#include <thread>
using namespace go2_moe;
go2_first_test::GuardInput state(){go2_first_test::GuardInput s;s.state_available=true;s.quaternion_wxyz={1,0,0,0};s.q=stand_sdk();s.tick=1;return s;}
std::array<unsigned char,64> packet(double stamp=10.,float vx=.2f,uint64_t seq=1,uint32_t mode=1) {
 std::array<unsigned char,64>b{};memcpy(b.data(),"G2L2",4);uint32_t ver=2;uint64_t ns=uint64_t(stamp*1e9);float f[4]={vx,0,0,1};memcpy(b.data()+4,&ver,4);memcpy(b.data()+8,&seq,8);memcpy(b.data()+16,&ns,8);memcpy(b.data()+24,f,16);memcpy(b.data()+40,&mode,4);uint32_t healthy=1;uint64_t session=1;memcpy(b.data()+44,&healthy,4);memcpy(b.data()+56,&session,8);return b;
}
int main(){auto s=state();auto f=frame(s,std::array<float,3>{.25f,-.06f,.20f},{});assert(f[6]==.5f&&f[7]==-.12f&&f[8]==.05f);auto old=frame(s,.6f,{});auto same=frame(s,std::array<float,3>{.6f,0,0},{});assert(old==same);
 std::mt19937 gen(11);std::uniform_real_distribution<float>dist(-100,100);
 for(int k=0;k<20000;++k){Action a;for(auto& x:a)x=dist(gen);auto c=decode(a);for(size_t i=0;i<12;++i){assert(c.q[i]>=go2_first_test::kTargetLower[i]);assert(c.q[i]<=go2_first_test::kTargetUpper[i]);}}
 go2_first_test::Guard g;g.configure_motion_budget(20.);auto c=decode(Action{});g.filter(s,c,0.,true);for(int i=1;i<10000;++i){++s.tick;assert(g.filter(s,c,i*.002,true).command_allowed);}++s.tick;assert(g.filter(s,c,20.,true).reason=="lidar_motion_budget_complete");
 go2_first_test::Guard parked;parked.configure_motion_budget(20.);s=state();parked.filter(s,c,0.,true);++s.tick;assert(parked.filter(s,c,.002,true).command_allowed);parked.finish_motion();for(int i=2;i<11000;++i){++s.tick;assert(parked.filter(s,c,i*.002,false).command_allowed);}++s.tick;assert(parked.filter(s,c,22.,true).reason=="policy_reentry_after_park");
 go2_first_test::Guard tracking;s=state();auto far=c;far.kp.fill(.001);far.q[1]+=2.01;assert(tracking.filter(s,far,0.,true).reason=="tracking_error_above_2rad");
 auto b=packet();assert(go2_lidar::decode(b.data(),64,10.1));assert(!go2_lidar::decode(b.data(),43,10.1));assert(!go2_lidar::decode(b.data(),64,10.301));assert(!go2_lidar::decode(b.data(),64,9.97));
 for(auto x:{.36f,-.01f,std::numeric_limits<float>::quiet_NaN()}){b=packet(10.,x);assert(!go2_lidar::decode(b.data(),64,10.));}b=packet(10.,.2f,1,2);assert(!go2_lidar::decode(b.data(),64,10.));
 {
  const auto path=std::string("/tmp/go2-lidar-socket-")+std::to_string(getpid());
  go2_lidar::CommandSocket socket(path);int fd=::socket(AF_UNIX,SOCK_DGRAM,0);assert(fd>=0);
  sockaddr_un addr{};addr.sun_family=AF_UNIX;strcpy(addr.sun_path,path.c_str());
  b=packet(10.,.2f,1);assert(sendto(fd,b.data(),b.size(),0,reinterpret_cast<sockaddr*>(&addr),sizeof(addr))==64);
  socket.poll(10.1,0);assert(socket.ready(10.1));
  b=packet(10.2,.3f,1);sendto(fd,b.data(),b.size(),0,reinterpret_cast<sockaddr*>(&addr),sizeof(addr));socket.poll(10.2,0);
  assert(socket.command().vx==.2f);assert(!socket.fresh(10.301)); // replay never refreshes.
  b=packet(10.2,.2f,2);sendto(fd,b.data(),b.size(),0,reinterpret_cast<sockaddr*>(&addr),sizeof(addr));socket.poll(10.25,0);assert(socket.ready(10.25));
  struct stat st{};assert(stat(path.c_str(),&st)==0 && (st.st_mode&0777)==0600);close(fd);
 }

 {
  go2_lidar::Qualification q;const uint64_t epoch=10000000000;q.begin(epoch);
  go2_lidar::Command c;c.epoch=epoch;c.session=1;c.healthy=1;
  for(int i=0;i<80;++i){c.stamp=10.+i*.05;assert(!q.update(c.stamp,c,true));}
  c.stamp=14.;assert(q.update(14.,c,true));
  assert(!q.update(14.05,c,false));c.stamp=14.1;assert(!q.update(14.1,c,true));
  c.stamp=18.2;assert(!q.update(18.2,c,true)); // skipped checks are not continuous health.
  for(int i=1;i<=81;++i){c.stamp=18.2+i*.05;q.update(c.stamp,c,true);}assert(q.qualified());
  c.session=2;c.stamp=22.3;assert(!q.update(22.3,c,true));
  for(int i=1;i<=81;++i){c.stamp=22.3+i*.05;q.update(c.stamp,c,true);}assert(q.qualified());
  c.epoch=0;c.stamp=26.4;assert(!q.update(26.4,c,true));
  q.begin(epoch);c.epoch=epoch;c.stamp=9.99;assert(!q.update(10.,c,true));
 }
 {
  const auto path=std::string("/tmp/g2-query-")+std::to_string(getpid());
  const auto client=path+"-client";go2_lidar::CommandSocket server(path);
  int fd=::socket(AF_UNIX,SOCK_DGRAM,0);assert(fd>=0);sockaddr_un peer{};peer.sun_family=AF_UNIX;strcpy(peer.sun_path,client.c_str());
  assert(bind(fd,reinterpret_cast<sockaddr*>(&peer),sizeof(peer))==0);
  sockaddr_un target{};target.sun_family=AF_UNIX;strcpy(target.sun_path,path.c_str());
  auto send=[&](const void* bytes,size_t size){assert(sendto(fd,bytes,size,0,reinterpret_cast<sockaddr*>(&target),sizeof(target))==ssize_t(size));};
  b=packet(10.,.2f,1);send(b.data(),b.size());server.poll(10.1,3);assert(server.ready(10.1));
  unsigned char reply[128];assert(recv(fd,reply,sizeof(reply),MSG_DONTWAIT)==28);
  unsigned char query[8]{};memcpy(query,"G2Q2",4);uint32_t ver=2;memcpy(query+4,&ver,4);
  send(query,8);server.poll(10.299,3);assert(recv(fd,reply,sizeof(reply),MSG_DONTWAIT)==28);
  assert(server.command().sequence==1 && !server.fresh(10.301)); // query never refreshes.
  server.set_epoch(10000000000ULL);assert(!server.ready(10.2));
  b=packet(10.2,.2f,2);send(b.data(),b.size());server.poll(10.25,3);assert(!server.ready(10.25)); // old epoch rejected.
  uint64_t epoch=10000000000ULL;memcpy(b.data()+48,&epoch,8);send(b.data(),b.size());server.poll(10.25,3);assert(server.ready(10.25));
  send("bad",3);server.poll(10.26,1);assert(!server.fresh(10.26));
  b=packet(10.27,.2f,3);memcpy(b.data()+48,&epoch,8);send(b.data(),b.size());server.poll(10.28,1);assert(!server.ready(10.28)); // no automatic motion recovery.
  close(fd);unlink(client.c_str());
 }
 std::cout<<"vector command parity; 240000 hard-clipped joint targets; 20s deadline; park/no-reentry; 2rad tracking; packet stale/future/NaN/bounds: PASS\n";
}
