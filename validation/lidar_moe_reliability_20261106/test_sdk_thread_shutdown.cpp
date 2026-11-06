// No DDS, subscriptions or publishers; reproduces only SDK thread ownership.
#include <unitree/common/thread/recurrent_thread.hpp>
#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <iostream>
int main(){
 for(int i=0;i<50;++i){
  std::atomic<unsigned> calls{0};
  auto worker=std::make_shared<unitree::common::RecurrentThread>("test",0,1000,[&](){++calls;});
  std::this_thread::sleep_for(std::chrono::milliseconds(4));
  worker.reset();
  const auto stopped=calls.load();std::this_thread::sleep_for(std::chrono::milliseconds(2));
  if(calls.load()!=stopped)return 2;
 }
 std::cout<<"SDK recurrent thread shutdown PASS; DDS initialized=false\n";
}
