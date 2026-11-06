#pragma once
// One producer (FSM) and one consumer (main). No filesystem I/O on the FSM thread.
#include <array>
#include <atomic>
#include <fstream>
#include <iomanip>
#include <string>
#include <stdexcept>
namespace go2_lidar {
class PolicyTrace {
public:
    static constexpr size_t kColumns=1+1+3+12+12+12+12+225;
    using Row=std::array<double,kColumns>;
    explicit PolicyTrace(const std::string& path):out_(path,std::ios::out|std::ios::app) {
        if(!out_) throw std::runtime_error("Cannot open policy trace");
        out_<<"monotonic_s,command_sequence,vx,vy,wz";
        for(const char* name:{"q","dq","raw_action","proposed_target"})
            for(size_t i=0;i<12;++i) out_<<','<<name<<'_'<<i;
        for(size_t i=0;i<225;++i) out_<<",obs_"<<i;
        out_<<'\n'<<std::setprecision(10);out_.flush();
    }
    void push(const Row& row) noexcept {
        auto h=head_.load(std::memory_order_relaxed);auto next=(h+1)%rows_.size();
        if(next==tail_.load(std::memory_order_acquire)) {dropped_.fetch_add(1);return;}
        rows_[h]=row;head_.store(next,std::memory_order_release);
    }
    void flush() {
        auto t=tail_.load(std::memory_order_relaxed);
        const auto h=head_.load(std::memory_order_acquire);
        while(t!=h) {const auto& row=rows_[t];for(size_t i=0;i<row.size();++i) out_<<(i?",":"")<<row[i];out_<<'\n';t=(t+1)%rows_.size();}
        tail_.store(t,std::memory_order_release);out_.flush();
        if(!out_) failed_.store(true);
    }
    bool failed() const {return failed_.load();}
    size_t dropped() const {return dropped_.load();}
private:
    std::ofstream out_;std::array<Row,256> rows_{};
    std::atomic<size_t> head_{0},tail_{0},dropped_{0};std::atomic<bool> failed_{false};
};
}
