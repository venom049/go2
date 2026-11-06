#pragma once
// No DDS calls. Epochs prevent pre-handoff data and sensor restarts authorizing motion.
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
namespace go2_lidar {
inline double now_seconds(){return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();}
struct Command {
    std::uint64_t sequence=0;double stamp=0.;float vx=0,vy=0,wz=0;
    std::uint32_t mode=0,healthy=0;std::uint64_t epoch=0,session=0;
};
inline std::optional<Command> decode(const unsigned char* data,size_t size,double now){
    if(size!=64 || std::memcmp(data,"G2L2",4)!=0)return {};
    std::uint32_t version,mode,healthy;std::uint64_t sequence,stamp,epoch,session;float v[4];
    std::memcpy(&version,data+4,4);std::memcpy(&sequence,data+8,8);std::memcpy(&stamp,data+16,8);
    std::memcpy(v,data+24,16);std::memcpy(&mode,data+40,4);std::memcpy(&healthy,data+44,4);
    std::memcpy(&epoch,data+48,8);std::memcpy(&session,data+56,8);const double source=stamp*1e-9;
    if(version!=2 || !sequence || !session || mode>3 || healthy>1 || !std::isfinite(now) || now-source<-.02 || now-source>.30)return {};
    for(float f:v)if(!std::isfinite(f))return {};
    if(v[0]<0 || v[0]>.35f || std::abs(v[1])>.10f || std::abs(v[2])>.35f || v[3]<0 || v[3]>1)return {};
    if(mode!=1 && (v[0]!=0 || v[1]!=0 || v[2]!=0))return {};
    if((mode==1 || mode==2) && v[3]!=1.f)return {};
    return Command{sequence,source,v[0],v[1],v[2],mode,healthy,epoch,session};
}
class CommandSocket {
public:
    explicit CommandSocket(std::string path):path_(std::move(path)){
        if(path_.size()>=sizeof(sockaddr_un::sun_path) || std::filesystem::exists(path_))throw std::runtime_error("Perception socket unavailable; inspect previous session");
        fd_=socket(AF_UNIX,SOCK_DGRAM|SOCK_NONBLOCK|SOCK_CLOEXEC,0);if(fd_<0)throw std::runtime_error("Cannot create perception socket");
        sockaddr_un a{};a.sun_family=AF_UNIX;std::strcpy(a.sun_path,path_.c_str());
        if(bind(fd_,reinterpret_cast<sockaddr*>(&a),sizeof(a))!=0){close(fd_);fd_=-1;throw std::runtime_error("Cannot bind perception socket");}chmod(path_.c_str(),0600);
    }
    ~CommandSocket(){if(fd_>=0){close(fd_);unlink(path_.c_str());}}
    CommandSocket(const CommandSocket&)=delete;
    void set_epoch(std::uint64_t epoch){epoch_=epoch;last_={};invalid_=false;}
    bool poll(double now,std::uint32_t active){
        for(int i=0;i<8;++i){
            unsigned char data[128];sockaddr_un peer{};socklen_t length=sizeof(peer);
            auto count=recvfrom(fd_,data,sizeof(data),MSG_DONTWAIT,reinterpret_cast<sockaddr*>(&peer),&length);if(count<0)break;
            std::uint32_t version=0;if(count==8)std::memcpy(&version,data+4,4);
            if(count==8 && !std::memcmp(data,"G2Q2",4) && version==2){send_feedback(peer,length,now,active);continue;}
            auto packet=decode(data,static_cast<size_t>(count),now);
            if(!packet){if(active==1)invalid_=true;else last_={};continue;}
            if(packet->sequence<=last_sequence_)continue;
            // An old epoch can remain queued briefly after the handoff; it
            // cannot refresh a command or revoke the new handoff identity.
            if(packet->epoch!=epoch_)continue;
            last_sequence_=packet->sequence;last_=*packet;peer_=peer;peer_length_=length;
        }
        if(peer_length_ && now-last_feedback_>=.05){send_feedback(peer_,peer_length_,now,active);last_feedback_=now;}
        return fresh(now);
    }
    bool fresh(double now)const{return !invalid_ && last_.sequence && last_.epoch==epoch_ && now-last_.stamp>=-.02 && now-last_.stamp<=.30;}
    bool healthy(double now)const{return fresh(now) && last_.healthy==1;}
    bool ready(double now)const{return healthy(now) && (last_.mode==1 || last_.mode==2);}
    const Command& command()const{return last_;}
private:
    void send_feedback(const sockaddr_un& peer,socklen_t length,double now,std::uint32_t active){
        unsigned char b[28]{};std::memcpy(b,"G2F2",4);std::uint32_t version=2;const auto stamp=static_cast<std::uint64_t>(now*1e9);
        std::memcpy(b+4,&version,4);std::memcpy(b+8,&stamp,8);std::memcpy(b+16,&active,4);std::memcpy(b+20,&epoch_,8);
        sendto(fd_,b,sizeof(b),MSG_DONTWAIT,reinterpret_cast<const sockaddr*>(&peer),length);
    }
    int fd_=-1;std::string path_;Command last_{};bool invalid_=false;
    std::uint64_t epoch_=0,last_sequence_=0;sockaddr_un peer_{};socklen_t peer_length_=0;double last_feedback_=0.;
};
class Qualification {
public:
    void begin(std::uint64_t epoch){epoch_=epoch;session_=0;since_=-1.;first_source_=0.;qualified_=false;last_poll_=-1.;last_source_=0.;}
    bool update(double now,const Command& c,bool healthy){
        if(last_poll_>=0. && (now<last_poll_ || now-last_poll_>.15 || c.stamp-last_source_>.35)) {since_=-1.;qualified_=false;}
        last_poll_=now;last_source_=c.stamp;
        if(!epoch_ || !healthy || c.epoch!=epoch_ || c.stamp<epoch_*1e-9 || !c.session){since_=-1.;qualified_=false;return false;}
        if(c.session!=session_){session_=c.session;since_=-1.;qualified_=false;}
        if(since_<0){since_=now;first_source_=c.stamp;}
        qualified_=now-since_>=4. && c.stamp-first_source_>=3.7;return qualified_;
    }
    bool qualified()const{return qualified_;}
private:
    std::uint64_t epoch_=0,session_=0;double since_=-1.,first_source_=0.,last_poll_=-1.,last_source_=0.;bool qualified_=false;
};
}
