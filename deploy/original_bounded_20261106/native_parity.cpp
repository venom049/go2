// File-only ARM parity runner. Never links or initializes DDS.
#include "MoeRunner.h"
#include <fstream>
#include <iostream>
int main(int argc,char** argv) {
    try {
        if(argc!=4) throw std::runtime_error("Expected bundle, input float32 file, new output file");
        if(std::filesystem::exists(argv[3])) throw std::runtime_error("Output already exists");
        const auto bytes=std::filesystem::file_size(argv[2]);
        if(!bytes || bytes%(225*sizeof(float))) throw std::runtime_error("Malformed observation fixtures");
        go2_moe::Runner runner(argv[1]);std::ifstream in(argv[2],std::ios::binary);
        std::ofstream out(argv[3],std::ios::binary);size_t count=0;
        go2_moe::Observation obs;
        while(in.read(reinterpret_cast<char*>(obs.data()),sizeof(obs))) {
            auto a=runner.infer(obs);out.write(reinterpret_cast<const char*>(a.data()),sizeof(a));++count;
        }
        if(count!=bytes/sizeof(obs) || !out) throw std::runtime_error("Incomplete fixture IO");
        std::cout<<"native_parity_cases="<<count<<" DDS_initialized=false motor_commands_sent=false\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
