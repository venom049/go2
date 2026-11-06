#pragma once
#include "MoePolicy.h"
#include "CandidateIdentity.h"
#include <onnxruntime_cxx_api.h>
#include <memory>
#include <vector>

namespace go2_moe {
class Runner {
public:
    explicit Runner(const std::filesystem::path& bundle) {
        if(!verified_candidate(bundle)) throw std::runtime_error("MoE pinned artifact identity mismatch");
        options_.SetIntraOpNumThreads(1);options_.SetInterOpNumThreads(1);
        options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_=std::make_unique<Ort::Session>(env_,(bundle/"exported/policy.onnx").c_str(),options_);
        if(session_->GetInputCount()!=1) throw std::runtime_error("MoE input count mismatch");
        Ort::AllocatorWithDefaultOptions allocator;
        auto name=session_->GetInputNameAllocated(0,allocator);
        auto ti=session_->GetInputTypeInfo(0);
        if(std::string(name.get())!="obs" || ti.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,225}
           || ti.GetTensorTypeAndShapeInfo().GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
            throw std::runtime_error("MoE observation contract mismatch");
        bool found=false;
        for(size_t i=0;i<session_->GetOutputCount();++i) {
            auto n=session_->GetOutputNameAllocated(i,allocator);
            if(std::string(n.get())!="actions") continue;
            auto type=session_->GetOutputTypeInfo(i);
            if(type.GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,12}
               || type.GetTensorTypeAndShapeInfo().GetElementType()!=ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
                throw std::runtime_error("MoE action contract mismatch");
            found=true;
        }
        if(!found) throw std::runtime_error("MoE named actions output missing");
    }
    Action infer(Observation obs) {
        for(float v:obs) finite(v);
        auto mem=Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,OrtMemTypeDefault);
        const std::array<int64_t,2> shape{1,225};
        auto input=Ort::Value::CreateTensor<float>(mem,obs.data(),obs.size(),shape.data(),shape.size());
        const char* in="obs";const char* out="actions";
        auto output=session_->Run(Ort::RunOptions{nullptr},&in,&input,1,&out,1);
        if(output.size()!=1 || !output[0].IsTensor() ||
           output[0].GetTensorTypeAndShapeInfo().GetShape()!=std::vector<int64_t>{1,12})
            throw std::runtime_error("MoE runtime action shape mismatch");
        Action result{};std::copy_n(output[0].GetTensorData<float>(),12,result.begin());
        for(float v:result) finite(v);
        return result;
    }
private:
    Ort::Env env_{ORT_LOGGING_LEVEL_WARNING,"moe1000"};
    Ort::SessionOptions options_;
    std::unique_ptr<Ort::Session> session_;
};
}
