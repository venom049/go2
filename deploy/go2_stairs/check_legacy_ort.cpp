// Offline compatibility check only: no Unitree SDK, DDS, or motor interface.
// Usage: check_legacy_ort model.onnx inputs.float32 expected.float32
// Fixtures are little-endian float32, shapes [N,294] and [N,16].
#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr std::size_t kInputs = 294;
constexpr std::size_t kOutputs = 16;
constexpr std::size_t kMaxSamples = 10000;
constexpr std::size_t kWarmup = 20;
constexpr std::size_t kTimedRuns = 100;
constexpr double kAbsTolerance = 5e-5;
constexpr double kRelTolerance = 1e-5;
constexpr double kBudgetMs = 20.0;

std::string json_string(const std::string& value) {
    std::string result = "\"";
    constexpr char hex[] = "0123456789abcdef";
    for (unsigned char c : value) {
        if (c == '\\' || c == '"') {
            result += '\\';
            result += static_cast<char>(c);
        } else if (c < 0x20) {
            result += "\\u00";
            result += hex[c >> 4];
            result += hex[c & 15];
        } else {
            result += static_cast<char>(c);
        }
    }
    return result + '"';
}

std::vector<float> read_fixture(const char* path, std::size_t width) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Cannot open fixture file.");
    const auto end = file.tellg();
    if (end <= 0) throw std::runtime_error("Fixture is empty or unreadable.");
    const auto bytes = static_cast<std::uint64_t>(end);
    const auto row_bytes = width * sizeof(float);
    if (bytes % row_bytes != 0 || bytes / row_bytes > kMaxSamples) {
        throw std::runtime_error("Fixture must contain 1..10000 complete rows.");
    }
    std::vector<float> values(static_cast<std::size_t>(bytes / sizeof(float)));
    file.seekg(0);
    if (!file.read(reinterpret_cast<char*>(values.data()),
                   static_cast<std::streamsize>(bytes))) {
        throw std::runtime_error("Fixture read was incomplete.");
    }
    if (file.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("Fixture changed size while reading.");
    }
    for (float value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error("Fixture contains NaN or infinity.");
        }
    }
    return values;
}

void require_tensor(const Ort::TypeInfo& info,
                    const std::vector<std::int64_t>& shape,
                    const char* label) {
    if (info.GetONNXType() != ONNX_TYPE_TENSOR) {
        throw std::runtime_error(std::string(label) + " must be a tensor.");
    }
    const auto tensor_info = info.GetTensorTypeAndShapeInfo();
    if (tensor_info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        tensor_info.GetShape() != shape) {
        throw std::runtime_error(std::string(label) +
                                 " has an incompatible dtype or fixed shape.");
    }
}

std::array<float, kOutputs> infer(Ort::Session& session,
                                const Ort::MemoryInfo& memory,
                                float* row,
                                const char* output_name) {
    constexpr std::array<std::int64_t, 2> shape{1, kInputs};
    const char* input_name = "obs";
    auto input = Ort::Value::CreateTensor<float>(
        memory, row, kInputs, shape.data(), shape.size());
    auto results = session.Run(Ort::RunOptions{nullptr}, &input_name, &input, 1,
                               &output_name, 1);
    if (results.size() != 1 || !results.front().IsTensor()) {
        throw std::runtime_error("Inference did not return one output tensor.");
    }
    const auto info = results.front().GetTensorTypeAndShapeInfo();
    if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        info.GetShape() != std::vector<std::int64_t>{1, kOutputs} ||
        info.GetElementCount() != kOutputs) {
        throw std::runtime_error("Runtime output is not float32 [1,16].");
    }
    const float* data = results.front().GetTensorData<float>();
    std::array<float, kOutputs> output{};
    for (std::size_t i = 0; i < kOutputs; ++i) {
        if (!std::isfinite(data[i])) {
            throw std::runtime_error("Inference output contains NaN or infinity.");
        }
        output[i] = data[i];
    }
    return output;
}
}  // namespace

int main(int argc, char** argv) {
    const char* runtime_version = OrtGetApiBase()->GetVersionString();
    try {
        if (argc != 4) {
            throw std::runtime_error(
                "Usage: check_legacy_ort model.onnx inputs.float32 expected.float32");
        }
        const std::uint16_t endian_probe = 1;
        if (sizeof(float) != 4 || !std::numeric_limits<float>::is_iec559 ||
            *reinterpret_cast<const unsigned char*>(&endian_probe) != 1) {
            throw std::runtime_error("Fixtures require little-endian IEEE float32.");
        }
        auto inputs = read_fixture(argv[2], kInputs);
        const auto expected = read_fixture(argv[3], kOutputs);
        const std::size_t count = inputs.size() / kInputs;
        if (expected.size() / kOutputs != count) {
            throw std::runtime_error("Input and expected sample counts differ.");
        }

        Ort::Env environment(ORT_LOGGING_LEVEL_WARNING, "offline_legacy_check");
        Ort::SessionOptions options;
        // Match the existing controller's OrtRunner; CPU provider and default threads.
        options.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        Ort::Session session(environment, argv[1], options);
        if (session.GetInputCount() != 1 || session.GetOutputCount() != 1) {
            throw std::runtime_error("Model must have exactly one input and output.");
        }
        Ort::AllocatorWithDefaultOptions allocator;
        const auto input_name = session.GetInputNameAllocated(0, allocator);
        const auto output_name = session.GetOutputNameAllocated(0, allocator);
        if (std::string(input_name.get()) != "obs") {
            throw std::runtime_error("Model input must be named obs.");
        }
        require_tensor(session.GetInputTypeInfo(0), {1, kInputs}, "Input");
        require_tensor(session.GetOutputTypeInfo(0), {1, kOutputs}, "Output");
        const auto memory = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator,
                                                      OrtMemTypeCPU);

        double max_abs_error = 0.0;
        double max_tolerance_ratio = 0.0;
        std::size_t mismatched_values = 0;
        for (std::size_t sample = 0; sample < count; ++sample) {
            const auto output = infer(session, memory, inputs.data() + sample * kInputs,
                                      output_name.get());
            for (std::size_t i = 0; i < kOutputs; ++i) {
                const double reference = expected[sample * kOutputs + i];
                const double error = std::abs(static_cast<double>(output[i]) - reference);
                const double tolerance = kAbsTolerance + kRelTolerance * std::abs(reference);
                max_abs_error = std::max(max_abs_error, error);
                max_tolerance_ratio = std::max(max_tolerance_ratio, error / tolerance);
                if (!(error < tolerance)) ++mismatched_values;
            }
        }

        for (std::size_t i = 0; i < kWarmup; ++i) {
            infer(session, memory, inputs.data() + (i % count) * kInputs,
                  output_name.get());
        }
        std::vector<double> timings;
        timings.reserve(kTimedRuns);
        for (std::size_t i = 0; i < kTimedRuns; ++i) {
            const auto started = std::chrono::steady_clock::now();
            infer(session, memory, inputs.data() + (i % count) * kInputs,
                  output_name.get());
            const auto stopped = std::chrono::steady_clock::now();
            timings.push_back(std::chrono::duration<double, std::milli>(stopped - started).count());
        }
        std::sort(timings.begin(), timings.end());
        // Nearest-rank percentiles, with 1-based ranks 50 and 95 for 100 samples.
        const double p50 = timings[49];
        const double p95 = timings[94];
        const double maximum = timings.back();
        const bool validation_passed = mismatched_values == 0;
        const bool budget_passed = maximum < kBudgetMs;
        std::cout << std::setprecision(12) << std::boolalpha
                  << "{\"ok\":" << (validation_passed && budget_passed)
                  << ",\"offline_only\":true,\"runtime_version\":" << json_string(runtime_version)
                  << ",\"samples_verified\":" << count
                  << ",\"input_shape\":[1,294],\"output_shape\":[1,16]"
                  << ",\"output_name\":" << json_string(output_name.get())
                  << ",\"validation_passed\":" << validation_passed
                  << ",\"mismatched_values\":" << mismatched_values
                  << ",\"max_abs_error\":" << max_abs_error
                  << ",\"max_tolerance_ratio\":" << max_tolerance_ratio
                  << ",\"abs_tolerance\":" << kAbsTolerance
                  << ",\"rel_tolerance\":" << kRelTolerance
                  << ",\"timing\":{\"warmup_runs\":" << kWarmup
                  << ",\"measured_runs\":" << kTimedRuns
                  << ",\"p50_ms\":" << p50 << ",\"p95_ms\":" << p95
                  << ",\"max_ms\":" << maximum << ",\"budget_ms\":" << kBudgetMs
                  << ",\"all_runs_within_budget\":" << budget_passed
                  << ",\"scope\":\"tensor_creation+Run+output_validation\"}"
                  << ",\"mechanical_safety_validated\":false}\n";
        if (!validation_passed) return 2;
        return budget_passed ? 0 : 3;
    } catch (const std::exception& error) {
        std::cout << "{\"ok\":false,\"offline_only\":true,\"runtime_version\":"
                  << json_string(runtime_version) << ",\"error\":"
                  << json_string(error.what()) << "}\n";
        return 1;
    }
}
