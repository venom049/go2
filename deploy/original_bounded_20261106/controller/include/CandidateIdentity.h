#pragma once

// File-only identity verification. No DDS, network, mode or motor operations.
#include <cerrno>
#include <filesystem>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

extern char **environ;

namespace go2_moe {
inline constexpr const char* kCandidateDirectory =
    "/home/unitree/go2-original-bounded-20261106/model";
inline constexpr const char* kOnnxSha256 =
    "bc08253fd68f661bf6be5ce5ca70cec555c3fde2c397d23fbe9814a998e056f5";
inline constexpr const char* kConfigSha256 =
    "7d04bfd00a60cf858572597667d9f33e78b94f71dca2b101104162caef0aca00";
inline constexpr const char* kControllerConfigSha256 =
    "a63dc32d00675ad41204e10eefacc401df1595507772c7f60a45ed641e722608";

inline std::string sha256_file(const std::filesystem::path& path) {
    if (std::filesystem::is_symlink(path) || !std::filesystem::is_regular_file(path))
        throw std::runtime_error("Candidate identity requires regular non-symlink files");
    int pipefd[2];
    if (pipe(pipefd) != 0) throw std::runtime_error("Cannot create hash reader pipe");
    posix_spawn_file_actions_t actions;
    if (posix_spawn_file_actions_init(&actions) != 0) {
        close(pipefd[0]); close(pipefd[1]);
        throw std::runtime_error("Cannot initialize hash reader");
    }
    const int a = posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO);
    const int b = posix_spawn_file_actions_addclose(&actions, pipefd[0]);
    const int c = posix_spawn_file_actions_addclose(&actions, pipefd[1]);
    const std::string filename = path.string();
    char binary[] = "/usr/bin/sha256sum", separator[] = "--";
    char* argv[] = {binary, separator, const_cast<char*>(filename.c_str()), nullptr};
    pid_t child = -1;
    const int rc = (a || b || c) ? -1 :
        posix_spawn(&child, binary, &actions, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(pipefd[1]);
    if (rc != 0) {
        close(pipefd[0]);
        throw std::runtime_error("Cannot execute hash reader");
    }
    std::string output;
    char buffer[512];
    bool read_ok = true;
    for (;;) {
        const auto count = read(pipefd[0], buffer, sizeof(buffer));
        if (count == 0) break;
        if (count < 0) {
            if (errno == EINTR) continue;
            read_ok = false; break;
        }
        if (output.size() + static_cast<std::size_t>(count) > 4096) {
            read_ok = false; break;
        }
        output.append(buffer, static_cast<std::size_t>(count));
    }
    close(pipefd[0]);
    int status = 0;
    pid_t waited;
    do { waited = waitpid(child, &status, 0); } while (waited < 0 && errno == EINTR);
    if (!read_ok || waited != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        output.size() < 66 || output[64] != ' ')
        throw std::runtime_error("Candidate hash reader failed");
    return output.substr(0, 64);
}

inline bool verified_candidate(const std::filesystem::path& directory) {
    return directory.is_absolute() && directory.lexically_normal() == kCandidateDirectory &&
        sha256_file(directory / "exported/policy.onnx") == kOnnxSha256 &&
        sha256_file(directory / "params/deploy.yaml") == kConfigSha256 &&
        sha256_file(directory / "limit_profile.json") == "bd3ea1830ca2ff7d8eb6eda1fb0b5077e33192a1ee663f2c16f96195d948c8af";
}
} // namespace go2_moe
