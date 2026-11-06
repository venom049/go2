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
    "/home/unitree/go2-moe1000-20261104";
inline constexpr const char* kOnnxSha256 =
    "f0bfb5ac6758b72fbeae96f1b82462950127440b5210807016d5fa5466529d56";
inline constexpr const char* kConfigSha256 =
    "7d04bfd00a60cf858572597667d9f33e78b94f71dca2b101104162caef0aca00";
inline constexpr const char* kControllerConfigSha256 =
    "ee56644806f2bb38c5fd5061be56823dc68d746076e6b2bf1d9c01e534aa8cff";

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
        sha256_file(directory / "limit_profile.json") == "b62566e6bc95385b15a0423a449b3d033c5e76f77a7ce650ba0a2c112e9585f0";
}
} // namespace go2_moe
