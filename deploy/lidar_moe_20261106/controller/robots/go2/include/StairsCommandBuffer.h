#pragma once

#include <algorithm>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>

class StairsCommandBuffer
{
public:
    void configure(float min_speed, float max_speed, float speed_step,
                   float initial_speed)
    {
        if(min_speed <= 0.0f || max_speed < min_speed || speed_step <= 0.0f)
        {
            throw std::runtime_error("Invalid stair command buffer range.");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        if(configured_)
        {
            return;
        }

        min_speed_ = min_speed;
        max_speed_ = max_speed;
        speed_step_ = speed_step;
        selected_speed_ = round_speed(
            std::clamp(initial_speed, min_speed_, max_speed_)
        );
        configured_ = true;
    }

    bool select(const std::string& key)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if(!configured_)
        {
            return false;
        }

        float selected = selected_speed_;
        if(key == "w" || key == "up")
        {
            selected = std::min(max_speed_, selected_speed_ + speed_step_);
        }
        else if(key == "s" || key == "down")
        {
            selected = std::max(min_speed_, selected_speed_ - speed_step_);
        }
        else if(key.size() == 1 && key[0] >= '1' && key[0] <= '9')
        {
            selected = min_speed_ + (key[0] - '1') * speed_step_;
            if(selected > max_speed_ + 1.0e-4f)
            {
                return false;
            }
        }
        else
        {
            return false;
        }

        selected_speed_ = round_speed(selected);
        armed_ = true;
        return true;
    }

    void begin_selection()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        armed_ = false;
    }

    bool armed() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return armed_;
    }

    float selected_speed() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return selected_speed_;
    }

    void set_auto_zero_seconds(double seconds)
    {
        if(seconds < 0.0)
        {
            throw std::runtime_error("auto zero seconds must be non-negative.");
        }

        std::lock_guard<std::mutex> lock(mutex_);
        auto_zero_seconds_ = seconds;
        auto_zero_reported_ = false;
    }

    void reset_auto_zero_report()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto_zero_reported_ = false;
    }

    bool auto_zero_due(long episode_length, float step_dt) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if(auto_zero_seconds_ <= 0.0 || episode_length < 0 || step_dt <= 0.0f)
        {
            return false;
        }
        return static_cast<double>(episode_length) *
            static_cast<double>(step_dt) >= auto_zero_seconds_;
    }

    bool consume_auto_zero_report()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if(auto_zero_seconds_ <= 0.0 || auto_zero_reported_)
        {
            return false;
        }
        auto_zero_reported_ = true;
        return true;
    }

private:
    static float round_speed(float speed)
    {
        return std::round(speed * 100.0f) / 100.0f;
    }

    mutable std::mutex mutex_;
    bool configured_ = false;
    bool armed_ = false;
    float min_speed_ = 0.5f;
    float max_speed_ = 0.8f;
    float speed_step_ = 0.1f;
    float selected_speed_ = 0.5f;
    double auto_zero_seconds_ = 0.0;
    bool auto_zero_reported_ = false;
};

inline StairsCommandBuffer stairs_command_buffer;
