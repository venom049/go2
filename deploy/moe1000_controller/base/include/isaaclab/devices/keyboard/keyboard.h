#pragma once

#include <string>
#include <deque>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <iostream>
#include <mutex>
#include <termios.h>
#include <unistd.h>
#include <thread>


/**
 * @brief Maintain a keyboard reading thread.
 * And get the latest key value.
 */
class Keyboard
{
public:
  Keyboard()
  {
    if(!isatty(fileno(stdin)) || tcgetattr(fileno(stdin), &_oldSettings) != 0)
    {
      return;
    }

    _available = true;
    _newSettings = _oldSettings;
    _newSettings.c_lflag &= (~ICANON & ~ECHO);
    _newSettings.c_cc[VMIN] = 0;
    _newSettings.c_cc[VTIME] = 1;

    _startKey();

    _thread_running = true;
    _readThread = std::thread([this] {
      while (_thread_running) {
        if(_running) {
          _read();
        } else {
          std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
      }
    });
  }

  ~Keyboard()
  {
    _thread_running = false;
    _running = false;
    if(_readThread.joinable()) {
      _readThread.join();
    }
    if(_available) {
      tcsetattr(fileno(stdin), TCSANOW, &_oldSettings);
    }
  }

  void update()
  {
    const auto current_key = key();
    if(current_key != _last_key)
    {
      on_pressed = !current_key.empty();
      on_released = current_key.empty();
    }
    else
    {
      on_pressed = false;
      on_released = false;
    }
    
    _last_key = current_key;
  }

  /**
   * @brief Get the current key value
   * 
   * @return std::string 
   */
  std::string key() const
  {
    std::lock_guard<std::mutex> lock(_mutex);
    return _key;
  }

  bool pop_key(std::string& key)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    if(_events.empty()) {
      return false;
    }
    key = std::move(_events.front());
    _events.pop_front();
    return true;
  }

  void clear_events()
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _events.clear();
  }

  bool available() const { return _available; }

  /**
   * @brief Get the String object from keyboard 
   * 
   * @param slogan Used to prompt the user for input
   * @return std::string 
   */
  std::string getString(std::string slogan)
  {
    // Stop reading keyboard value
    _running = false;
    _pauseKey();

    std::string stringtemp;
    std::cout << slogan << std::endl;// prompt
    std::getline(std::cin, stringtemp);

    // Restart reading keyboard value
    _startKey();
    _running = true;

    return stringtemp;
  }

  /**
   * flags; available after update()
   */
  bool on_pressed = false;
  bool on_released = false;

  private:
  std::atomic<bool> _thread_running{false};
  std::atomic<bool> _running{false};
  bool _available = false;
  std::thread _readThread;

  void _setKey(const std::string& key, bool enqueue)
  {
    std::lock_guard<std::mutex> lock(_mutex);
    _key = key;
    if(enqueue && !key.empty()) {
      constexpr std::size_t max_events = 64;
      if(_events.size() == max_events) {
        _events.pop_front();
      }
      _events.push_back(key);
    }
  }

  void _read()
  {
    FD_ZERO(&_fd_set);
    FD_SET(fileno(stdin), &_fd_set);

    _tv.tv_sec = 0;
    _tv.tv_usec = 80000;

    if(select(fileno(stdin) + 1, &_fd_set, NULL, NULL, &_tv) > 0)
    {
      char value = '\0';
      if(read(fileno(stdin), &value, 1) != 1)
      {
        _setKey("", false);
        return;
      }

      if(value != '\033')
      {
        _setKey(std::string(1, value), true);
        return;
      }

      char bracket = '\0';
      char direction = '\0';
      if(read(fileno(stdin), &bracket, 1) == 1 && bracket == '[' &&
         read(fileno(stdin), &direction, 1) == 1)
      {
        switch(direction)
        {
          case 'A': _setKey("up", true); return;
          case 'B': _setKey("down", true); return;
          case 'C': _setKey("right", true); return;
          case 'D': _setKey("left", true); return;
          default: break;
        }
      }
      _setKey("", false);
    }
    else
    {
      _setKey("", false);
    }
  }

  /**
   * @brief Restore keyboard default settings.
   */
  void _pauseKey()
  {
    _running = false;
    if(_available) {
      tcsetattr(fileno(stdin), TCSANOW, &_oldSettings);
    }
  }

  /**
   * @brief Disable canonical mode and echoing of input characters.
   */
  void _startKey()
  {
    if(_available) {
      tcsetattr(fileno(stdin), TCSANOW, &_newSettings);
      _running = true;
    }
  }

  mutable std::mutex _mutex;
  std::deque<std::string> _events;
  fd_set _fd_set;
  std::string _key, _last_key;

  termios _oldSettings{}, _newSettings{};
  timeval _tv{};
};
