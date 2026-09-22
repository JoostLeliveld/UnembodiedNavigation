#include <ignition/msgs/boolean.pb.h>
#include <ignition/msgs/world_control.pb.h>
#include <ignition/transport/Node.hh>

#include <geometry_msgs/msg/twist.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>

using namespace std::chrono_literals;

namespace
{
std::optional<std::string> JsonString(const std::string &json, const std::string &key)
{
  const std::regex expression("\\\"" + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
  std::smatch match;
  if (!std::regex_search(json, match, expression))
    return std::nullopt;
  return match[1].str();
}
}

class LockstepScheduler final : public rclcpp::Node
{
public:
  LockstepScheduler() : Node("lockstep_scheduler")
  {
    worldName_ = declare_parameter<std::string>("world_name", "warehouse_v2");
    stepIterations_ = declare_parameter<int>("control_step_iterations", 100);
    cameraEveryControlSteps_ = declare_parameter<int>("camera_every_control_steps", 2);
    timeoutS_ = declare_parameter<double>("barrier_timeout_s", 30.0);
    startupDelayS_ = declare_parameter<double>("startup_delay_s", 2.0);
    maxControlSteps_ = declare_parameter<int>("max_control_steps", 0);
    if (stepIterations_ <= 0 || cameraEveryControlSteps_ <= 0 || timeoutS_ <= 0.0)
      throw std::invalid_argument("lockstep scheduler parameters must be positive");

    detectorSub_ = create_subscription<std_msgs::msg::String>(
      "/perception/camera_batch_outcome", rclcpp::QoS(256).reliable(),
      [this](const std_msgs::msg::String &message) { OnDetector(message.data); });
    managerSub_ = create_subscription<std_msgs::msg::String>(
      "/reliability/camera_manager/batch_outcome", rclcpp::QoS(256).reliable(),
      [this](const std_msgs::msg::String &message) { OnManager(message.data); });
    odomSub_ = create_subscription<nav_msgs::msg::Odometry>(
      "/odom", rclcpp::QoS(100),
      [this](const nav_msgs::msg::Odometry &) { ++odomCount_; cv_.notify_all(); });
    commandSub_ = create_subscription<geometry_msgs::msg::Twist>(
      "/cmd_vel", rclcpp::QoS(100),
      [this](const geometry_msgs::msg::Twist &) { ++commandCount_; cv_.notify_all(); });

    worker_ = std::thread([this]() { Run(); });
  }

  ~LockstepScheduler() override
  {
    stopping_ = true;
    cv_.notify_all();
    if (worker_.joinable()) worker_.join();
  }

private:
  void OnDetector(const std::string &json)
  {
    const auto status = JsonString(json, "status");
    if (!status) return;
    std::lock_guard<std::mutex> lock(mutex_);
    if (*status == "session_started") detectorReady_ = true;
    if (*status == "published") {
      const auto id = JsonString(json, "source_batch_id");
      if (id && *id != lastDetectorBatch_) {
        lastDetectorBatch_ = *id;
        ++detectorPublishedCount_;
      }
    }
    cv_.notify_all();
  }

  void OnManager(const std::string &json)
  {
    const auto status = JsonString(json, "status");
    const auto id = JsonString(json, "source_batch_id");
    if (!status || !id || *status != "manager_decision") return;
    std::lock_guard<std::mutex> lock(mutex_);
    lastManagerBatch_ = *id;
    cv_.notify_all();
  }

  template<typename Predicate>
  bool WaitFor(Predicate predicate, const std::string &barrier)
  {
    std::unique_lock<std::mutex> lock(mutex_);
    const bool ready = cv_.wait_for(
      lock, std::chrono::duration<double>(timeoutS_),
      [this, &predicate]() { return stopping_ || predicate(); });
    if (!ready || stopping_) {
      RCLCPP_ERROR(get_logger(), "lockstep barrier timed out: %s", barrier.c_str());
      return false;
    }
    return true;
  }

  bool Control(const ignition::msgs::WorldControl &request)
  {
    ignition::msgs::Boolean response;
    bool transportResult = false;
    const bool executed = gzNode_.Request(
      "/world/" + worldName_ + "/control", request, 5000, response, transportResult);
    return executed && transportResult && response.data();
  }

  void Run()
  {
    if (!WaitFor([this]() { return detectorReady_ && odomCount_ > 0; }, "startup")) return;
    std::this_thread::sleep_for(std::chrono::duration<double>(startupDelayS_));

    ignition::msgs::WorldControl pause;
    pause.set_pause(true);
    if (!Control(pause)) {
      RCLCPP_ERROR(get_logger(), "failed to pause /world/%s", worldName_.c_str());
      return;
    }
    RCLCPP_INFO(get_logger(), "lockstep active: %d iterations/control step, camera every %d steps",
      stepIterations_, cameraEveryControlSteps_);

    const auto wallStart = std::chrono::steady_clock::now();
    std::uint64_t controlStep = 0;
    std::optional<std::string> pendingManagerBatch;
    while (rclcpp::ok() && !stopping_ &&
           (maxControlSteps_ <= 0 || controlStep < static_cast<std::uint64_t>(maxControlSteps_))) {
      if (pendingManagerBatch) {
        const std::string expected = *pendingManagerBatch;
        if (!WaitFor([this, &expected]() { return lastManagerBatch_ == expected; },
                     "manager decision for " + expected)) return;
        pendingManagerBatch.reset();
      }

      const auto odomBefore = odomCount_.load();
      const auto commandBefore = commandCount_.load();
      const auto detectorBefore = detectorPublishedCount_.load();
      ignition::msgs::WorldControl step;
      step.set_pause(true);
      step.set_multi_step(static_cast<std::uint64_t>(stepIterations_));
      if (!Control(step)) {
        RCLCPP_ERROR(get_logger(), "Gazebo step request failed at control step %lu", controlStep);
        return;
      }
      ++controlStep;
      if (!WaitFor([this, odomBefore]() { return odomCount_ > odomBefore; }, "odometry")) return;
      if (commandBefore > 0 &&
          !WaitFor([this, commandBefore]() { return commandCount_ > commandBefore; }, "controller command")) return;

      if (controlStep % static_cast<std::uint64_t>(cameraEveryControlSteps_) == 0) {
        if (!WaitFor([this, detectorBefore]() {
              return detectorPublishedCount_ > detectorBefore;
            }, "detector publication")) return;
        std::lock_guard<std::mutex> lock(mutex_);
        pendingManagerBatch = lastDetectorBatch_;
      }
    }

    const double wallS = std::chrono::duration<double>(
      std::chrono::steady_clock::now() - wallStart).count();
    const double simS = static_cast<double>(controlStep * stepIterations_) * 0.001;
    RCLCPP_INFO(get_logger(),
      "lockstep summary: control_steps=%lu sim_s=%.3f wall_s=%.3f rtf=%.4f",
      controlStep, simS, wallS, wallS > 0.0 ? simS / wallS : 0.0);
  }

  std::string worldName_;
  int stepIterations_{100};
  int cameraEveryControlSteps_{2};
  double timeoutS_{30.0};
  double startupDelayS_{2.0};
  int maxControlSteps_{0};
  ignition::transport::Node gzNode_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::thread worker_;
  std::atomic<bool> stopping_{false};
  std::atomic<std::uint64_t> odomCount_{0};
  std::atomic<std::uint64_t> commandCount_{0};
  bool detectorReady_{false};
  std::atomic<std::uint64_t> detectorPublishedCount_{0};
  std::string lastDetectorBatch_;
  std::string lastManagerBatch_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr detectorSub_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr managerSub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odomSub_;
  rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr commandSub_;
};

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<LockstepScheduler>();
  rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(), 2);
  executor.add_node(node);
  executor.spin();
  rclcpp::shutdown();
  return 0;
}
