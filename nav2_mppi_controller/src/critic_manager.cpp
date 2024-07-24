// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "nav2_mppi_controller/critic_manager.hpp"

namespace mppi
{

void CriticManager::on_configure(
  rclcpp_lifecycle::LifecycleNode::WeakPtr parent, const std::string & name,
  std::shared_ptr<nav2_costmap_2d::Costmap2DROS> costmap_ros, ParametersHandler * param_handler)
{
  parent_ = parent;
  costmap_ros_ = costmap_ros;
  name_ = name;
  auto node = parent_.lock();
  logger_ = node->get_logger();
  parameters_handler_ = param_handler;

  getParams();
  loadCritics();
}

void CriticManager::getParams()
{
  auto node = parent_.lock();
  auto getParam = parameters_handler_->getParamGetter(name_);
  getParam(critic_names_, "critics", std::vector<std::string>{}, ParameterType::Static);
}

void CriticManager::loadCritics()
{
  if (!loader_) {
    loader_ = std::make_unique<pluginlib::ClassLoader<critics::CriticFunction>>(
      "nav2_mppi_controller", "mppi::critics::CriticFunction");
  }

  critics_.clear();
  for (auto name : critic_names_) {
    std::string fullname = getFullName(name);
    auto instance = std::unique_ptr<critics::CriticFunction>(
      loader_->createUnmanagedInstance(fullname));
    critics_.push_back(std::move(instance));
    critics_.back()->on_configure(
      parent_, name_, name_ + "." + name, costmap_ros_,
      parameters_handler_);
    RCLCPP_INFO(logger_, "Critic loaded : %s", fullname.c_str());
  }
}

std::string CriticManager::getFullName(const std::string & name)
{
  return "mppi::critics::" + name;
}

void CriticManager::evalTrajectoriesScores(
  CriticData & data) const
{
  for (size_t q = 0; q < critics_.size(); q++) {
    if (data.fail_flag) {
      break;
    }

    xt::xtensor<float, 1> prev_costs = data.costs;
    critics_[q]->score(data);
    std::string critic_name = critics_[q]->getName();
    if (critics_[q]->critic_cost_visualization_publisher_->get_subscription_count() > 0)
    {
      critics_[q]->critic_cost_visualization_publisher_->on_activate();
      xt::xtensor<float, 1> critic_costs = data.costs - prev_costs;
      visualization_msgs::msg::MarkerArray vis_msg;
      const models::Trajectories & traj = data.trajectories;
      int pos = critic_name.find(std::string("."));
      std::string critic_subname = critic_name.substr(pos + 1, critic_name.length());
      int marker_id = 0;

      auto & shape = traj.x.shape();
      int trajectory_step = 80;
      int time_step = 3;
      for (size_t i = 0; i < shape[0]; i += trajectory_step) {
          for (size_t j = 0; j < shape[1]; j += time_step) {

            auto pose = utils::createPose(traj.x(i, j), traj.y(i, j), 0.03);
            auto scale = utils::createScale(0.05, 0.05, 0.05);


            float red_component = critic_costs(i) / (1 + critic_costs(i));
            float green_component = 1 / (1 + critic_costs(i));
            auto color = utils::createColor(red_component, green_component, 0, 1);

            
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = "odom";
            marker.header.stamp = rclcpp::Time(0, 0);
            marker.ns = "crtics_trajectories";
            marker.id = marker_id;
            marker.type = visualization_msgs::msg::Marker::SPHERE;
            marker.action = visualization_msgs::msg::Marker::ADD;

            marker.pose = pose;
            marker.scale = scale;
            marker.color = color;
            vis_msg.markers.push_back(marker);
            marker_id += 1;
          }
      }
      critics_[q]->critic_cost_visualization_publisher_->publish(vis_msg);
      critics_[q]->critic_cost_visualization_publisher_->on_deactivate();
    }
  }
}

}  // namespace mppi
