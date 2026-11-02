# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import math
import torch
import torch.nn as nn
import torch.optim as optim

from ..modules import HIMActorCritic
from ..storage import HIMRolloutStorage

class HIMPPO:
    actor_critic: HIMActorCritic
    def __init__(self,
                 actor_critic,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 action_clip=None,
                 action_mean_penalty_coef=0.0,
                 min_learning_rate=1.0e-5,
                 max_learning_rate=1.0e-2,
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        if (not math.isfinite(min_learning_rate) or not math.isfinite(max_learning_rate)
                or not 0 < min_learning_rate <= max_learning_rate):
            raise ValueError("Require finite 0 < min_learning_rate <= max_learning_rate")
        if not math.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if action_clip is not None and (not math.isfinite(action_clip) or action_clip <= 0):
            raise ValueError("action_clip must be positive and finite, or None")
        if not math.isfinite(action_mean_penalty_coef) or action_mean_penalty_coef < 0:
            raise ValueError("action_mean_penalty_coef must be finite and nonnegative")
        if action_mean_penalty_coef > 0 and action_clip is None:
            raise ValueError("action_mean_penalty_coef requires action_clip")
        self.min_learning_rate = min_learning_rate
        self.max_learning_rate = max_learning_rate
        self.action_clip = action_clip
        self.action_mean_penalty_coef = action_mean_penalty_coef
        self.last_update_metrics = {}

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = HIMRolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = HIMRolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.eval()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):
        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()
        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos, next_critic_obs):
        self.transition.next_critic_observations = next_critic_obs.clone()
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            # The wrapper supplies terminal observations BEFORE auto-reset.
            # Infinite-horizon timeouts bootstrap V(s_next), not V(s_current).
            with torch.no_grad():
                next_values = self.actor_critic.evaluate(next_critic_obs).squeeze(-1)
            self.transition.rewards += self.gamma * next_values * infos['time_outs'].to(self.device)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def action_mean_overflow(self, mean):
        """Penalize means beyond the actuator clipping range, not sampled noise.

        Sampling and PPO likelihoods remain those of the raw Gaussian. The
        environment wrapper alone clips physical actions; storing clipped
        actions with Gaussian log probabilities would change the estimator.
        """
        if self.action_clip is None:
            return mean.sum() * 0.0
        return (mean.abs() - self.action_clip).clamp(min=0.0).square().sum(dim=-1).mean()

    def update(self):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_estimation_loss = 0
        mean_swap_loss = 0
        mean_action_mean_overflow = 0
        self.last_update_metrics = {}
        if self.action_clip is not None:
            # Storage contains the actual raw rollout samples used for PPO.
            self.last_update_metrics["action_clip_fraction"] = (
                self.storage.actions.abs() > self.action_clip
            ).float().mean().item()
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for obs_batch, critic_obs_batch, actions_batch, next_critic_obs_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch in generator:
                
                self.actor_critic.act(obs_batch)
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch)
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(self.min_learning_rate, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(self.max_learning_rate, self.learning_rate * 1.5)
                        self.learning_rate = min(self.max_learning_rate, max(self.min_learning_rate, self.learning_rate))
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                #Estimator Update
                estimation_loss, swap_loss = self.actor_critic.estimator.update(obs_batch, next_critic_obs_batch, lr=self.learning_rate)

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                action_mean_overflow = self.action_mean_overflow(mu_batch)
                loss = (surrogate_loss + self.value_loss_coef * value_loss
                        - self.entropy_coef * entropy_batch.mean()
                        + self.action_mean_penalty_coef * action_mean_overflow)

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()
                self.actor_critic.project_noise_std_()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()
                mean_estimation_loss += estimation_loss
                mean_swap_loss += swap_loss
                mean_action_mean_overflow += action_mean_overflow.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_estimation_loss /= num_updates
        mean_swap_loss /= num_updates
        self.last_update_metrics["action_mean_overflow"] = mean_action_mean_overflow / num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss, mean_estimation_loss, mean_swap_loss
