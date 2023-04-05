import os
from typing import Dict, List, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from src.Rainbow.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from src.Rainbow.Network import Network
from src.Rainbow.ReplayBuffer import ReplayBuffer


class DQNAgent:
    """DQN Agent interacting with environment.

    Attribute:
        env (gym.Env): openAI Gym environment
        memory (ReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        epsilon (float): parameter for epsilon greedy policy
        epsilon_decay (float): step size to decrease epsilon
        max_epsilon (float): max value of epsilon
        min_epsilon (float): min value of epsilon
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including
                           state, action, reward, next_state, done
    """

    def __init__(
            self,
            obs_dim: int,
            action_dim: int,
            memory_size: int,
            batch_size: int,
            target_update: int,
            lr: float = 2.5e-4,
            gamma: float = 0.99,
            chkpt_dir='models',
            # PER
            alpha: float = 0.2,
            beta: float = 0.6,
            prior_eps: float = 1e-6,
            # Categorical DQN parameters
            v_min: float = 0.0,
            v_max: float = 200.0,
            atom_size: int = 51,
            # N-step Learning
            n_step: int = 3,

    ):
        """Initialization.

        Args:
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            epsilon_decay (float): step size to decrease epsilon
            lr (float): learning rate
            max_epsilon (float): max value of epsilon
            min_epsilon (float): min value of epsilon
            gamma (float): discount factor
        """
        # obs_dim = env.observation_space.shape[0]
        # action_dim = env.action_space.n
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_space = [i for i in range(action_dim)]
        self.batch_size = batch_size

        self.target_update = target_update
        self.gamma = gamma
        self.lr = lr
        self.update_cnt = 0

        # PER
        self.beta = beta
        self.prior_eps = prior_eps
        self.memory = PrioritizedReplayBuffer(obs_dim, memory_size, batch_size, alpha)
        # device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma)

        # C51
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)

        # networks: dqn, dqn_target
        self.dqn = Network(obs_dim, action_dim, name='dqn', chkpt_dir=chkpt_dir, atom_size=self.atom_size,
                           support=self.support).to(self.device)
        self.dqn_target = Network(obs_dim, action_dim, name='dqn_target', chkpt_dir=chkpt_dir, atom_size=self.atom_size,
                                  support=self.support).to(self.device)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()

        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters(), lr=self.lr, eps=0.005 / self.batch_size)

    def select_action(self, state: np.ndarray) -> np.ndarray:
        """Select an action from the input state."""
        selected_action = self.dqn(torch.FloatTensor(state).to(self.device)).argmax()
        selected_action = selected_action.detach().cpu().numpy(force=True)
        return selected_action

    def exploit_action(self, state: np.ndarray) -> np.ndarray:

        return self.select_action(state)

    def store_experience(self, state, action, reward, next_state, done):
        self.memory.store(state, action, reward, next_state, done)

        if self.use_n_step:
            one_step_transition = self.memory_n.store(state, action, reward, next_state, done)
            # 1-step transition
        else:
            one_step_transition = self.transition
        # add a single step transition
        if one_step_transition:
            self.memory.store(*one_step_transition)

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        self.update_cnt += 1
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        # PER: importance sampling before average
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        loss = torch.mean(elementwise_loss * weights)
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        # if hard update is needed
        if self.update_cnt % self.target_update == 0:
            print("update")
            self._target_hard_update()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)  # TODO should probably use TD_ERROR instead of loss.

        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()

    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        loss = self.update_model()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma) -> torch.Tensor:
        """Return dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())

    def save_models(self):
        self.dqn.save_checkpoint()
        self.dqn_target.save_checkpoint()

    def load_models(self):
        self.dqn.load_checkpoint()
        self.dqn_target.load_checkpoint()

    def update_beta(self, frame_idx, num_frames):
        fraction = min(frame_idx / num_frames, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)
