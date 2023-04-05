import math
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.Tensor(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)
        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x, self.weight_mu + self.weight_sigma * self.weight_epsilon,
               self.bias_mu + self.bias_sigma * self.bias_epsilon, )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.randn(size)
        return x.sign().mul(x.abs().sqrt())


class Network(nn.Module):
    def __init__(self, in_dim, out_dim: int, name, chkpt_dir, atom_size: int, support: torch.Tensor):
        """Initialization."""
        super(Network, self).__init__()
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        self.conv1 = nn.Conv2d(in_dim[0], 32, 2, stride=1)
        self.conv2 = nn.Conv2d(32, 64, 2, stride=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc_input_dims = self.calculate_conv_output_dims(in_dim)

        self.feature = nn.Linear(self.fc_input_dims, 1024)

        self.value_noisy_layer1 = NoisyLinear(1024, 512)
        self.value_noisy_layer2 = NoisyLinear(512, atom_size)

        self.adv_noisy_layer1 = NoisyLinear(1024, 512)
        self.adv_noisy_layer2 = NoisyLinear(512, out_dim * atom_size)

    def calculate_conv_output_dims(self, input_dims):
        state = torch.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, self.fc_input_dims)
        feature = F.relu(self.feature(x))
        hidden_v = F.relu(self.value_noisy_layer1(feature))
        hidden_adv = F.relu(self.adv_noisy_layer1(feature))

        advantage = self.adv_noisy_layer2(hidden_adv).view(-1, self.out_dim, self.atom_size)
        value = self.value_noisy_layer2(hidden_v).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.log_softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-6)  # for avoiding nans

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""

        self.value_noisy_layer1.reset_noise()
        self.value_noisy_layer2.reset_noise()

        self.adv_noisy_layer1.reset_noise()
        self.adv_noisy_layer2.reset_noise()

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(torch.load(self.checkpoint_file))
