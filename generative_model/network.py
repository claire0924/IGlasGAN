# -*- coding: utf-8 -*-
"""
Defines the Generator and Critic (discriminator) neural-network architectures used by the GAN training.

"""

from torch import nn
from torch.nn import LeakyReLU
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, noise_dim, num_features, device='cuda'):
        super(Generator, self).__init__()

        self.device = device
        self.model = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(512, num_features),
            nn.Sigmoid()
        )

    def forward(self, noise):
        samples = self.model(noise)
        return samples


class Critic(nn.Module):
    def __init__(self, num_features):
        super(Critic, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(num_features, 512),
            LeakyReLU(negative_slope=0.02),
            nn.Dropout(p=0.1),
            nn.Linear(512, 256),
            LeakyReLU(negative_slope=0.02),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            LeakyReLU(negative_slope=0.02),
            nn.Dropout(p=0.1),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        validity = self.model(x)
        return validity
