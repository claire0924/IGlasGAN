# -*- coding: utf-8 -*-
"""
Implements the WGAN-GP training loop with gradient penalty, a differentiable mass-closure composition penalty, and a differentiable one-sided (ReLU) property penalty on the four normalised properties; only out-of-bound samples are penalised and all penalties back-propagate into the generator.

"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from network import Critic, Generator
from params import Parameters

params = Parameters()


class Trainer():
    def __init__(self, data_max, data_min, dataloader, generator, critic, n_epochs, root_path, device="cuda"):
        super(Trainer, self).__init__()
        self.generator = generator
        self.critic = critic
        self.n_epochs = n_epochs
        self.dataloader = dataloader
        self.root_path = root_path
        self.device = device
        self.data_max = data_max
        self.data_min = data_min
        self.data_max_t = torch.as_tensor(
            np.asarray(self.data_max, dtype=np.float32), device=self.device
        ).reshape(-1)
        self.data_min_t = torch.as_tensor(
            np.asarray(self.data_min, dtype=np.float32), device=self.device
        ).reshape(-1)
        self.generator_opt = torch.optim.Adam(self.generator.parameters(), lr=params.lr_g, betas=(0.5, 0.999))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=params.lr_c, betas=(0.5, 0.999))

    def gradient_penalty(self, real_data, fake_data, critic, lamda=10):
        batch_size = real_data.size(0)
        device = real_data.device
        alpha = torch.rand(batch_size, 1).to(device)
        mix_data = alpha * real_data + (1 - alpha) * fake_data
        mix_score = critic(mix_data)

        gradients = torch.autograd.grad(
            inputs=mix_data,
            outputs=mix_score,
            grad_outputs=torch.ones_like(mix_score, dtype=torch.float32),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = lamda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    def train(self):
        total_critic_loss = []
        total_generator_loss = []
        for epoch in tqdm(range(self.n_epochs)):
            critic_loss = 0
            generator_loss = 0
            for real in self.dataloader:
                curr_batch_size = len(real[0])
                real_data = real[0].to(torch.float32)
                real_data = real_data.to(self.device)

                critic_loss_batch = 0
                for time in range(params.crit_circle):
                    self.critic_opt.zero_grad()
                    noise = torch.rand((curr_batch_size, params.z_dim))
                    noise = noise.to(self.device)
                    fake_data = self.generator(noise)

                    critic_fake_pred = self.critic(fake_data.detach())
                    critic_real_pred = self.critic(real_data)
                    grad_penalty = self.gradient_penalty(real_data, fake_data, critic=self.critic)
                    critic_loss_c = critic_fake_pred.mean() - critic_real_pred.mean() + grad_penalty
                    critic_loss_batch += critic_loss_c.item()
                    critic_loss_c.backward()
                    self.critic_opt.step()

                mean_critic_loss = critic_loss_batch / params.crit_circle
                critic_loss += mean_critic_loss

                generator_loss_batch = 0
                for time in range(params.generator_circle):
                    self.generator_opt.zero_grad()
                    noise = torch.rand((curr_batch_size, params.z_dim))
                    noise = noise.to(self.device)

                    generator_fake_pred = self.generator(noise)

                    critic_fake_pred = self.critic(generator_fake_pred)

                    oxides = generator_fake_pred[:, :25] * (
                        self.data_max_t[:25] - self.data_min_t[:25]
                    ) + self.data_min_t[:25]
                    comp_penalty = (100.0 - oxides.sum(dim=1)).abs().mean()

                    props = generator_fake_pred[:, 25:29]
                    property_penalty = (
                        torch.relu(0.1873 - props[:, 0])
                        +torch.relu(props[:, 0] - 0.2290)
                        + torch.relu(0.5454 - props[:, 1])
                        + torch.relu(0.4664 - props[:, 2] )
                        + torch.relu( props[:, 3] - 0.4061)
                    ).sum() / curr_batch_size

                    generator_loss_c = - critic_fake_pred.mean() + 10 * property_penalty + 0.1 * comp_penalty
                    generator_loss_batch += generator_loss_c.item()
                    generator_loss_c.backward()
                    self.generator_opt.step()
                mean_generator_loss = generator_loss_batch / params.generator_circle
                generator_loss += mean_generator_loss
            generator_loss = generator_loss / len(self.dataloader)
            critic_loss = critic_loss / len(self.dataloader)

            if epoch == 0:
                self.save_checkpoint(self.root_path, "critic", self.critic, epoch,
                                     critic_loss)
                self.save_checkpoint(self.root_path, "generator", self.generator, epoch,
                                     generator_loss)
                old_generator_loss = generator_loss
                old_critic_loss = critic_loss
            else:
                if generator_loss <= old_generator_loss and np.abs(critic_loss) <= np.abs(old_critic_loss):
                    self.save_checkpoint(self.root_path, "critic", self.critic, epoch,
                                         critic_loss)
                    self.save_checkpoint(self.root_path, "generator", self.generator, epoch,
                                         generator_loss)
                    old_generator_loss = generator_loss
                    old_critic_loss = critic_loss
                    record_epoch = epoch
                if epoch == self.n_epochs - 1:
                    print(
                        f"Epoch: {epoch}:Generator loss: {generator_loss}, critic loss: {critic_loss}")

            if epoch % params.show_step == 0:
                print(
                    f"Epoch: {epoch}:Generator loss: {generator_loss}, critic loss: {critic_loss}")

            total_critic_loss.append(critic_loss)
            total_generator_loss.append(generator_loss)

        print(
            f'the best generator at Epoch: {record_epoch}, generator_loss: {old_generator_loss}, '
            f'critic_loss: {old_critic_loss}')

        y1 = total_critic_loss
        y2 = total_generator_loss
        x = list(range(self.n_epochs))
        plt.plot(x, y1, label='critic_loss')
        plt.plot(x, y2, label='generator_loss')
        plt.legend(fontsize=20)
        plt.xlabel('Iterations', fontsize=24)
        plt.ylabel('Loss', fontsize=24)
        plt.show()

    def save_checkpoint(self, root_path, name, model, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model': model,
            'loss': loss,
        }
        torch.save(model, root_path + "/best_{}.pth".format(name))
        print("Successfully saved {} checkpoint at epoch {} with loss {}".format(name, epoch, loss))
