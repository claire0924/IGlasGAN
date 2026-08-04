# -*- coding: utf-8 -*-
"""
Entry point that trains a WGAN-GP with differentiable composition/property penalties so the generator yields glass compositions whose properties fall in the target range.

"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from network import Critic, Generator
from params import Parameters
from trainer import Trainer

df = pd.read_csv('property_4.csv')
df = df.set_index('GlasNo')

max = [54, 84, 27.66, 46.2, 32.208, 100, 18.48, 23.1, 43.536, 6, 2.772, 10.272, 43.716, 1.428, 16.44, 5.952, 10.128,
       2.4, 0.684, 2.4, 24, 13.128, 0.36, 12.696, 55.188, 150, 1068, 129.6, 3.96]
min = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6.04, 302.4, 36.64, 1.672]
index = ['Li$_{2}$O', 'B$_{2}$O$_{3}$', 'Na$_{2}$O', 'MgO', 'Al$_{2}$O$_{3}$', 'SiO$_{2}$', 'P$_{2}$O$_{5}$',
         'K$_{2}$O', 'CaO', 'TiO$_{2}$', 'Fe$_{2}$O$_{3}$', 'CuO', 'ZnO', 'As$_{2}$O$_{3}$', 'SrO', 'Y$_{2}$O$_{3}$',
         'ZrO$_{2}$', 'In$_{2}$O$_{3}$', 'SnO$_{2}$', 'Sb$_{2}$O$_{3}$', 'BaO', 'La$_{2}$O$_{3}$', 'CeO$_{2}$', 'PbO',
         'Bi$_{2}$O$_{3}$', 'ANY_TEC', 'SPT', 'MOD_UNG', 'DENSITY']
data_min = pd.Series(min, index=index)
data_max = pd.Series(max, index=index)

df_standardized = (df - data_min) / (data_max - data_min)
numpy_array = df_standardized.values

df_standardized_tensor = torch.from_numpy(numpy_array)

glass_dataset = TensorDataset(df_standardized_tensor)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=30000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--train_samples", type=int, default=2445, help="number of training data")
parser.add_argument("--latent_dim", type=int, default=29, help="dimensionality of the latent space")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--n_generator", type=int, default=1, help="number of training steps for generator per iter")
parser.add_argument("--show_step", type=int, default=100, help="number of training steps at each epoch to show result")

opt = parser.parse_args()

params = Parameters()
params.batch_size = opt.batch_size
params.epochs = opt.n_epochs
params.z_dim = opt.latent_dim
params.crit_circle = opt.n_critic
params.generator_circle = opt.n_generator
params.show_step = opt.show_step
params.train_samples = opt.train_samples

if not os.path.exists("./checkpoint"):
    os.makedirs("./checkpoint")

dataloader = DataLoader(glass_dataset, batch_size=params.batch_size, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

generator = Generator(params.z_dim, params.z_dim).to(device)
critic = Critic(params.z_dim).to(device)

trainer = Trainer(data_max=data_max, data_min=data_min, dataloader=dataloader, generator=generator, critic=critic,
                  n_epochs=params.epochs, device=device, root_path="./checkpoint")

trainer.train()

torch.save(generator, 'generator_oxide_property(4)_extend.pth')
