from params import Parameters
import torch
from torch import nn
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from torch.utils.data import TensorDataset
import pandas as pd
from params import Parameters
from network import Critic, Generator
import argparse
import os
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

params = Parameters()


class Trainer():
    def __init__(self,data_max,data_min,dataloader, generator, critic, n_epochs, root_path,  device="cuda"):
        super(Trainer, self).__init__()
        self.generator = generator
        self.critic = critic
        self.n_epochs = n_epochs
        self.dataloader = dataloader
        self.root_path = root_path
        self.device = device
        self.data_max=data_max
        self.data_min=data_min
        self.generator_opt = torch.optim.Adam(self.generator.parameters(), lr=params.lr_g, betas=(0.5, 0.999))
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=params.lr_c, betas=(0.5, 0.999))

    # 计算gradient_penalty
    def gradient_penalty(self, real_data, fake_data, critic, lamda=10):
        batch_size = real_data.size(0)
        device = real_data.device
        alpha = torch.rand(batch_size, 1).to(device)  #将alpha转换为一个（batch_size，1）的张量，保证可以于真数据和假数据进行逐元素乘法
        mix_data = alpha * real_data + (1 - alpha) * fake_data
        mix_score = critic(mix_data)


        gradients = torch.autograd.grad(
            inputs=mix_data,
            outputs=mix_score,
            grad_outputs=torch.ones_like(mix_score,dtype=torch.float32),
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = lamda * ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty

    #训练判别器和生成器
    def train(self):
        total_critic_loss = []
        total_generator_loss = []
        for epoch in tqdm(range(self.n_epochs)):
            critic_loss=0
            generator_loss=0
            # 从dataloader中提取真实数据进行训练
            for real in self.dataloader:
                curr_batch_size = len(real[0])
                real_data = real[0].to(torch.float32)
                real_data=real_data.to(self.device)

                # ---------Train Critic ----------------
                critic_loss_batch = 0
                for time in range(params.crit_circle):
                    self.critic_opt.zero_grad()
                    #生成随机噪声
                    noise = torch.rand((curr_batch_size, params.z_dim))
                    noise = noise.to(self.device)
                    # 生成假样本
                    fake_data = self.generator(noise)

                    # 获得判别器对真实样本和假样本的输出
                    critic_fake_pred = self.critic(fake_data.detach())
                    critic_real_pred = self.critic(real_data)
                    # 计算gradient_penalty
                    grad_penalty = self.gradient_penalty(real_data, fake_data, critic=self.critic)
                    # 计算判别器损失并累积多次训练判别器的损失
                    critic_loss_c = critic_fake_pred.mean() - critic_real_pred.mean() + grad_penalty
                    critic_loss_batch += critic_loss_c.item()
                    # 反向传播优化判别器参数
                    critic_loss_c.backward()
                    self.critic_opt.step()

                # 求该批次下判别器平均损失
                mean_critic_loss = critic_loss_batch / params.crit_circle
                # 将该批次下判别器的平均损失累计入判别器总损失
                critic_loss += mean_critic_loss

                # ------------- Train Generator ------------------
                generator_loss_batch = 0
                for time in range(params.generator_circle):
                    self.generator_opt.zero_grad()
                    # 生成随机噪声
                    noise = torch.rand((curr_batch_size, params.z_dim))
                    noise = noise.to(self.device)

                    # 获得生成器输出
                    generator_fake_pred = self.generator(noise)

                    # 获得判别器对假样本的输出
                    critic_fake_pred = self.critic(generator_fake_pred)

                    #将假样本转移到cpu上并转换成numpy格式
                    fake_data = generator_fake_pred.cpu()
                    fake_data = fake_data.detach().numpy()
                    #计算组分惩罚项和性能惩罚项
                    column =  ['Li$_{2}$O','B$_{2}$O$_{3}$','Na$_{2}$O','MgO','Al$_{2}$O$_{3}$','SiO$_{2}$','P$_{2}$O$_{5}$','K$_{2}$O','CaO','TiO$_{2}$','Fe$_{2}$O$_{3}$','CuO','ZnO','As$_{2}$O$_{3}$','SrO','Y$_{2}$O$_{3}$','ZrO$_{2}$','In$_{2}$O$_{3}$','SnO$_{2}$','Sb$_{2}$O$_{3}$','BaO','La$_{2}$O$_{3}$','CeO$_{2}$','PbO','Bi$_{2}$O$_{3}$','ANY_TEC','SPT','MOD_UNG','DENSITY']
                    fake_data_pd = pd.DataFrame(fake_data, columns=column)
                    fake_data_pd_de = fake_data_pd * (self.data_max - self.data_min) + self.data_min
                    fake_data_pd_de = fake_data_pd_de.reindex(columns=column)
                    comp_penalty = np.sum(np.abs(100 - np.sum(fake_data_pd_de.iloc[:, :-4], axis=1))) / curr_batch_size
                    #组分扩展20%
                    property_penalty = np.sum(fake_data_pd.iloc[:, -4] - 0.22895) / curr_batch_size + np.sum(
                        0.54545 - fake_data_pd.iloc[:, -3]) / curr_batch_size + np.sum(
                        0.466437 - fake_data_pd.iloc[:, -2]) / curr_batch_size + np.sum(
                        fake_data_pd.iloc[:, -1] - 0.4055944) / curr_batch_size
                    #获得生成器损失函数
                    generator_loss_c = - critic_fake_pred.mean() +  10 * property_penalty + 0.1*comp_penalty
                    #累积多次训练生成器的损失
                    generator_loss_batch += generator_loss_c.item()
                    # 反向传播更新生成器参数
                    generator_loss_c.backward()
                    self.generator_opt.step()
                # 求该批次下生成器平均损失
                mean_generator_loss = generator_loss_batch / params.generator_circle
                # 累计该批次下生成器的平均损失到生成器的总损失中
                generator_loss += mean_generator_loss
            generator_loss = generator_loss / len(self.dataloader)
            critic_loss = critic_loss / len(self.dataloader)
            # ---------------Saving and visualizing进度和可视化-------------------
            # 训练过程中保存第一步的生成器和判别器的模型并记录对应损失函数值为old_generator_loss和old_critic_loss
            if epoch == 0:
                self.save_checkpoint(self.root_path, "critic", self.critic, epoch,
                                         critic_loss)
                # Save generator cp
                self.save_checkpoint(self.root_path, "generator", self.generator, epoch,
                                         generator_loss)
                old_generator_loss = generator_loss
                old_critic_loss = critic_loss
            # 训练过程中若存在生成器损失小于等于此前记录的 old_generator_loss 并且判别器损失的绝对值小于等于此前记录的old_critic_loss的情况，保存对应的生成器判别器模型，并更新 old_generator_loss和old_critic_loss的值，同时记录步数
            else:
                if generator_loss <= old_generator_loss and np.abs(critic_loss) <= np.abs(old_critic_loss):
                     self.save_checkpoint(self.root_path, "critic", self.critic, epoch,
                                             critic_loss)
                     # Save generator cp
                     self.save_checkpoint(self.root_path, "generator", self.generator, epoch,
                                             generator_loss)
                     old_generator_loss = generator_loss
                     old_critic_loss = critic_loss
                     record_epoch = epoch
                # 输出最后一步的判别器损失和生成器损失以进行对比
                if epoch == self.n_epochs-1:
                    print(
                        f"Epoch: {epoch}:Generator loss: {generator_loss}, critic loss: {critic_loss}")

            # 每间隔params.show_step步输出一次生成器和判别器的损失函数值
            if epoch % params.show_step == 0:
                print(
                    f"Epoch: {epoch}:Generator loss: {generator_loss}, critic loss: {critic_loss}")

            # 记录每步迭代生成器和判别器的损失函数用于做损失函数图
            total_critic_loss.append(critic_loss)
            total_generator_loss.append(generator_loss)

        print(
            f'the best generator at Epoch: {record_epoch}, generator_loss: {old_generator_loss}, critic_loss: {old_critic_loss}')
        # 做损失函数图
        y1 = total_critic_loss
        y2 = total_generator_loss
        x = list(range(self.n_epochs))
        plt.plot(x, y1, label='critic_loss')
        plt.plot(x, y2, label='generator_loss')
        # 添加图例和坐标轴标签
        plt.legend(fontsize=20)
        plt.xlabel('Iterations', fontsize=24)
        plt.ylabel('Loss', fontsize=24)
        # 显示图形
        plt.show()

    # 定义模型保存函数
    def save_checkpoint(self, root_path, name, model, epoch, loss):
        checkpoint = {
            'epoch': epoch,
            'model': model,
            'loss': loss,
        }
        torch.save(model, root_path + "/best_{}.pth".format(name))
        print("Successfully saved {} checkpoint at epoch {} with loss {}".format(name, epoch, loss))

