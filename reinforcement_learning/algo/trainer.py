import time

import setup_path

from multiprocessing import Queue, Process
from copy import deepcopy

import numpy as np
import torch as th
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter

from .memory import ReplayMemory, Experience
from .network import Critic, Actor
from .misc import get_folder, soft_update, FloatTensor
from .visual import NetLooker, net_visual


def update(queue, actor, actor_target, critic, critic_target, a_optimizer, c_optimizer, mse_loss,
           tau, gamma, memory, batch_size, update_target):
    transitions = memory.sample(batch_size)
    batch = Experience(*zip(*transitions))

    state_batch = th.from_numpy(np.array(batch.states)).type(FloatTensor)
    action_batch = th.from_numpy(np.array(batch.actions)).type(FloatTensor)
    next_states_batch = th.from_numpy(np.array(batch.next_states)).type(FloatTensor)
    reward_batch = th.from_numpy(np.array(batch.rewards)).type(FloatTensor).unsqueeze(dim=-1)
    done_batch = th.from_numpy(np.array(batch.dones)).type(FloatTensor).unsqueeze(dim=-1)

    c_optimizer.zero_grad()
    current_q = critic(state_batch, action_batch)
    next_actions = actor_target(next_states_batch)
    target_next_q = critic_target(next_states_batch, next_actions)
    target_q = target_next_q * gamma * (1 - done_batch[:, :]) + reward_batch[:, :]
    loss_q = mse_loss(current_q, target_q.detach())
    loss_q.backward()
    c_optimizer.step()

    a_optimizer.zero_grad()
    ac = action_batch.clone()
    ac[:, :] = actor(state_batch[:, :])
    loss_p = -critic(state_batch, ac).mean()
    loss_p.backward()
    a_optimizer.step()

    if update_target:
        soft_update(critic_target, critic, tau)
        soft_update(actor_target, actor, tau)

    if queue is not None:
        queue.put([loss_q.item(), loss_p.item()])
    else:
        return loss_q.item(), loss_p.item()


class Trainer:
    def __init__(self, dim_obs, dim_act, args, folder=None):
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.batch_size = args.batch_size
        self.gamma = args.gamma
        self.tau = args.tau
        self.var = 1.0

        self.memory = ReplayMemory(args.memory_length)

        self.actor = Actor(dim_obs, dim_act)
        self.actor_target = deepcopy(self.actor)
        self.actor_optimizer = Adam(self.actor.parameters(), lr=args.a_lr)

        self.critic = Critic(dim_obs, dim_act)
        self.critic_target = deepcopy(self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=args.c_lr)
        self.mse_loss = nn.MSELoss()

        self.use_cuda = th.cuda.is_available()
        if self.use_cuda:
            self.actor.cuda()
            self.critic.cuda()
            self.actor_target.cuda()
            self.critic_target.cuda()
            self.mse_loss.cuda()

        self.c_losses, self.a_losses = [], []
        self.step = 0
        self.__addiction(dim_obs, dim_act, folder)

    def __addiction(self, dim_obs, dim_act, folder):
        self.writer = None
        self.actor_looker = None
        self.critic_looker = None

        if folder is None:
            return

        # 数据记录（计算图、logs和网络参数）的保存文件路径
        self.path = get_folder(folder,
                               has_graph=True,
                               has_log=True,
                               has_model=True,
                               allow_exist=True)
        if self.path['log_path'] is not None:
            self.writer = SummaryWriter(self.path['log_path'])
        if self.path['graph_path'] is not None:
            print('Draw the net of Actor and Critic!')
            net_visual([(1,) + dim_obs],
                       self.actor,
                       d_type=FloatTensor,
                       filename='actor',
                       directory=self.path['graph_path'],
                       format='png',
                       cleanup=True)
            self.actor_looker = NetLooker(net=self.actor,
                                          name='actor',
                                          is_look=False,
                                          root=self.path['graph_path'])
            net_visual([(1,) + dim_obs, (1, dim_act)],
                       self.critic,
                       d_type=FloatTensor,
                       filename='critic',
                       directory=self.path['graph_path'],
                       format='png',
                       cleanup=True)
            self.critic_looker = NetLooker(net=self.critic,
                                           name='critic',
                                           is_look=False)
            print()

    def add_experience(self, obs, act, next_obs, rew, done):
        self.memory.push(obs, act, next_obs, rew, done)

    def act(self, obs, var_decay=False):
        obs = th.from_numpy(obs).type(FloatTensor)
        sb = obs.detach()
        act = self.actor(sb.unsqueeze(0)).squeeze()
        act += th.from_numpy(np.random.randn(self.n_actions) * self.var).type(FloatTensor)
        act = th.clamp(act, -1.0, 1.0)
        if var_decay and self.var > 0.05:
            self.var *= 0.99998
        return act.data.cpu().numpy()

    def update(self, interval=10):
        self.step += 1
        start = time.time()
        update_target = (self.step % interval == 0)

        queue = Queue()
        work_args = (queue,
                     self.actor, self.actor_target, self.critic, self.critic_target,
                     self.actor_optimizer, self.critic_optimizer, self.mse_loss,
                     self.tau, self.gamma,
                     self.memory, self.batch_size,
                     update_target,)
        worker = Process(target=update, args=work_args)
        worker.start()

        [loss_q, lost_p] = queue.get()
        self.c_losses.append(loss_q)
        self.a_losses.append(lost_p)

        if update_target:
            # Record and visual the loss value of Actor and Critic
            self.scalar(key='critic_loss', value=np.mean(self.c_losses), episode=self.step)
            self.scalar(key='actor_loss', value=np.mean(self.a_losses), episode=self.step)
            self.scalar(key="Param", value=self.var, episode=self.step)
            self.c_losses, self.a_losses = [], []
        print(self.step, time.time() - start)

    # def update(self, step):
    #     start = time.time()
    #
    #     transitions = self.memory.sample(self.batch_size)
    #     batch = Experience(*zip(*transitions))
    #
    #     state_batch = th.from_numpy(np.array(batch.states)).type(FloatTensor)
    #     action_batch = th.from_numpy(np.array(batch.actions)).type(FloatTensor)
    #     next_states_batch = th.from_numpy(np.array(batch.next_states)).type(FloatTensor)
    #     reward_batch = th.from_numpy(np.array(batch.rewards)).type(FloatTensor).unsqueeze(dim=-1)
    #     done_batch = th.from_numpy(np.array(batch.dones)).type(FloatTensor).unsqueeze(dim=-1)
    #
    #     self.critic_optimizer.zero_grad()
    #     current_q = self.critic(state_batch, action_batch)
    #     next_actions = self.actor_target(next_states_batch)
    #     target_next_q = self.critic_target(next_states_batch, next_actions)
    #     target_q = target_next_q * self.gamma * (1 - done_batch[:, :]) + reward_batch[:, :]
    #     loss_q = self.mse_loss(current_q, target_q.detach())
    #     loss_q.backward()
    #     self.critic_optimizer.step()
    #
    #     self.actor_optimizer.zero_grad()
    #     ac = action_batch.clone()
    #     ac[:, :] = self.actor(state_batch[:, :])
    #     loss_p = -self.critic(state_batch, ac).mean()
    #     loss_p.backward()
    #     self.actor_optimizer.step()
    #
    #     self.c_losses.append(loss_q.item())
    #     self.a_losses.append(loss_p.item())
    #
    #     if step % 100 == 0:
    #         soft_update(self.critic_target, self.critic, self.tau)
    #         soft_update(self.actor_target, self.actor, self.tau)
    #
    #         # Record and visual the loss value of Actor and Critic
    #         self.scalar(key='critic_loss', value=np.mean(self.c_losses), episode=step)
    #         self.scalar(key='actor_loss', value=np.mean(self.a_losses), episode=step)
    #         self.c_losses, self.a_losses = [], []
    #     print(step, time.time() - start)

    def load_model(self, load_path=None):
        if load_path is None:
            load_path = self.path['model_path']

        if load_path is not None:
            actor, critic = self.actor, self.critic
            actor_state_dict = th.load(load_path + 'actor.pth').state_dict()
            critic_state_dict = th.load(load_path + 'critic.pth').state_dict()

            actor.load_state_dict(actor_state_dict)
            critic.load_state_dict(critic_state_dict)
            self.actor_target = deepcopy(actor)
            self.critic_target = deepcopy(critic)
        else:
            print('Load path is empty!')
            raise NotImplementedError

    def save_model(self, save_path=None):
        if save_path is None:
            save_path = self.path['model_path']

        if save_path is not None:
            th.save(self.actor, save_path + 'actor.pth')
            th.save(self.critic, save_path + 'critic.pth')
        else:
            print('Save path is empty!')
            raise NotImplementedError

    def scalars(self, key, value, episode):
        self.writer.add_scalars(key, value, episode)

    def scalar(self, key, value, episode):
        self.writer.add_scalar(key, value, episode)

    def close(self):
        if self.writer is not None:
            self.writer.close()
        if self.actor_looker is not None:
            self.actor_looker.close()
        if self.critic_looker is not None:
            self.critic_looker.close()
