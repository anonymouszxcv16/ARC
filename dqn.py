import copy
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import optim

# Layer normalization.
def AvgL1Norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


class ReplayMemory:
    def __init__(self, capacity, args):
        self.capacity = capacity
        self.args = args
        self.ptr = 0
        self.size = 0

        # Allocate tensor buffers
        self.state = torch.zeros((capacity, *args.state_shape), dtype=torch.float32, device=args.device)
        self.action = torch.zeros((capacity, *args.action_shape), dtype=torch.long, device=args.device)
        self.next_state = torch.zeros((capacity, *args.state_shape), dtype=torch.float32, device=args.device)
        self.reward = torch.zeros((capacity, 1), dtype=torch.float32, device=args.device)
        self.done = torch.zeros((capacity, 1), dtype=torch.bool, device=args.device)

        self.priority = torch.ones((capacity,), dtype=torch.float32, device=args.device)

    # Add tuple.
    def push(self, state, action, next_state, reward, done):
        self.state[self.ptr:self.ptr + state.shape[0]] = state
        self.action[self.ptr:self.ptr + state.shape[0]] = action
        self.next_state[self.ptr:self.ptr + state.shape[0]] = next_state
        self.reward[self.ptr:self.ptr + state.shape[0]] = reward
        self.done[self.ptr:self.ptr + state.shape[0]] = done

        self.priority[self.ptr:self.ptr + state.shape[0]] = self.priority.max().item() if self.size > 0 else 1.0  # max priority

        self.ptr = (self.ptr + state.shape[0]) % self.capacity
        self.size = min(self.size + state.shape[0], self.capacity)

    def sample(self, priority=False):
        if priority:
            probs = self.priority[:self.size]
            probs /= probs.sum()

            idxs = torch.multinomial(probs, self.args.batch_size, replacement=True)
        else:
            idxs = torch.randint(0, self.size, (self.args.batch_size,), device=self.args.device)

        batch = {
            'state': self.state[idxs],
            'action': self.action[idxs],
            'next_state': self.next_state[idxs],
            'reward': self.reward[idxs],
            'done': self.done[idxs],
            'idxs': idxs
        }

        return batch

    def update_priorities(self, idxs, td_errors):
        self.priority[idxs] = td_errors.squeeze().detach()

    def __len__(self):
        return self.size

class Encoder(nn.Module):
    def __init__(self, args, zs_dim=256, hdim=256, activ=F.elu):
        super(Encoder, self).__init__()

        self.activ = activ

        # state encoder
        self.zs1 = nn.Linear(args.state_shape[0], hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)

        # state-action encoder
        self.zsa1 = nn.Linear(zs_dim + args.action_shape[0], hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

        self.args = args

    def zs(self, state):
        # Fully connected.
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))

        # Normalization.
        zs = AvgL1Norm(self.zs3(zs))

        return zs

    def zsa(self, zs, action):
        # Fully connected.
        zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)

        return zsa

class DQN(nn.Module):

    def __init__(self, args):
        super(DQN, self).__init__()

        self.layer1 = nn.ParameterList([nn.Linear(*args.state_shape, args.hdim) for _ in range(args.N)])
        self.layer2 = nn.ParameterList([nn.Linear(args.zs_dim + args.hdim, args.hdim) for _ in range(args.N)])
        self.layer3 = nn.ParameterList([nn.Linear(args.hdim, args.n_actions) for _ in range(args.N)])

        self.args = args

    # Called with either one element to determine next action, or a batch during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x, zs):
        q_values = []

        for i in range(self.args.N):
            q = AvgL1Norm(self.layer1[i](x))

            q = torch.cat([q, zs], 1)

            q = F.relu(self.layer2[i](q))
            q_value = self.layer3[i](q)

            q_values.append(q_value)

        q_values = torch.stack(q_values, dim=0)
        return q_values


class Agent():
    def __init__(self, args, env, preprocessor):
        self.env = env
        self.args = args
        self.preprocessor = preprocessor

        self.t = 0

        self.policy_net = DQN(args).to(args.device)
        self.target_net = DQN(args).to(args.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.encoder = Encoder(args).to(args.device)
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=args.encoder_lr)
        self.fixed_encoder = copy.deepcopy(self.encoder)
        self.fixed_encoder_target = copy.deepcopy(self.encoder)

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=args.lr, amsgrad=True)

        self.memory = ReplayMemory(args.replay_size, args)

        self.std_inverse_sum = 0

    def get_epsilon(self):
        return self.args.eps_end + (self.args.eps_start - self.args.eps_end) * math.exp(-1. * self.t / self.args.eps_decay)

    def avd_shaping(self, q_old, q_new):
        advantage = (1 - self.args.avd_alpha) * q_old + self.args.avd_alpha * q_new
        return torch.tanh(advantage)  # Bounded shaping

    def scaled_sigmoid(self, x):
        x = x.cpu()
        return self.args.alpha_min + (self.args.alpha_max - self.args.alpha_min) / (1 + np.exp(-x))

    def select_action(self, state, q_idx=0, greedy=False):
        sample = random.random()
        eps_threshold = self.get_epsilon()

        if sample > eps_threshold or greedy:
            with torch.no_grad():
                zs = self.fixed_encoder.zs(state)
                # t.max(1) will return the largest column value of each row.
                action = self.policy_net(state, zs)[q_idx].argmax(-1)
                return torch.tensor(action, dtype=torch.long).unsqueeze(1).to(self.args.device)
        else:
            return torch.tensor([[self.env.action_space.sample()]], device=self.args.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.args.batch_size:
            return

        # Sampling a batch
        batch = self.memory.sample(priority="PER" in self.args.policy or "RELO" in self.args.policy)

        # Mask for non-final states
        non_final_mask = ~batch['done'].squeeze(-1)  # shape [B]

        non_final_next_states = batch['next_state'][non_final_mask]
        state = batch['state']
        action = batch['action']
        reward = batch['reward']

        if "SALE" in self.args.policy:
            with torch.no_grad():
                next_zs = self.encoder.zs(batch['next_state'])

            zs = self.encoder.zs(state)
            pred_zs = self.encoder.zsa(zs, action)

            # Loss.
            encoder_loss = F.mse_loss(pred_zs, next_zs)
            self.encoder_optimizer.zero_grad()
            encoder_loss.backward()
            self.encoder_optimizer.step()

        with torch.no_grad():
            zs = self.fixed_encoder.zs(state)

        q = self.policy_net(state, zs).squeeze(2).gather(index=action.reshape(1, state.shape[0], 1), dim=-1).flip(0).squeeze(-1)
        q_next = torch.zeros(self.args.N, state.shape[0], device=self.args.device, dtype=torch.long)

        with torch.no_grad():
            zs_target = self.fixed_encoder_target.zs(non_final_next_states)
            q_next[:, non_final_mask] = (self.target_net(non_final_next_states, zs_target).amax(-1).squeeze(-1).clone().detach().to(self.args.device,
                                                                                                                                    dtype=torch.long))
        # Tanh
        if "TANH" in self.args.policy:
            reward = torch.tanh(reward)

        # Softsign
        if "SOFTSIGN" in self.args.policy:
            reward = torch.nn.functional.softsign(reward)

        # AVD
        if "AVD" in self.args.policy:
            with torch.no_grad():
                zs_target = self.fixed_encoder_target.zs(state)
                q_target = self.target_net(state, zs_target).squeeze(2).gather(index=action.reshape(1, state.shape[0], 1), dim=-1).flip(0).squeeze(-1)

            reward = reward + self.avd_shaping(q, q_target)

        # Target
        if "RRS" in self.args.policy or "NRS" in self.args.policy:
            # Q-targets.
            rewards_mean, rewards_max = self.memory.reward[
                                        :self.memory.size].mean(), self.memory.reward[
                                                                          :self.memory.size].max()
            reward = reward.expand((self.args.batch_size, self.args.N)).clone()
            reward = ((reward - rewards_mean) / rewards_max)

            if "RRS" in self.args.policy:
                reward = (1 / self.args.alpha) * 1 / (1 + (-self.args.alpha * reward).exp()) * (
                            1 + (self.args.alpha * reward).exp()).log()

                if self.args.auto_alpha == 1:
                    rewards_std = self.memory.reward.std()
                    std_inverse = 1 / rewards_std

                    self.std_inverse_sum += std_inverse

                    if (self.t + self.args.timesteps_before_training) % self.args.auto_alpha_interval == 0:
                        std_inverse_mean = (self.std_inverse_sum / self.args.auto_alpha_interval).cpu()
                        self.args.alpha = self.scaled_sigmoid(std_inverse_mean)

                        self.std_inverse_sum = 0

        q_next = (q_next * self.args.discount) + reward

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        td_loss = criterion(q, q_next)

        if "PER" in self.args.policy:
            td_error = (q - q_next).abs()
            priority = td_error.max(1)[0].pow(self.args.alpha_per)
            self.memory.update_priorities(batch['idxs'], priority)

        elif "RELO" in self.args.policy:
            with torch.no_grad():
                zs_target = self.fixed_encoder_target.zs(state)
                q_target = self.target_net(state, zs_target).squeeze(2).gather(index=action.reshape(1, state.shape[0], 1), dim=-1).flip(0).squeeze(-1)
                loss = (q - q_target).abs()

            priority = loss.max(1)[0].pow(self.args.alpha_per)
            self.memory.update_priorities(batch['idxs'], priority)

        # Optimize the model
        self.optimizer.zero_grad()
        td_loss.backward()

        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()