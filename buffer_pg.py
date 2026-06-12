import random
import numpy as np
import torch


# Replay buffer
class LAP(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		device,
		args,
		max_size=1e6,
		batch_size=256,
		max_action=1,
		normalize_actions=True,
	):

		# Parameters.
		max_size = int(max_size)
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.device = device
		self.batch_size = batch_size
		self.action_dim = action_dim
		self.state_dim = state_dim

		# Memory
		self.state = torch.zeros((max_size, state_dim)).to(args.device)
		self.action = torch.zeros((max_size, action_dim)).to(args.device)

		# ppo
		self.log_policy = torch.zeros((max_size, 1)).to(args.device)
		self.next_state = torch.zeros((max_size, state_dim)).to(args.device)
		self.reward = torch.zeros((max_size, 1)).to(args.device)

		# mc score
		self.mc_score = torch.zeros((max_size, 1)).to(args.device)
		self.not_done = torch.zeros((max_size, 1)).to(args.device)

		self.normalize_actions = max_action if normalize_actions else 1

		self.args = args


	# Add tuple.
	def add(self, state, action, next_state, reward, done, mc_score=None):
		self.state[self.ptr:self.ptr + state.shape[0]] = state
		self.action[self.ptr:self.ptr + state.shape[0]] = action / self.normalize_actions
		self.next_state[self.ptr:self.ptr + state.shape[0]] = next_state
		self.reward[self.ptr:self.ptr + state.shape[0]] = reward
		self.not_done[self.ptr:self.ptr + state.shape[0]] = 1. - done

		if mc_score is not None:
			self.mc_score[self.ptr:self.ptr + state.shape[0]] = mc_score

		self.ptr = (self.ptr + state.shape[0]) % self.max_size
		self.size = min(self.size + state.shape[0], self.max_size)

	# Sample tuple.
	def sample(self, ind=None):
		if ind is not None:
			self.ind = ind

		elif "REINFORCE" in self.args.policy:
			# trajectory
			start = random.randint(0, self.size - 1 - self.batch_size)
			end = start + self.batch_size - 1

			self.ind = torch.range(start, end, dtype=torch.int64)

		else:
			self.ind = np.random.randint(0, self.size, size=min(self.batch_size, self.size), dtype=np.int64)

		return (
			torch.tensor(self.state[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.action[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.log_policy[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.next_state[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.reward[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.mc_score[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.not_done[self.ind], dtype=torch.float, device=self.device)
		)