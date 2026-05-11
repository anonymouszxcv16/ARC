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
		prioritized=True
	):

		# Parameters.
		max_size = int(max_size)
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.device = device
		self.batch_size = batch_size

		# Memory
		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.prioritized = prioritized

		if prioritized:
			self.priority = torch.zeros(max_size, device=device)
			self.prioritized = True
			self.max_priority = 1

		self.normalize_actions = max_action if normalize_actions else 1

		self.args = args

	# Add tuple.
	def add(self, state, action, next_state, reward, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action / self.normalize_actions
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		
		if self.prioritized:
			self.priority[self.ptr] = self.max_priority

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	# Sample tuple.
	def sample(self, prioritized=False):
		if prioritized:
			csum = torch.cumsum(self.priority[:self.size], 0)
			val = torch.rand(size=(self.batch_size,), device=self.device) * csum[-1]
			self.ind = torch.searchsorted(csum, val).cpu().data.numpy()

		else:
			self.ind = np.random.randint(0, self.size, size=self.batch_size)

		return (
			torch.tensor(self.state[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.action[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.next_state[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.reward[self.ind], dtype=torch.float, device=self.device),
			torch.tensor(self.not_done[self.ind], dtype=torch.float, device=self.device)
		)

	def update_priority(self, priority):
		self.priority[self.ind] = priority.reshape(-1).detach()
		self.max_priority = max(float(priority.max()), self.max_priority)

	def reset_max_priority(self):
		self.max_priority = float(self.priority[:self.size].max())

	# Load offline dataset.
	def load_D4RL(self, dataset):
		"""Load Minari dataset and convert to D4RL format."""

		# Convert to D4RL-style flat arrays - use lists
		states, actions, next_states, rewards, not_dones = [], [], [], [], []

		# Iterate over episodes
		for episode in dataset.iterate_episodes():  # or just 'for episode in dataset'
			observations = episode.observations  # numpy array (T+1, obs_dim)
			actions_arr = episode.actions  # numpy array (T, act_dim)
			rewards_arr = episode.rewards  # numpy array (T,)
			terminations = episode.terminations  # numpy array (T,)
			truncations = getattr(episode, 'truncations', np.zeros_like(episode.rewards))

			# Align observations with actions/rewards
			T = len(actions_arr)
			states.extend(observations[:-1])  # First T observations
			actions.extend(actions_arr)
			next_states.extend(observations[1:])  # Next T observations
			rewards.extend(rewards_arr)
			not_dones.extend(1.0 - (terminations | truncations))

		# Convert to tensors
		self.state = torch.tensor(np.array(states), dtype=torch.float32)
		self.action = torch.tensor(np.array(actions), dtype=torch.float32)
		self.next_state = torch.tensor(np.array(next_states), dtype=torch.float32)
		self.reward = torch.tensor(np.array(rewards), dtype=torch.float32).reshape(-1, 1)
		self.not_done = torch.tensor(np.array(not_dones), dtype=torch.float32).reshape(-1, 1)
		self.size = self.state.shape[0]

		if self.prioritized:
			self.priority = torch.ones(self.size).to(self.device)