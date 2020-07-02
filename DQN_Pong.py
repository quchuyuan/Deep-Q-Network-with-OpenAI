"""
Adapted from OpenAI Baselines
https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""

from collections import deque
import numpy as np
import gym
from PIL import Image

def make_env(env, stack_frames=True, episodic_life=True, clip_rewards=False, scale=False):
	if episodic_life:
		env = EpisodicLifeEnv(env)

	env = NoopResetEnv(env, noop_max=30)
	env = MaxAndSkipEnv(env, skip=4)
	if 'FIRE' in env.unwrapped.get_action_meanings():
		env = FireResetEnv(env)

	env = WarpFrame(env)
	if stack_frames:
		env = FrameStack(env, 4)
	if clip_rewards:
		env = ClipRewardEnv(env)
	return env

class RewardScaler(gym.RewardWrapper):

	def reward(self, reward):
		return reward * 0.1

class ClipRewardEnv(gym.RewardWrapper):
	def __init__(self, env):
		gym.RewardWrapper.__init__(self, env)

	def reward(self, reward):
		"""Bin reward to {+1, 0, -1} by its sign."""
		return np.sign(reward)

class LazyFrames(object):
	def __init__(self, frames):
		"""This object ensures that common frames between the
		observations are only stored once. It exists purely to
		optimize memory usage which can be huge for DQN's 1M frames
		replay buffers.

		This object should only be converted to numpy array before
		being passed to the model. You'd not believe how complex the
		previous solution was."""
		self._frames = frames
		self._out = None

	def _force(self):
		if self._out is None:
			self._out = np.concatenate(self._frames, axis=2)
			self._frames = None
		return self._out

	def __array__(self, dtype=None):
		out = self._force()
		if dtype is not None:
			out = out.astype(dtype)
		return out

	def __len__(self):
		return len(self._force())

	def __getitem__(self, i):
		return self._force()[i]

class FrameStack(gym.Wrapper):
	def __init__(self, env, k):
		"""Stack k last frames.
		Returns lazy array, which is much more memory efficient.
		See Also
		--------
		baselines.common.atari_wrappers.LazyFrames
		"""
		gym.Wrapper.__init__(self, env)
		self.k = k
		self.frames = deque([], maxlen=k)
		shp = env.observation_space.shape
		self.observation_space = gym.spaces.Box(low=0, high=255, shape=(shp[0], shp[1], shp[2] * k), dtype=env.observation_space.dtype)

	def reset(self):
		ob = self.env.reset()
		for _ in range(self.k):
			self.frames.append(ob)
		return self._get_ob()

	def step(self, action):
		ob, reward, done, info = self.env.step(action)
		self.frames.append(ob)
		return self._get_ob(), reward, done, info

	def _get_ob(self):
		assert len(self.frames) == self.k
		return LazyFrames(list(self.frames))

class WarpFrame(gym.ObservationWrapper):
	def __init__(self, env):
		"""Warp frames to 84x84 as done in the Nature paper and later
		work."""
		gym.ObservationWrapper.__init__(self, env)
		self.width = 84
		self.height = 84
		self.observation_space = gym.spaces.Box(low=0, high=255,
			shape=(self.height, self.width, 1), dtype=np.bool)

	def observation(self, frame):
		img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
		img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
		img = Image.fromarray(img)
		resized_screen = img.resize((84, 84), Image.BILINEAR)
		resized_screen = np.array(resized_screen)
		return resized_screen[:, :, None]

class FireResetEnv(gym.Wrapper):
	def __init__(self, env=None):
		"""For environments where the user need to press FIRE for the
		game to start."""
		super(FireResetEnv, self).__init__(env)
		assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
		assert len(env.unwrapped.get_action_meanings()) >= 3

	def step(self, action):
		return self.env.step(action)

	def reset(self):
		self.env.reset()
		obs, _, done, _ = self.env.step(1)
		if done:
			self.env.reset()
		obs, _, done, _ = self.env.step(2)
		if done:
			self.env.reset()
		return obs

class EpisodicLifeEnv(gym.Wrapper):
	def __init__(self, env=None):
		"""Make end-of-life == end-of-episode, but only reset on true
		game over. Done by DeepMind for the DQN and co. since it helps
		value estimation."""
		super(EpisodicLifeEnv, self).__init__(env)
		self.lives = 0
		self.was_real_done = True
		self.was_real_reset = False

	def step(self, action):
		obs, reward, done, info = self.env.step(action)
		self.was_real_done = done
		lives = self.env.unwrapped.ale.lives()
		if lives < self.lives and lives > 0:
			done = True
		self.lives = lives
		return obs, reward, done, info

	def reset(self):
		"""Reset only when lives are exhausted.
		This way all states are still reachable even though lives are
		episodic, and the learner need not know about any of this
		behind-the-scenes."""
		if self.was_real_done:
			obs = self.env.reset()
			self.was_real_reset = True
		else:
			obs, _, _, _ = self.env.step(0)
			self.was_real_reset = False
		self.lives = self.env.unwrapped.ale.lives()
		return obs

class MaxAndSkipEnv(gym.Wrapper):
	def __init__(self, env=None, skip=4):
		"""Return only every `skip`-th frame"""
		super(MaxAndSkipEnv, self).__init__(env)
		self._obs_buffer = deque(maxlen=2)
		self._skip = skip

	def step(self, action):
		total_reward = 0.0
		done = None
		for _ in range(self._skip):
			obs, reward, done, info = self.env.step(action)
			self._obs_buffer.append(obs)
			total_reward += reward
			if done:
				break
		max_frame = np.max(np.stack(self._obs_buffer), axis=0)
		return max_frame, total_reward, done, info

	def reset(self):
		"""Clear past frame buffer and init. to first obs. from inner
		env."""
		self._obs_buffer.clear()
		obs = self.env.reset()
		self._obs_buffer.append(obs)
		return obs

class NoopResetEnv(gym.Wrapper):
	def __init__(self, env=None, noop_max=30):
		"""Sample initial states by taking random number of no-ops on
		reset. No-op is assumed to be action 0.
		"""
		super(NoopResetEnv, self).__init__(env)
		self.noop_max = noop_max
		self.override_num_noops = None
		assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

	def step(self, action):
		return self.env.step(action)

	def reset(self):
		""" Do no-op action for a number of steps in [1, noop_max]."""
		self.env.reset()
		if self.override_num_noops is not None:
			noops = self.override_num_noops
		else:
			noops = np.random.randint(1, self.noop_max + 1)
		assert noops > 0
		obs = None
		for _ in range(noops):
			obs, _, done, _ = self.env.step(0)
			if done:
				obs = self.env.reset()
		return obs

# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.nn.functional as F

class DQNbn(nn.Module):
	def __init__(self, in_channels=4, n_actions=14):
		"""
		Initialize Deep Q Network

		Args:
			in_channels (int): number of input channels
			n_actions (int): number of outputs
		"""
		super(DQNbn, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
		self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		self.bn3 = nn.BatchNorm2d(64)
		self.fc4 = nn.Linear(7 * 7 * 64, 512)
		self.head = nn.Linear(512, n_actions)

	def forward(self, x):
		x = x.float() / 255
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))
		x = F.relu(self.fc4(x.view(x.size(0), -1)))
		return self.head(x)

class DQN(nn.Module):
	def __init__(self, in_channels=4, n_actions=14):
		"""
		Initialize Deep Q Network

		Args:
			in_channels (int): number of input channels
			n_actions (int): number of outputs
		"""
		super(DQN, self).__init__()
		self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
		# self.bn1 = nn.BatchNorm2d(32)
		self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
		# self.bn2 = nn.BatchNorm2d(64)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
		# self.bn3 = nn.BatchNorm2d(64)
		self.fc4 = nn.Linear(7 * 7 * 64, 512)
		self.head = nn.Linear(512, n_actions)

	def forward(self, x):
		x = x.float() / 255
		x = F.relu(self.conv1(x))
		x = F.relu(self.conv2(x))
		x = F.relu(self.conv3(x))
		x = x.reshape(x.size(0), -1)
		x = F.relu(self.fc4(x))
		return self.head(x)

from collections import namedtuple
import random

Transition = namedtuple('Transion',
			('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.position = 0

	def push(self, *args):
		if len(self.memory) < self.capacity:
			self.memory.append(None)
		self.memory[self.position] = Transition(*args)
		self.position = (self.position + 1) % self.capacity

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def __len__(self):
		return len(self.memory)

class PrioritizedReplay(object):
	def __init__(self, capacity):
		pass

# -*- coding: utf-8 -*-
import random
import numpy as np
from itertools import count
import gym

import math, time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

#from common.atari_wrapper import *
#from deepq.model import *
#from deepq.replay_buffer import ReplayMemory
from collections import namedtuple

# hyperparameters
lr = 1e-4
INITIAL_MEMORY = 10000
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1
EPS_END = 0.02
EPS_DECAY = 1000000
TARGET_UPDATE = 1000
RENDER = False
MEMORY_SIZE = 10 * INITIAL_MEMORY

Transition = namedtuple('Transion',
			('state', 'action', 'next_state', 'reward'))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize replay memory
memory = ReplayMemory(MEMORY_SIZE)

# create networks
policy_net = DQN(n_actions=4).to(device)
target_net = DQN(n_actions=4).to(device)
target_net.load_state_dict(policy_net.state_dict())

# setup optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=lr)

def select_action(state, steps_done, device):
	sample = random.random()
	eps_threshold = EPS_END + (EPS_START - EPS_END)* \
		math.exp(-1. * steps_done / EPS_DECAY)
	steps_done += 1
	if sample > eps_threshold:
		with torch.no_grad():
			return policy_net(state.to(device)).max(1)[1].view(1,1), steps_done
	else:
		return torch.tensor([[random.randrange(4)]], device=device, dtype=torch.long), steps_done

def optimize_model(device):
	if len(memory) < BATCH_SIZE:
		return
	transitions = memory.sample(BATCH_SIZE)
	batch = Transition(*zip(*transitions))

	actions = tuple((map(lambda a: torch.tensor([[a]], device=device), batch.action)))
	rewards = tuple((map(lambda r: torch.tensor([r], device=device), batch.reward)))

	non_final_mask = torch.tensor(
		tuple(map(lambda s: s is not None, batch.next_state)),
		device=device, dtype=torch.bool)

	non_final_next_states = torch.cat([s for s in batch.next_state
					if s is not None]).to(device)

	state_batch = torch.cat(batch.state).to(device)
	action_batch = torch.cat(actions)
	reward_batch = torch.cat(rewards)

	state_action_values = policy_net(state_batch).gather(1, action_batch)

	next_state_values = torch.zeros(BATCH_SIZE, device=device)
	next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
	expected_state_action_values = (next_state_values * GAMMA) + reward_batch

	loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

	optimizer.zero_grad()
	loss.backward()
	for param in policy_net.parameters():
		param.grad.data.clamp_(-1, 1)
	optimizer.step()

def get_state(obs):
	state = np.array(obs)
	state = state.transpose((2, 0, 1))
	state = torch.from_numpy(state)
	return state.unsqueeze(0)

def train(env, n_episodes, steps_done, device, render=False):
	rewardes=[]
	epis=[]
	y_plot=[]
	for episode in range(n_episodes):
		obs = env.reset()
		state = get_state(obs)
		total_reward = 0.0
		for t in count():
			action, steps_done = select_action(state, steps_done, device)

			if render:
				env.render()

			obs, reward, done, info = env.step(action)

			total_reward += reward

			if not done:
				next_state = get_state(obs)
			else:
				next_state = None

			reward = torch.tensor([reward], device=device)

			memory.push(state, action.to(device), next_state, reward.to(device))
			state = next_state

			if steps_done > INITIAL_MEMORY:
				optimize_model(device)

				if steps_done % TARGET_UPDATE == 0:
					target_net.load_state_dict(policy_net.state_dict())

			if done:
				rewardes.append(total_reward)
				break
		if episode % 20 == 0:
			epis.append(episode)
			print('Total steps: {} \t Episode: {}/{} \t Total reward: {}'.format(steps_done, episode, t, np.mean(rewardes[-20:], 0)))
			y_plot.append(np.mean(rewardes[-20:], 0))
	env.close()
	torch.save(policy_net, "dqn_pong_model")
	torch.save(policy_net.state_dict(), 'checkpoint1.pth')
	plt.plot(epis, y_plot)
	plt.title('Reward graph') 
	plt.ylabel('Marginal rewards') 
	plt.xlabel('episode')
	plt.savefig('go.png')
	
	print("Our model: \n\n", policy_net, '\n')
	print("The state dict keys: \n\n",policy_net.state_dict().keys())
	return

def test(env, n_episodes, policy, device, render=True):
	env = gym.wrappers.Monitor(env, './videos/' + 'dqn_pong_video')
	policy_net = torch.load("dqn_pong_model")
	for episode in range(n_episodes):
		obs = env.reset()
		state = get_state(obs)
		total_reward = 0.0
		for t in count():
			action = policy(state.to(device)).max(1)[1].view(1,1)

			if render:
				env.render()
				time.sleep(0.02)

			obs, reward, done, info = env.step(action)

			total_reward += reward

			if not done:
				next_state = get_state(obs)
			else:
				next_state = None

			state = next_state

			if done:
				print("Finished Episode {} with reward {}".format(episode, total_reward))
				break

		env.close()
	return

import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

#from common.atari_wrapper import *
#from deepq.model import *
#from deepq.learn import *

if __name__ == '__main__':
	# set device
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# create environment
	env = gym.make("PongNoFrameskip-v4")
	env = make_env(env)

	steps_done = 0

	# train model
	train(env, 1400, steps_done, device)

	# test model
	test(env, 5, policy_net, device, render=False)

