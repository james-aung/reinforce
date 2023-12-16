import numpy as np

from collections import deque

import matplotlib.pyplot as plt

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

# Gym
from gym.wrappers import RecordVideo
import gym
import gym_pygame

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Should show cuda:0 if using gpu
print(device)

"""Reinforce"""

def reinforce(policy, optimizer, n_training_episodes, max_t, gamma, print_every):

    scores_deque = deque(maxlen=100)
    scores = []

    # Repeat for number of training episodes
    for i_episode in range(1, n_training_episodes+1):
        state = env.reset()

        saved_log_probs = []
        rewards = []

        # Progress through timesteps until max timesteps
        for t in range(max_t):
            action, log_prob = policy.act(state) # Policy acting implementation returns these 2 things
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break

        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # We now calculate the return of each timestep
        returns = deque(maxlen=max_t)
        n_steps = len(rewards)

        for t in range(n_steps)[::-1]: # Reverse through array
            disc_return_t = (returns[0] if len(returns)>0 else 0) # Get the most recently saved return
            returns.appendleft(gamma*disc_return_t + rewards[t]) # Save the return to the beginning of the deque

        returns = torch.tensor(returns)
        eps = np.finfo(np.float32).eps.item() # Adding epsilon to avoid division by 0
        returns = (returns - returns.mean()) / (returns.std() + eps) # Normalising returns

        policy_loss = []
        for log_prob, disc_return in zip(saved_log_probs, returns): # For each return associated with a timestep
            policy_loss.append(-log_prob * disc_return)
        policy_loss = torch.cat(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    return scores

def evaluate_agent(env, max_steps, n_eval_episodes, policy):
  """
  Evaluate the agent for ``n_eval_episodes`` episodes and returns average reward and std of reward.
  :param env: The evaluation environment
  :param n_eval_episodes: Number of episode to evaluate the agent
  :param policy: The Reinforce agent
  """
  episode_rewards = []
  for episode in range(n_eval_episodes):
    state = env.reset()
    step = 0
    done = False
    total_rewards_ep = 0

    for step in range(max_steps):
      action, _ = policy.act(state)
      new_state, reward, done, info = env.step(action)
      total_rewards_ep += reward

      if done:
        break
      state = new_state
    episode_rewards.append(total_rewards_ep)
  mean_reward = np.mean(episode_rewards)
  std_reward = np.std(episode_rewards)

  return mean_reward, std_reward

"""Environment and Policy"""

def record_episode(episode_id):
    return episode_id % 1000 == 0

env_id = "Pixelcopter-PLE-v0"
env = gym.make(env_id)
eval_env = gym.make(env_id)
env = RecordVideo(env, video_folder='./video', episode_trigger=record_episode)
s_size = env.observation_space.shape[0]
a_size = env.action_space.n

print("_____OBSERVATION SPACE_____ \n")
print("The State Space is: ", s_size)
print("Sample observation", env.observation_space.sample()) # Get a random observation

print("\n _____ACTION SPACE_____ \n")
print("The Action Space is: ", a_size)
print("Action Space Sample", env.action_space.sample()) # Take a random action

print(env.observation_space.shape)

class Policy(nn.Module):
    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, h_size*2)
        self.fc3 = nn.Linear(h_size*2, a_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)

pixelcopter_hyperparameters = {
    "h_size": 64,
    "n_training_episodes": 10000,
    "n_evaluation_episodes": 10,
    "max_t": 1000,
    "gamma": 1.0,
    "lr": 1e-4,
    "env_id": env_id,
    "state_space": s_size,
    "action_space": a_size,
}

# Create policy and place it to the device
pixelcopter_policy = Policy(pixelcopter_hyperparameters["state_space"], pixelcopter_hyperparameters["action_space"], pixelcopter_hyperparameters["h_size"]).to(device)
pixelcopter_optimizer = optim.Adam(pixelcopter_policy.parameters(), lr=pixelcopter_hyperparameters["lr"])

scores = reinforce(pixelcopter_policy,
                   pixelcopter_optimizer,
                   pixelcopter_hyperparameters["n_training_episodes"],
                   pixelcopter_hyperparameters["max_t"],
                   pixelcopter_hyperparameters["gamma"],
                   100, # print every
                   )

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()

evaluate_agent(
    eval_env, pixelcopter_hyperparameters["max_t"], pixelcopter_hyperparameters["n_evaluation_episodes"], pixelcopter_policy
)