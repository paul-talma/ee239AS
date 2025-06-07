import ipdb
import numpy as np
import random
import torch
import gymnasium as gym
from env_wrapper import EnvWrapper
import model
from replay_buffer import ReplayBufferDQN

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.mps.is_available()
    else "cpu"
)
# environment
env = gym.make("CarRacing-v3", continuous=False, render_mode=None)
env = EnvWrapper(env)

OBS_SPACE = env.observation_space.shape
N_ACTIONS = env.action_space.n


# agent
class Agent:
    def __init__(
        self,
        env: EnvWrapper,
        model: model.Nature_Paper_Conv,
        rb: ReplayBufferDQN,
        batch_size: int = 32,
        gamma: float = 0.99,
        loss: str = "mse_loss",
        lr: float = 0.95,
    ) -> None:
        self.env = env
        self.model = model.to(torch.float32)
        self.rb = rb
        self.action_space = env.action_space.n
        self.batch_size = batch_size
        self.gamma = gamma
        if loss == "mse_loss":
            self.loss = torch.nn.functional.mse_loss
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def sample_action(self, state, epsilon=0.2):
        if random.random() < epsilon:
            return np.random.randint(self.action_space)
        state = torch.tensor(state, device=device, dtype=torch.float32)
        batched_state = state.unsqueeze(0)
        q_vals = self.model(batched_state)
        max_q_vals = torch.argmax(q_vals, dim=1, keepdim=True)
        return max_q_vals.item()

    def train_step(self):
        if len(self.rb) < self.rb.buffer_size:
            return 0

        states, actions, rewards, next_states, dones = self.rb.sample(
            batch_size=self.batch_size, device=device
        )

        # current q_vals
        q_vals = self.model(states)  # (B, A)
        q_vals = q_vals.gather(1, actions.unsqueeze(1)).squeeze()  # (B,)

        # next q_vals
        with torch.no_grad():
            next_q_vals = self.model(next_states)  # (B, A)
        next_q_vals = torch.max(next_q_vals, dim=1)[0].squeeze()  # (B, 1)

        target_vals = rewards + self.gamma * next_q_vals

        self.optimizer.zero_grad()
        l = self.loss(q_vals, target_vals)
        ipdb.set_trace()
        l.backward()
        self.optimizer.step()

        return l.item()


# model
m = model.Nature_Paper_Conv(OBS_SPACE, N_ACTIONS).to(device=device, dtype=torch.float32)

# replay buffer
rb = ReplayBufferDQN(buffer_size=1_000)

# init agent
agent = Agent(env=env, model=m, rb=rb)

# training loop
obs, _ = env.reset()
done = False

while True:
    # sample action
    action = agent.sample_action(obs)

    # take action
    next_obs, reward, terminated, truncated, info = env.step(action)

    # update buffer
    done = terminated or truncated
    rb.add(obs, action, reward, next_obs, done)

    agent.train_step()

    if done:
        next_obs, _ = env.reset()
    obs = next_obs
