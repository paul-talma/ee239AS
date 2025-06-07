import DQN
import wandb
import numpy as np
import utils
from env_wrapper import EnvWrapper
from model import Nature_Paper_Conv
import gymnasium as gym

wandb.login()
wandb.init(project="racing-car-dqn")

env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
env.np_random = np.random.RandomState(42)

trainerDQN = DQN.DQN(
    env=EnvWrapper(env),
    model=Nature_Paper_Conv,
    lr=0.00025,
    gamma=0.95,
    buffer_size=10_000,
    batch_size=32,
    loss_fn="mse_loss",
    use_wandb=True,
    device="mps",
    seed=42,
    epsilon_scheduler=utils.exponential_decay(1, 700, 0.1),
    save_path=utils.get_save_path("DQN", "./runs/"),
)

trainerDQN.train(200, 50, 30, 50, 50)
