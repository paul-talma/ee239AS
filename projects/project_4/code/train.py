import rl_test
import gymnasium as gym
import numpy as np
import model
from replay_buffer import ReplayBufferDQN
import DQN
import utils
from env_wrapper import EnvWrapper

env = gym.make("CarRacing-v3", continuous=False, render_mode="rgb_array")
env.np_random = np.random.RandomState(42)


print("Testing EnvWrapper ...")

rl_test.test_wrapper(EnvWrapper)


print("Testing model ...")
rl_test.test_model_DQN(model.Nature_Paper_Conv)


print("Testing buffer ...")
rl_test.test_DQN_replay_buffer(ReplayBufferDQN)


print("Training DQN ...")
trainerDQN = DQN.DQN(
    EnvWrapper(env),
    model.Nature_Paper_Conv,
    lr=0.00025,
    gamma=0.95,
    buffer_size=100000,
    batch_size=32,
    loss_fn="mse_loss",
    use_wandb=False,
    device="cuda",
    seed=42,
    epsilon_scheduler=utils.exponential_decay(1, 700, 0.1),
    save_path=utils.get_save_path("DQN", "./runs/"),
)

# trainerDQN.train(2, 50, 30, 50, 50)
