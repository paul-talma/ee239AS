import random 
import torch 
import numpy as np

class ReplayBufferDDPG:
    def __init__(self, buffer_size:int, seed:int=42):
        self.buffer_size = buffer_size
        self.seed = seed
        self.buffer = []
        random.seed(self.seed)
    
    def add(self, state:np.ndarray, action:int, reward:float, next_state:np.ndarray
            , done:bool):
        """
        Add a new experience to the buffer

        Args:
            state (np.ndarray): the current state of shape [n_c,h,w]
            action (int): the action taken
            reward (float): the reward received
            next_state (np.ndarray): the next state of shape [n_c,h,w]
            done (bool): whether the episode is done
        """
        self.buffer.append((state, action, reward, next_state, done))
        if len(self.buffer) > self.buffer_size:
            self.buffer.pop(0)


    def sample(self, batch_size: int, device='cpu'):
        """
        Randomly sample a batch of experiences from the replay buffer.

        Args:
            batch_size (int): the number of samples to take

        Returns:
            states (torch.Tensor): a np.ndarray of shape [batch_size,n_c,h,w] of dtype float32
            actions (torch.Tensor): a np.ndarray of shape [batch_size,action_dim] of dtype float 32
            rewards (torch.Tensor): a np.ndarray of shape [batch_size] of dtype float32
            next_states (torch.Tensor): a np.ndarray of shape [batch_size,n_c,h,w] of dtype float32
            dones (torch.Tensor): a np.ndarray of shape [batch_size] of dtype bool
        """
        idx = random.sample(range(len(self.buffer)), batch_size)
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in idx:
            state, action, reward, next_state, done = self.buffer[i]
            states.append(torch.tensor(state).float())
            actions.append(torch.tensor(action).float())
            rewards.append(reward)
            next_states.append(torch.tensor(next_state).float())
            dones.append(done)
        
        states = torch.stack(states).to(device)
        actions = torch.stack(actions).to(device)
        rewards = torch.tensor(rewards).to(device).float()
        next_states = torch.stack(next_states).to(device)
        dones = torch.tensor(dones).to(device)
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        return len(self.buffer)