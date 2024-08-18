import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ReplayBuffer():
    def __init__(self, max_size, input_shape, n_actions):
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size, *input_shape))
        self.new_state_memory = np.zeros((self.mem_size, *input_shape))
        self.action_memory = np.zeros((self.mem_size, n_actions))
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=bool)

    def store_transition(self, state, action, reward, state_, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = done
        self.reward_memory[index] = reward
        self.action_memory[index] = action

        self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)

        batch = np.random.choice(max_mem, batch_size)

        states = self.state_memory[batch]
        states_ = self.new_state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        dones = self.terminal_memory[batch]

        return states, actions, rewards, states_, dones


class ActorNetwork(nn.Module):
    def __init__(self, alpha, state_dim, action_dim, fc1_dim, fc2_dim, max_action, name, model_ckpt_folder):
        super(ActorNetwork, self).__init__()
        self.name = name
        self.checkpoint_dir = model_ckpt_folder
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')
        self.max_action = max_action

        self.state_dim = state_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.action_dim = action_dim
       
        self.l1 = nn.Linear(self.state_dim, self.fc1_dim)
        self.l2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.l3 = nn.Linear(self.fc2_dim, self.action_dim)

        self.device = self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.to(self.device)
        
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        #a = F.sigmoid(self.l1(state))
        #a = F.sigmoid(self.l2(a))
        a = torch.tanh(self.l3(a))
        return a
    
    def save_model(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_model(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

        
class CriticNetwork(nn.Module):
    def __init__(self, beta, state_dim, action_dim, fc1_dim, fc2_dim, name, model_ckpt_folder):
        super(CriticNetwork, self).__init__()
        self.name = name
        self.checkpoint_dir = model_ckpt_folder
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        
        self.l1 = nn.Linear(state_dim + action_dim, fc1_dim)
        self.l2 = nn.Linear(fc1_dim, fc2_dim)
        self.l3 = nn.Linear(fc2_dim, 1)

        self.device = self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.to(self.device)
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], dim=1)
        
        q = F.relu(self.l1(state_action))
        q = F.relu(self.l2(q))
        #q = F.sigmoid(self.l1(state_action))
        #q = F.sigmoid(self.l2(q))
        q = self.l3(q)
        return q
    
    def save_model(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_model(self):
        self.load_state_dict(torch.load(self.checkpoint_file))