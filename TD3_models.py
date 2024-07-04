import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, max_size=5e5):
        self.buffer = []
        self.max_size = int(max_size)
        self.size = 0
    
    def add(self, transition):
        self.size +=1
        # transiton is tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        # delete 1/5th of the buffer when full
        if self.size > self.max_size:
            del self.buffer[0:int(self.size/5)]
            self.size = len(self.buffer)
        
        indexes = np.random.randint(0, len(self.buffer), size=batch_size)
        state, action, reward, next_state, done = [], [], [], [], []
        
        for i in indexes:
            s, a, r, s_, d = self.buffer[i]
            state.append(np.array(s, copy=False))
            action.append(np.array(a, copy=False))
            reward.append(np.array(r, copy=False))
            next_state.append(np.array(s_, copy=False))
            done.append(np.array(d, copy=False))
        
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim, max_action, name, model_ckpt_dir = "tmp/td3"):
        super(ActorNetwork, self).__init__()
        self.name = name
        self.checkpoint_dir = model_ckpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')
        self.max_action = max_action

        self.state_dim = state_dim
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        self.action_dim = action_dim
       
        self.l1 = nn.Linear(self.state_dim, self.fc1_dim)
        self.l2 = nn.Linear(self.fc1_dim, self.fc2_dim)
        self.l3 = nn.Linear(self.fc2_dim, self.action_dim)

        self.to(device)
        
        
    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        a = torch.tanh(self.l3(a)) * self.max_action
        return a
    
    def save_model(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_model(self):
        self.load_state_dict(torch.load(self.checkpoint_file))

        
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, fc1_dim, fc2_dim, name, model_ckpt_dir = "tmp/td3"):
        super(CriticNetwork, self).__init__()
        self.name = name
        self.checkpoint_dir = model_ckpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_td3')

        self.state_dim = state_dim
        self.action_dim = action_dim 
        self.fc1_dim = fc1_dim
        self.fc2_dim = fc2_dim
        
        self.l1 = nn.Linear(state_dim + action_dim, fc1_dim)
        self.l2 = nn.Linear(fc1_dim, fc2_dim)
        self.l3 = nn.Linear(fc2_dim, 1)

        self.to(device)
        
    def forward(self, state, action):
        state_action = torch.cat([state, action], 1)
        
        q = F.relu(self.l1(state_action))
        q = F.relu(self.l2(q))
        q = self.l3(q)
        return q
    
    def save_model(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_model(self):
        self.load_state_dict(torch.load(self.checkpoint_file))