from TD3_models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RewardComputer():
    def __init__(self, approach_direction, reward_type):
        self.docking_ports = {  # w.r.t. to the COM of the Target vehicle in its TNW frame 
            "pos_R-bar": np.array([-2, -2 , 0])
        }

        self.reward_function = None
        self.port_loc = None
        self.last_rel_pos = None

        self.eta = 0.1
        self.kappa = 0.1
        self.lamda = 0.1  # tanh scaling
        self.mu = 0.05
        #self.delta = 100

        if approach_direction == "pos_R-bar":
            if reward_type == "simple":
                self.reward_function = self.pos_R_simple
                self.port_loc = self.docking_ports[approach_direction]

    def set_initial_rel_pos(self, state):
        self.last_rel_pos = state[0:3] - self.port_loc
 
    def pos_R_simple (self, state, action):
        rel_pos = state[0:3] - self.port_loc
        rwd_position = -self.eta * np.linalg.norm(rel_pos)                          # reward for getting closer
        rwd_position_heading = -self.kappa * np.tanh(self.lamda * np.dot(rel_pos,state[3:6]))
        penal_taking_action = -self.mu * np.linalg.norm(action)
        return rwd_position + rwd_position_heading + penal_taking_action

    def compute_final_reward_bonus(self, state):
        return


    def get_reward(self, state, action):
        return self.reward_function(state, action)


class Agent():
    def __init__(self, alpha, beta, state_dim, action_dim, fc1_dim, fc2_dim, max_action,
                 batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay,
                 exploration_noise, approach_direction, reward_type):
        
        self.actor = ActorNetwork(state_dim, action_dim, fc1_dim, fc2_dim, max_action, name="actor")
        self.actor_target = ActorNetwork(state_dim, action_dim, fc1_dim, fc2_dim, max_action, name="target_actor")
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        
        self.critic_1 = CriticNetwork(state_dim, action_dim, fc1_dim, fc2_dim, name="critic_1")
        self.critic_1_target = CriticNetwork(state_dim, action_dim, fc1_dim, fc2_dim, name="target_critic_1")
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=beta)
        
        self.critic_2 = CriticNetwork(state_dim, action_dim, fc1_dim, fc2_dim, name="critic_2")
        self.critic_2_target = CriticNetwork(state_dim, action_dim, fc1_dim, fc2_dim, name="target_critic_2")
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=beta)
        
        self.replay_buffer = ReplayBuffer()

        self.max_action = max_action
        self.batch_size = batch_size
        self.gamma = gamma
        self.polyak = polyak
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.exploration_noise = exploration_noise
        self.episode_reward = 0
        self.approach_direction = approach_direction
        self.reward_type = reward_type

        self.reward_computer = RewardComputer(self.approach_direction, self.reward_type)
        
    
    def compute_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def update(self, n_iter):
        for i in range(n_iter):
            # Sample a batch of transitions from replay buffer:
            state, action_, reward, next_state, done = self.replay_buffer.sample(self.batch_size)
            state = torch.FloatTensor(state).to(device)
            action = torch.FloatTensor(action_).to(device)
            reward = torch.FloatTensor(reward).reshape((self.batch_size,1)).to(device)
            next_state = torch.FloatTensor(next_state).to(device)
            done = torch.FloatTensor(done).reshape((self.batch_size,1)).to(device)
            
            # Select next action according to target policy:
            noise = torch.FloatTensor(action_).data.normal_(0, self.policy_noise).to(device)
            noise = noise.clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise)
            next_action = next_action.clamp(-self.max_action, self.max_action)
            
            # Compute target Q-value:
            target_Q1 = self.critic_1_target(next_state, next_action)
            target_Q2 = self.critic_2_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + ((1-done) * self.gamma * target_Q).detach()
            
            # Optimize Critic 1:
            current_Q1 = self.critic_1(state, action)
            loss_Q1 = F.mse_loss(current_Q1, target_Q)
            self.critic_1_optimizer.zero_grad()
            loss_Q1.backward()
            self.critic_1_optimizer.step()
            
            # Optimize Critic 2:
            current_Q2 = self.critic_2(state, action)
            loss_Q2 = F.mse_loss(current_Q2, target_Q)
            self.critic_2_optimizer.zero_grad()
            loss_Q2.backward()
            self.critic_2_optimizer.step()
            
            # Delayed policy updates:
            if i % self.policy_delay == 0:
                # Compute actor loss:
                actor_loss = -self.critic_1(state, self.actor(state)).mean()
                
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Polyak averaging update:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_( (self.polyak * target_param.data) + ((1-self.polyak) * param.data))
                
                for param, target_param in zip(self.critic_1.parameters(), self.critic_1_target.parameters()):
                    target_param.data.copy_( (self.polyak * target_param.data) + ((1-self.polyak) * param.data))
                
                for param, target_param in zip(self.critic_2.parameters(), self.critic_2_target.parameters()):
                    target_param.data.copy_( (self.polyak * target_param.data) + ((1-self.polyak) * param.data))

    def save_models(self):
        print('... saving model checkpoint ...')
        self.actor.save_model()
        self.actor_target.save_model()
        self.critic_1.save_model()
        self.critic_2.save_model()
        self.critic_1_target.save_model()
        self.critic_2_target.save_model()


    def load_models(self):
        print('... loading model checkpoint ...')
        self.actor.load_model()
        self.actor_target.load_model()
        self.critic_1.load_model()
        self.critic_2.load_model()
        self.critic_1_target.load_model()
        self.critic_2_target.load_model()

