from TD3_models import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class RewardComputer():
    def __init__(self, approach_direction, reward_type, reward_parameters, docking_ports, docking_settings):
        self.docking_ports = docking_ports

        self.reward_function = None
        self.port_loc = None
        self.last_rel_pos = None

        self.is_docking_flag = False
        self.outside_cone_flag = False
        self.too_far_flag = False

        self.reward_type = reward_type

        self.KOS_size = docking_settings["KOS_size"]
        self.corridor_angle = docking_settings["corridor_angle"]
        self.corridor_base_radius = docking_settings["corridor_base_radius"]
        self.max_distance = docking_settings["max_distance"]
        self.max_offdir_pos = docking_settings["max_offdir_pos"]
        self.max_offdir_vel = docking_settings["max_offdir_vel"]
        self.min_dir_vel = docking_settings["min_dir_vel"]
        self.max_dir_vel = docking_settings["max_dir_vel"]
        self.ideal_dir_vel = docking_settings["ideal_dir_vel"]

        #self.eta = reward_parameters["eta"]
        #self.kappa = reward_parameters["kappa"]
        #self.lamda = reward_parameters["lamda"]
        #self.mu = reward_parameters["mu"]
        self.corridor_penalty = reward_parameters["corridor_penalty"]
        self.far_away_penalty = reward_parameters["far_away_penalty"]
        self.docking_pos_bonus = reward_parameters["docking_pos_bonus"]
        self.docking_vel_bonus = reward_parameters["docking_vel_bonus"]
        #self.docking_pos_bonus_scaling = reward_parameters["docking_pos_bonus_scaling"]
        #self.docking_vel_bonus_scaling = reward_parameters["docking_vel_bonus_scaling"]


        self.offir_pos_w = reward_parameters["offir_pos_w"]
        self.dir_pos_w = reward_parameters["dir_pos_w"]
        self.offdir_rate_w = reward_parameters["offdir_rate_w"]
        self.dir_rate_w = reward_parameters["dir_rate_w"]
        self.dir_rate_s = reward_parameters["dir_rate_s"]

        self.conditinal_reward_w = reward_parameters["conditinal_reward_w"]

        self.last_rel_pos = None

        offdir_indices = [0, 1, 2]
        self.is_done = False

        if approach_direction == "pos_R-bar":
                self.port_loc = self.docking_ports[approach_direction]
                #self.port_loc[1] = -15

                # some logic to determine what elements of position array are needed to compute approach corridor
                self.dir_index = 1
                self.is_positive = -1   # = 1 if the approach happens from positive direction in TNW coordinates, else(-1)
                offdir_indices.remove(self.dir_index) 
                self.offdir_index_1 = offdir_indices[0]
                self.offdir_index_2 = offdir_indices[1]
                self.max_pos_sum_rwd = 3*self.max_distance


    def outside_cone(self, pos):
        if np.linalg.norm(pos) < self.KOS_size:
            if np.sqrt(pos[self.offdir_index_1]**2 + pos[self.offdir_index_2]**2) > self.corridor_base_radius + self.is_positive*pos[self.dir_index]*np.tan(self.corridor_angle):
                return True
            else:
                return False
        else:
            return False
        
    def is_docking(self, pos):
        #print(pos)
        if np.linalg.norm(pos) < self.KOS_size:
            #print(self.is_positive, pos[self.dir_index])
            if self.is_positive*pos[self.dir_index] < 0:
                return True
            else:
                return False
        else:
            return False
        
    def store_first_position_state(self, state):
        pos_state = state[0:3]
        rel_pos = pos_state - self.port_loc
        self.last_rel_pos = rel_pos


    def get_reward(self, state, action):
        pos_state = state[0:3]
        vel_state = state[3:6]
        rel_pos = pos_state - self.port_loc

        tot_reward = 0.0

        # subreward 1: reward based on current position and attitude
        state_reward = 0
        # translation
        state_reward += (1-self.offir_pos_w*np.abs(rel_pos[0]))
        state_reward += (1-self.dir_pos_w*np.abs(rel_pos[1]))
        state_reward += (1-self.offir_pos_w*np.abs(rel_pos[2]))
        # rotation
        #state_reward += 3

        # subreward 2: reward based on position and attitude rates
        rate_reward = 0
        # translation
        rate_reward += -self.offdir_rate_w*(np.abs(rel_pos[0]) - np.abs(self.last_rel_pos[0]))
        #rate_reward += -self.dir_rate_w*(np.abs(rel_pos[1]) - np.abs(self.last_rel_pos[1]))
        rate_reward += self.dir_rate_w*(1 - self.dir_rate_s*(np.abs(vel_state[self.dir_index]-self.ideal_dir_vel*-1*self.is_positive)))
        rate_reward += -self.offdir_rate_w*(np.abs(rel_pos[2]) - np.abs(self.last_rel_pos[2]))
        # rotation
        #rate_reward += 3

        # conditional reward 1: heading to ref while close to ref
        heading_and_close_reward = 0.0
        # translation
        for i in range(3):
            if np.abs(rel_pos[i]) - np.abs(self.last_rel_pos[i]) <= 0.0 and np.abs(rel_pos[i]) < 0.4:
                heading_and_close_reward += self.conditinal_reward_w
        # rotation
        #heading_and_close_reward += 3*self.conditinal_reward_w4

        failure_reward = 0.0
        if self.reward_type == "full":
            self.is_docking_flag = self.is_docking(rel_pos)
            self.outside_cone_flag = self.outside_cone(rel_pos)
            self.too_far_flag = np.linalg.norm(rel_pos) > self.max_distance

            # Big penalty (+ termination) if outside docking corridor
            if self.outside_cone_flag:
                failure_reward += -1*self.corridor_penalty
                #tot_reward -= self.corridor_penalty
                print("Sim should terminate: vehicle outside cone")


            # Big penalty (+ termination) if gets too far away
            if self.too_far_flag:
                failure_reward += -1*self.far_away_penalty
                #tot_reward -= self.far_away_penalty
                print("Sim should terminate: vehicle too far away")

        docking_reward = 0.0
        # Termination) if docking position is reached
        if self.is_docking_flag:
            print("Sim should terminate: vehicle is docking!")
            docking_reward += self.docking_pos_bonus
            # Bonus depending on position accuracy
            docking_pos_rwd = 0
            docking_pos_rwd += self.docking_pos_bonus*(1-np.abs(rel_pos[self.offdir_index_1])/self.max_offdir_pos)
            docking_pos_rwd += self.docking_pos_bonus*(1-np.abs(rel_pos[self.offdir_index_2])/self.max_offdir_pos)
            docking_reward += docking_pos_rwd

            # Bonus depending on relative velocity
            docking_vel_rwd = 0
            docking_vel_rwd += self.docking_vel_bonus*(1 - np.abs(vel_state[self.offdir_index_1])/self.max_dir_vel)
            docking_vel_rwd += self.docking_vel_bonus*(1 - np.abs(vel_state[self.offdir_index_2])/self.max_dir_vel)
            docking_vel_rwd += self.docking_vel_bonus*(1 - (np.abs(vel_state[self.dir_index]+self.ideal_dir_vel*self.is_positive)/self.ideal_dir_vel))
            docking_reward += docking_vel_rwd

        #print("\n New epoch")
        #print(state[0:3], state[3:6], action)
        #print(rwd_position, rwd_position_heading, penal_taking_action)
        
        if self.too_far_flag or self.outside_cone_flag or self.is_docking_flag:
            self.is_done = True
        else:
            self.is_done = False

        tot_reward += state_reward + rate_reward + heading_and_close_reward + failure_reward + docking_reward
        self.last_rel_pos = rel_pos
        

        return tot_reward, self.is_done
    
    # Functions to serve merely as interface to tell tudat to terminate simulation if docking occurs
    # or if the vehicle gets outside the cone
    def is_docking_interface(self, time:float):
        return self.is_docking_flag
    
    def outside_cone_interface(self, time:float):
        return self.outside_cone_flag
    
    def too_far_interface(self, time:float):
        return self.too_far_flag
    

class Agent():
    def __init__(self, alpha, beta, state_dim, action_dim, fc1_dim, fc2_dim, max_action,
                 buffer_size, batch_size, gamma, polyak, policy_noise, noise_clip, policy_delay,
                 exploration_noise, approach_direction, reward_type, reward_parameters, docking_ports, docking_settings,
                 save_folder):
        
        self.actor = ActorNetwork(state_dim, action_dim, fc1_dim, fc2_dim, max_action, name="actor", model_ckpt_folder=save_folder)
        self.actor_target = ActorNetwork(state_dim, action_dim, fc1_dim, fc2_dim, max_action, name="target_actor", model_ckpt_folder=save_folder)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=alpha)
        
        self.critic_1 = CriticNetwork(state_dim, action_dim, fc1_dim, fc2_dim, name="critic_1", model_ckpt_folder=save_folder)
        self.critic_1_target = CriticNetwork(state_dim, action_dim, fc1_dim, fc2_dim, name="target_critic_1", model_ckpt_folder=save_folder)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=beta)
        
        self.critic_2 = CriticNetwork(state_dim, action_dim, fc1_dim, fc2_dim, name="critic_2", model_ckpt_folder=save_folder)
        self.critic_2_target = CriticNetwork(state_dim, action_dim, fc1_dim, fc2_dim, name="target_critic_2", model_ckpt_folder=save_folder)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=beta)
        
        self.replay_buffer = ReplayBuffer(buffer_size)

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
        self.reward_parameters = reward_parameters
        self.docking_ports = docking_ports
        self.docking_settings = docking_settings
        self.action_dim = action_dim

        self.reward_computer = RewardComputer(self.approach_direction, self.reward_type, self.reward_parameters, 
                                              self.docking_ports, self.docking_settings)
        
    
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
                
                #   averaging update:
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

