from TD3_models import *

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

        self.eta = reward_parameters["eta"]
        self.kappa = reward_parameters["kappa"]
        self.lamda = reward_parameters["lamda"]
        self.mu = reward_parameters["mu"]
        self.corridor_penalty = reward_parameters["corridor_penalty"]
        self.far_away_penalty = reward_parameters["far_away_penalty"]
        self.docking_pos_bonus = reward_parameters["docking_pos_bonus"]
        self.docking_vel_bonus = reward_parameters["docking_vel_bonus"]
        self.docking_pos_bonus_scaling = reward_parameters["docking_pos_bonus_scaling"]
        self.docking_vel_bonus_scaling = reward_parameters["docking_vel_bonus_scaling"]

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


    def get_reward(self, state, action):
        pos_state = state[0:3]
        vel_state = state[3:6]
        rel_pos = pos_state - self.port_loc

        tot_reward = 0.0
        rwd_x = self.max_distance - np.abs(rel_pos[0])
        rwd_y = self.max_distance - np.abs(rel_pos[1])
        rwd_z = self.max_distance - np.abs(rel_pos[2])

        vel_rwd = 2*(1 - 5*(np.abs(vel_state[self.dir_index]-self.ideal_dir_vel*-1*self.is_positive)))
        #rwd_position = self.max_distance - np.linalg.norm(rel_pos)  # reward for getting closer
        #rwd_position_heading = -self.eta * np.tanh(self.kappa * np.dot(rel_pos,vel_state)) # reward for moving towards target
        #rwd_taking_no_action = self.lamda*(np.sqrt(3)-np.linalg.norm(action))  # small bonus for not taking any action (to reduce fuel usage and prevent oscillating towards target)
        #tot_reward = rwd_position + rwd_position_heading + rwd_taking_no_action
        #tot_reward = rwd_position + rwd_taking_no_action
        #tot_reward = rwd_position
        tot_reward = (rwd_x+rwd_y+rwd_z)
        #print(rwd_x, rwd_y, rwd_z, rwd_taking_no_action)
        tot_reward /= self.max_pos_sum_rwd

        tot_reward += vel_rwd
        #print(rwd_position, tot_reward)
        #print(tot_reward)
        
        if self.reward_type == "full":
            self.is_docking_flag = self.is_docking(rel_pos)
            self.outside_cone_flag = self.outside_cone(rel_pos)
            self.too_far_flag = np.linalg.norm(rel_pos) > self.max_distance

            # Big penalty (+ termination) if outside docking corridor
            if self.outside_cone_flag:
                tot_reward -= self.corridor_penalty
                print("Sim should terminate: vehicle outside cone")


            # Big penalty (+ termination) if gets too far away
            if self.too_far_flag:
                tot_reward -= self.far_away_penalty
                print("Sim should terminate: vehicle too far away")


        # Termination) if docking position is reached
        if self.is_docking_flag:
            print("Sim should terminate: vehicle is docking!")

            # Bonus depending on position accuracy
            docking_pos_rwd = 0
            docking_pos_rwd += self.docking_pos_bonus - self.docking_pos_bonus_scaling*np.abs(rel_pos[self.offdir_index_1])
            docking_pos_rwd += self.docking_pos_bonus - self.docking_pos_bonus_scaling*np.abs(rel_pos[self.offdir_index_2])
            tot_reward += docking_pos_rwd

            # Bonus depending on relative velocity
            docking_vel_rwd = 0
            docking_vel_rwd += (self.docking_vel_bonus - self.docking_vel_bonus_scaling*np.abs(vel_state[self.offdir_index_1]))
            docking_vel_rwd += (self.docking_vel_bonus - self.docking_vel_bonus_scaling*np.abs(vel_state[self.offdir_index_2]))
            docking_vel_rwd += (self.docking_vel_bonus - self.docking_vel_bonus_scaling*(np.abs(vel_state[self.dir_index])-self.ideal_dir_vel))
            tot_reward += docking_vel_rwd

        #print("\n New epoch")
        #print(state[0:3], state[3:6], action)
        #print(rwd_position, rwd_position_heading, penal_taking_action)
        
        if self.too_far_flag or self.outside_cone_flag or self.is_docking_flag:
            self.is_done = True
        else:
            self.is_done = False

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
                 buffer_size, batch_size, gamma, tau, policy_noise, noise_clip, policy_delay,
                 exploration_noise, approach_direction, reward_type, reward_parameters, docking_ports, docking_settings,
                 save_folder, warmup):
             
        self.actor = ActorNetwork(alpha, state_dim, action_dim, fc1_dim, fc2_dim, max_action, name="actor", model_ckpt_folder=save_folder)
        self.actor_target = ActorNetwork(alpha, state_dim, action_dim, fc1_dim, fc2_dim, max_action, name="target_actor", model_ckpt_folder=save_folder)
        #self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic_1 = CriticNetwork(beta, state_dim, action_dim, fc1_dim, fc2_dim, name="critic_1", model_ckpt_folder=save_folder)
        self.critic_1_target = CriticNetwork(beta, state_dim, action_dim, fc1_dim, fc2_dim, name="target_critic_1", model_ckpt_folder=save_folder)
        #self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        
        self.critic_2 = CriticNetwork(beta, state_dim, action_dim, fc1_dim, fc2_dim, name="critic_2", model_ckpt_folder=save_folder)
        self.critic_2_target = CriticNetwork(beta, state_dim, action_dim, fc1_dim, fc2_dim, name="target_critic_2", model_ckpt_folder=save_folder)
        #self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.replay_buffer = ReplayBuffer(buffer_size, (12,), 3)

        self.max_action = max_action
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
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
        self.warmup = warmup
        self.actions_taken = 0
        self.learnings_done = 0

        self.reward_computer = RewardComputer(self.approach_direction, self.reward_type, self.reward_parameters, 
                                              self.docking_ports, self.docking_settings)
        
        self.update_network_parameters(tau=1)

    
    def compute_action(self, state):
        if self.actions_taken < self.warmup:
            action = torch.FloatTensor(np.random.normal(scale=self.exploration_noise, size=(self.action_dim,))).to(self.actor.device)
        else:
            state = torch.FloatTensor(state).to(self.actor.device)
            action = self.actor.forward(state).to(self.actor.device)

        action_prime = action + torch.FloatTensor(np.random.normal(scale=self.exploration_noise, size=(self.action_dim,))).to(self.actor.device)
        action_prime = torch.clamp(action_prime, -self.max_action, self.max_action)
        self.actions_taken += 1
        
        return action_prime.cpu().detach().numpy()
    
    def remember(self, state, action, reward, new_state, done):
        self.replay_buffer.store_transition(state, action, reward, new_state, done)

    def learn(self):
        if self.replay_buffer.mem_cntr < self.batch_size:
            return 
        state, action, reward, new_state, done = \
                self.replay_buffer.sample_buffer(self.batch_size)

        reward = torch.tensor(reward, dtype=torch.float).to(self.critic_1.device)
        done = torch.tensor(done).to(self.critic_1.device)
        state_ = torch.tensor(new_state, dtype=torch.float).to(self.critic_1.device)
        state = torch.tensor(state, dtype=torch.float).to(self.critic_1.device)
        action = torch.tensor(action, dtype=torch.float).to(self.critic_1.device)
        
        target_actions = self.actor_target.forward(state_)
        target_actions = target_actions + \
                torch.clamp(torch.tensor(np.random.normal(scale=self.policy_noise)), -self.noise_clip, self.noise_clip)
        target_actions = torch.clamp(target_actions, -self.max_action, 
                                self.max_action)
        
        q1_ = self.critic_1_target.forward(state_, target_actions)
        q2_ = self.critic_2_target.forward(state_, target_actions)

        q1 = self.critic_1.forward(state, action)
        q2 = self.critic_2.forward(state, action)

        q1_[done] = 0.0
        q2_[done] = 0.0

        q1_ = q1_.view(-1)
        q2_ = q2_.view(-1)

        critic_value_ = torch.min(q1_, q2_)

        target = reward + self.gamma*critic_value_
        target = target.view(self.batch_size, 1)

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()

        q1_loss = F.mse_loss(target, q1)
        q2_loss = F.mse_loss(target, q2)
        critic_loss = q1_loss + q2_loss
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        self.learnings_done += 1

        if self.learnings_done % self.policy_delay != 0:
            return

        self.actor.optimizer.zero_grad()
        actor_q1_loss = self.critic_1.forward(state, self.actor.forward(state))
        actor_loss = -torch.mean(actor_q1_loss)
        actor_loss.backward()
        self.actor.optimizer.step()

        self.update_network_parameters()

    def update_network_parameters(self, tau=None):
        if tau is None:
            tau = self.tau

        actor_params = self.actor.named_parameters()
        critic_1_params = self.critic_1.named_parameters()
        critic_2_params = self.critic_2.named_parameters()
        target_actor_params = self.actor_target.named_parameters()
        target_critic_1_params = self.critic_1_target.named_parameters()
        target_critic_2_params = self.critic_2_target.named_parameters()

        critic_1 = dict(critic_1_params)
        critic_2 = dict(critic_2_params)
        actor = dict(actor_params)
        target_actor = dict(target_actor_params)
        target_critic_1 = dict(target_critic_1_params)
        target_critic_2 = dict(target_critic_2_params)

        for name in critic_1:
            critic_1[name] = tau*critic_1[name].clone() + \
                    (1-tau)*target_critic_1[name].clone()

        for name in critic_2:
            critic_2[name] = tau*critic_2[name].clone() + \
                    (1-tau)*target_critic_2[name].clone()

        for name in actor:
            actor[name] = tau*actor[name].clone() + \
                    (1-tau)*target_actor[name].clone()

        self.critic_1_target.load_state_dict(critic_1)
        self.critic_2_target.load_state_dict(critic_2)
        self.actor_target.load_state_dict(actor)

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

    def decrease_lr(self, factor):
        for g in self.actor.optimizer.param_groups:
            g['lr'] = g['lr']/factor
        for g in self.actor_target.optimizer.param_groups:
            g['lr'] = g['lr']/factor
        for g in self.critic_1.optimizer.param_groups:
            g['lr'] = g['lr']/factor
        for g in self.critic_1_target.optimizer.param_groups:
            g['lr'] = g['lr']/factor
        for g in self.critic_2.optimizer.param_groups:
            g['lr'] = g['lr']/factor        
        for g in self.critic_2_target.optimizer.param_groups:
            g['lr'] = g['lr']/factor
