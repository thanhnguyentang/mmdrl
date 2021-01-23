import sys,os 
import numpy as np 
import tensorflow as tf 
import tensorflow_probability as tfp 
import random 
from scipy.stats import norm, bernoulli 
import collections 
from copy import deepcopy

from dopamine.agents.dqn import dqn_agent
from dopamine.utils import gym_lib
from dopamine.replay_memory import prioritized_replay_buffer, circular_replay_buffer
from particle_net import ParticleDQNet
import gin.tf

ReplayElement = (
    collections.namedtuple('shape_type', ['name', 'shape', 'type']))


def beta_identity_fn(training_steps, init_value=1):
    return init_value 

def beta_intrinsic_suppress_fn(training_steps, init_value=50):
    return init_value * np.sqrt( np.log( training_steps +3.) / (training_steps + 3.)  )

@gin.configurable
def bandwidth_linear_annealing_fn(training_steps, init_value=10., final_value=1., annealing_period=1000000):
    if training_steps >= annealing_period:
        return final_value
    else:
        return (training_steps / annealing_period) * final_value + (1. - training_steps / annealing_period) * init_value 


def gaussian_rbf_kernel(d, sigmas):
    """
    Args:
        d: (b,n,n) squared distance.
        sigmas: a list of sigma values.  
    """
    tf_sigmas = tf.constant(sigmas, dtype=tf.float32) 
    h = 1./ tf.reshape(tf_sigmas, [-1,1]) # (k,1)
    s = tf.matmul(h, tf.reshape(d, [1,-1])) # (k,b*n*n) 
    return tf.reshape(tf.reduce_sum(tf.exp(-s), axis=0), shape=tf.shape(d)) #(b,n,n) 
    
def hubber_loss(u,kappa=1):
    if kappa == 0:
        return tf.abs(u) 
    else:
        huber_loss_case_one = tf.to_float(tf.abs(u) <= kappa) * 0.5 * u ** 2
        huber_loss_case_two = tf.to_float(tf.abs(u) > kappa) * kappa * (tf.abs(u) - 0.5 * kappa)
        return huber_loss_case_one + huber_loss_case_two 

@gin.configurable 
class MMDAgent(dqn_agent.DQNAgent):
    """MMDAgent inherited from DQN agent.
    """
    def __init__(self,
                sess,
                num_actions=4, 
                num_atoms=10,
                delta=0.1, 
                kappa=0, 
                target_estimator='mean', 
                policy='eps_greedy', 
                beta_fn=beta_intrinsic_suppress_fn, 
                bandwidth_annealing_fn=bandwidth_linear_annealing_fn, 
                bandwidth_selection_type='med', 
                debug=False, 
                double_dqn=False, 
                observation_shape=dqn_agent.NATURE_DQN_OBSERVATION_SHAPE, 
                observation_dtype=dqn_agent.NATURE_DQN_DTYPE, 
                stack_size=dqn_agent.NATURE_DQN_STACK_SIZE, 
                network=ParticleDQNet, 
                gamma=0.99, 
                update_horizon=1,
                min_replay_history=20000,
                update_period=4, 
                target_update_period=8000,
                monitor_step=10000, 
                epsilon_fn=dqn_agent.linearly_decaying_epsilon,
                epsilon_train=0.01, 
                epsilon_eval=0.001,
                epsilon_decay_period=250000,
                replay_scheme='prioritized',
                tf_device='/cpu:*',
                use_staging=True,
                optimizer=tf.train.AdamOptimizer(
                    learning_rate=0.00025, epsilon=0.0003125
                ),
                summary_writer=None, 
                summary_writing_frequency=500):
        """
        Args:
            bandwidth_selection_type: str, [med/annealing/mixture/const]

        """

        self._replay_scheme = replay_scheme
        self._double_dqn = double_dqn 
        self._debug = debug
        self._num_atoms = num_atoms
        self._policy = policy 
        self._target_estimator = target_estimator 
        self._delta = delta 
        self._action_sampler = ParticlePolicy(delta) 
        self._beta_ph = tf.placeholder(tf.float32, (), name='beta_schedule')
        self.beta_fn = beta_fn
        self.bandwidth_annealing_fn=bandwidth_annealing_fn
        self.h_ph = tf.placeholder(tf.float32, (), name='h')
        self.bandwidth_selection_type = bandwidth_selection_type
        self.kappa = kappa 

        if debug:
            self.statistics_collection = []

        super(MMDAgent, self).__init__(
            sess=sess, 
            num_actions=num_actions, 
            observation_shape=observation_shape,
            observation_dtype=observation_dtype,
            stack_size=stack_size, 
            network=network,
            gamma=gamma,
            update_horizon=update_horizon,
            min_replay_history=min_replay_history,
            update_period=update_period,
            target_update_period=target_update_period,
            epsilon_fn=epsilon_fn, 
            epsilon_train=epsilon_train,
            epsilon_eval=epsilon_eval,
            epsilon_decay_period=epsilon_decay_period,
            tf_device=tf_device,
            use_staging=use_staging,
            optimizer=optimizer, 
            summary_writer=summary_writer,
            summary_writing_frequency=summary_writing_frequency) 
    
    def _create_network(self, name):
        return self.network(num_actions = self.num_actions, num_atoms=self._num_atoms, name=name)

    def _build_replay_buffer(self, use_staging):
        if self._replay_scheme not in ['uniform', 'prioritized']:
            raise ValueError('Invalid replay scheme: {}'.format(self._replay_scheme))
        return prioritized_replay_buffer.WrappedPrioritizedReplayBuffer(
            observation_shape=self.observation_shape,
            stack_size=self.stack_size,
            use_staging=use_staging,
            update_horizon=self.update_horizon,
            gamma=self.gamma,
            observation_dtype=self.observation_dtype.as_numpy_dtype)

    def _store_transition(self,
                        last_observation,
                        action,
                        reward,
                        is_terminal,
                        priority=None):
        if priority is None:
            if self._replay_scheme == 'uniform':
                priority = 1.
            else:
                priority = self._replay.memory.sum_tree.max_recorded_priority

        if not self.eval_mode:
            self._replay.add(last_observation, action, reward, is_terminal, priority)     

    def _build_target_particles(self):
        batch_size = self._replay.batch_size
        rewards = self._replay.rewards[:, None, None] #(bs,1,1)
        is_terminal_multiplier = 1. - tf.cast(self._replay.terminals, tf.float32)
        gamma_with_terminal = self.cumulative_gamma * is_terminal_multiplier 
        gamma_with_terminal = gamma_with_terminal[:, None, None] #(bs,1,1)
        next_particles = self._replay_next_target_net_outputs.particles #(bs,na,n) 
        target_particles = rewards + gamma_with_terminal * next_particles #(bs,na,n)
        return self._action_sampler.compute_target(target_particles, estimator=self._target_estimator)

    def _build_networks(self):
        self.online_convnet = self._create_network(name='Online')
        self.target_convnet = self._create_network(name='Target')
        self._net_outputs = self.online_convnet(self.state_ph)

        # action_prob = tf.cast(tf.equal(tf.reduce_max(self._net_outputs.q_values, axis=1)[:,None], self._net_outputs.q_values), tf.float32) 
        # action_prob = action_prob / tf.reduce_sum(action_prob, axis=-1) 
        # self._q_argmax = tfp.distributions.Categorical(probs=action_prob).sample(1)[:,0][0]
        self._q_argmax = self._action_sampler.draw_action(self._net_outputs.particles, 'mean')[0]
        self._q_argmax_explore = self._action_sampler.draw_action(self._net_outputs.particles, self._policy, beta=self._beta_ph)[0]
        self._replay_net_outputs = self.online_convnet(self._replay.states)
        self._replay_next_target_net_outputs = self.target_convnet(self._replay.next_states)

    def _build_train_op(self): # *Required.
        target_particles = tf.stop_gradient(self._build_target_particles()) #(bs,n)
        indices = tf.range(tf.shape(self._replay_net_outputs.particles)[0])[:, None] #(bs,1)
        reshaped_actions = tf.concat([indices, self._replay.actions[:, None]], 1) #(bs,2)
        chosen_action_particles = tf.gather_nd(self._replay_net_outputs.particles, reshaped_actions) #(bs,n)

        # h = tf.gather_nd(self._replay_net_outputs.h, reshaped_actions)[:,None,None] + 1e-5 #(b,)

        if self.kappa == 0:
            d1 = tf.square(chosen_action_particles[:,:, None] - chosen_action_particles[:,None,:]) # (b,n,n)
            d2 = tf.square(chosen_action_particles[:,:, None] - target_particles[:,None,:]) # (b,n,n)
            d3 = tf.square(target_particles[:,:, None] - target_particles[:,None,:]) # (b,n,n)
        else:
            d1 = hubber_loss(chosen_action_particles[:,:, None] - chosen_action_particles[:,None,:], kappa=self.kappa) # (b,n,n)
            d2 = hubber_loss(chosen_action_particles[:,:, None] - target_particles[:,None,:], kappa=self.kappa) # (b,n,n)
            d3 = hubber_loss(target_particles[:,:, None] - target_particles[:,None,:], kappa=self.kappa) # (b,n,n)

        tf.logging.info('Bandwith type: {}.'.format(self.bandwidth_selection_type))
        
        if self.bandwidth_selection_type == 'mixture': # Mixute of kernels from MMD GAN
            batch_size = chosen_action_particles.shape.as_list()[0]

            # sigmas = [1e-3, 1e-2, 1e-1, 1, 2, 4, 5, 10, 20]
            # sigmas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 4, 5, 10, 15, 20, 25, 30, 100]
            # sigmas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 2, 4, 5, 10, 15, 20, 25, 30, 50, 80, 100]
            # sigmas = [1.]
            sigmas = [1,2,3,4,5,6,7,8,9,10]
            # sigmas = [0.1,0.3,1,3,10]
            # sigmas = [1,2,4,8,10]
            # sigmas = [1,2,3,4,5]

            h = sigmas[-1] # for debug only
            xixj = gaussian_rbf_kernel(d1, sigmas) 
            xixj = tf.reduce_mean(tf.reduce_mean(xixj, axis=-1), axis=-1) # (b.)

            xiyj = gaussian_rbf_kernel(d2, sigmas)
            xiyj = tf.reduce_mean(tf.reduce_mean(xiyj, axis=-1), axis=-1) #(b,)

            yiyj = gaussian_rbf_kernel(d3, sigmas) 
            yiyj = tf.reduce_mean(tf.reduce_mean(yiyj, axis=-1), axis=-1)
            mmd_squared = xixj + yiyj - 2*xiyj

            mmd_squared = tf.compat.v1.where(mmd_squared > 0, mmd_squared, tf.constant([0.] * batch_size), name='value')

            loss = tf.reduce_mean(mmd_squared)
            
            train_op = self.optimizer.minimize(loss)
        else:
            if self.bandwidth_selection_type == 'med': # Use median trick
                batch_size = chosen_action_particles.shape.as_list()[0] 
                # aggregated_d = d1
                aggregated_d = tf.concat(  [ tf.reshape(d1[:,:,0], [batch_size,-1]), tf.reshape(d2, [batch_size, -1]), tf.reshape(d3[:,:,0], [batch_size,-1])], axis=1)
                # aggregated_d = tf.concat(  [ tf.reshape(d1[:,:,0], [batch_size,-1]), tf.reshape(d2, [batch_size, -1])], axis=1)

                med_sq = tfp.stats.percentile(tf.reshape(aggregated_d, [batch_size,-1]), q=50., axis=1, interpolation='midpoint')  # (bs,)
                # # h = med_sq / tf.log(2*self._num_atoms + 1.)
                h = 2*med_sq #* 0.002 
                h = tf.stop_gradient(h[:,None,None]) + 1e-5 
            elif self.bandwidth_selection_type == 'annealing':
                h = self.h_ph #20.
            elif self.bandwidth_selection_type == 'const':
                h = 1. 
            else:
                raise ValueError('Unrecognized bandwidth type: {}.'.format(self.bandwidth_selection_type))
            
            xixj = tf.reduce_mean(tf.reduce_mean(tf.exp(-d1/h), axis=-1), axis=-1) #/ (self._num_atoms * (self._num_atoms-0)) # (b.)
            xiyj = tf.reduce_mean(tf.reduce_mean(tf.exp(-d2/h), axis=-1), axis=-1) #(b,)
            yiyj = tf.reduce_mean(tf.reduce_mean(tf.exp(-d3/h), axis=-1), axis=-1)
            mmd = xixj + yiyj - 2*xiyj
            loss = tf.reduce_mean(mmd) 

            train_op = self.optimizer.minimize(loss)

        ## Monitor 
        debug_particles = self._replay_net_outputs.particles # (bs,na,n)
        p_std = tf.reduce_mean(tf.math.reduce_std(debug_particles, axis=-1))
        p_mean = tf.reduce_mean(tf.reduce_mean(debug_particles, axis=-1))
        p_min = tf.reduce_mean(tf.reduce_min(debug_particles, axis=-1))
        p_max = tf.reduce_mean(tf.reduce_max(debug_particles, axis=-1))
        debug_var = [p_min, p_max, p_mean, p_std, loss, tf.constant(h)] 
        return loss, debug_var, train_op #train_op

    def _train_step(self):
        """Runs a single training step.

        Runs a training op if both:
        (1) A minimum number of frames have been added to the replay buffer.
        (2) `training_steps` is a multiple of `update_period`.

        Also, syncs weights from online to target network if training steps is a
        multiple of target update period.
        """
        # Run a train op at the rate of self.update_period if enough training steps
        # have been run. This matches the Nature DQN behaviour.
        if self._replay.memory.add_count > self.min_replay_history:
            if self.training_steps % self.update_period == 0:
                # for _ in range(10):
                #     self._sess.run(self._train_op[-1]) 
                # loss,debug_v, _ = self._sess.run([self._train_op[0], self._train_op[1], self._train_op[2]])
                h = self.bandwidth_annealing_fn(self.training_steps)
                loss,debug_v, _ = self._sess.run(self._train_op, feed_dict={self.h_ph: h})

                if self._debug:
                    self.statistics_collection.append(debug_v) 
                if (self.summary_writer is not None and
                        self.training_steps > 0 and
                        self.training_steps % self.summary_writing_frequency == 0):
                    summary = self._sess.run(self._merged_summaries)
                    self.summary_writer.add_summary(summary, self.training_steps) 
                
            if self.training_steps % self.target_update_period == 0:
                self._sess.run(self._sync_qt_ops)

        self.training_steps += 1

    def begin_episode(self, observation):
        return super(MMDAgent, self).begin_episode(observation)
    
    def step(self, reward, observation):
        """Records the most recent transition and returns the agent's next action.

        We store the observation of the last time step since we want to store it
        with the reward.

        Args:
            reward: float, the reward received from the agent's most recent action. r_t
            observation: numpy array, the most recent observation. s_tp1 

        Returns:
            int, the selected action.
        """
        self._last_observation = self._observation
        self._record_observation(observation)
        if not self.eval_mode:
            self._store_transition(self._last_observation, self.action, reward, False) # @ s_t, a_t, r_t
            self._train_step()

        self.action = self._select_action() # s_tp1 
        return self.action
    
    def end_episode(self, reward):
        if not self.eval_mode:
            self._store_transition(self._observation, self.action, reward, True)

    def _select_action(self):
        """Select an action from the set of available actions.

        Chooses an action randomly with probability self._calculate_epsilon(), and
        otherwise acts greedily according to the current Q-value estimates.

        Returns:
            int, the selected action.
        """
        if self._policy == 'eps_greedy':
            if self.eval_mode:
                epsilon = self.epsilon_eval
            else:
                epsilon = self.epsilon_fn(
                    self.epsilon_decay_period,
                    self.training_steps,
                    self.min_replay_history,
                    self.epsilon_train)
            if random.random() <= epsilon:
                # Choose a random action with probability epsilon.
                return random.randint(0, self.num_actions - 1)
            else:
                # Choose the action with highest Q-value at the current state.
                return self._sess.run(self._q_argmax, {self.state_ph: self.state})
        else:
            if self.eval_mode:
                epsilon = self.epsilon_eval
                if random.random() <= epsilon:
                    return random.randint(0, self.num_actions - 1)
                else: 
                    if self._policy in ['ucb', 'ps']:
                        return self._sess.run(self._q_argmax, {self.state_ph: self.state})
                    else: # 'ps2', 'ps3'
                        return self._sess.run(self._q_argmax, {self.state_ph: self.state})
            else:
                beta = self.beta_fn(self.training_steps)
                return self._sess.run(self._q_argmax_explore, {self.state_ph: self.state, self._beta_ph: beta}) 

class ParticlePolicy(object):
    def __init__(self, delta=0.1, quantile_index=None):
        """
        Args:
            target_type: str, [mode/separate]
        """
        self.delta = delta 
        self.beta = delta #norm.ppf(1 - delta, loc=0, scale=1) Too big beta might explode the learning
        self.quantile_index = quantile_index 

    @staticmethod
    def compute_thompson_matrix(particles):
        """Compute Thompson probability matrix.
        Args:
            particles: (bs,na,n)
        Returns:
            logits: (bs,na)
        """
        shape = particles.shape.as_list()
        bs = shape[0]
        na = shape[1]
        n = shape[2]
        indices = tf.range(n)
        logits = [] 
        for i in range(na):
            q1 = particles[:,i,:]
            i_index = tf.constant(np.array([j for j in range(na) if j != i]))
            q2 = tf.gather(particles, i_index, axis=1)
            s = tf.cast(tf.greater_equal(q1[:,None,None,:], q2[:,:,:,None]), dtype=tf.float32)
            logits.append(tf.reduce_sum(tf.math.reduce_prod(tf.reduce_sum(s, axis=2), axis=1), axis=1))
        logits = tf.stack(logits, axis=1)
        return logits 
    @staticmethod 
    def sample_from_action_probability(action_values):
        """
        Args:
            action_values: (bs,na) 
        
        Returns:
            selected_action: (bs,), one of the actions with maximum value. 
        """
        action_prob = tf.cast(tf.equal(tf.reduce_max(action_values, axis=1)[:,None], action_values), tf.float32) 
        # selected_action = tf.random.categorical(logits=action_prob, num_samples=1)[:,0] # FLAG: logits=[0,1,1] -> 0.16 0.42 0.42 
        action_prob = action_prob / tf.reduce_sum(action_prob, axis=-1) 
        selected_action = tfp.distributions.Categorical(probs=action_prob).sample(1)[:,0]
        return selected_action #tf.squeeze(selected_action)

    def draw_action(self, particles, policy='mean', head_index=0, random_weights=np.array([1]), beta=None):
        """Compute selected action based on the approximate posterior particles. 
        Args:
            particles: (bs,na,n)
            policy: str, [eps_greedy/mean/ucb/ps/boot/ensemble]. [mean/optimistic/posterior] for target estimator.
            head_index: int (for boot policy)
            random_weights: 

        Returns: 
            selected_action: (bs,)
        """
        if policy in ['eps_greedy', 'mean']:
            q_values = tf.reduce_mean(particles, axis=2) 

            # selected_action = tf.argmax(q_values, axis=1)
            # return selected_action 
            return self.sample_from_action_probability(q_values) 

        elif policy in ['ucb', 'optimistic']:
            q_mean_values = tf.reduce_mean(particles, axis=2) #(bs,na)
            if self.quantile_index is None:
                q_std_values = tf.math.reduce_std(particles, axis=2) #(bs,na)
                if beta is None:
                    beta = self.beta 
                q_values = q_mean_values + beta * q_std_values 
            else:
                q_values = q_mean_values + particles[:,:, self.quantile_index]
            # selected_action = tf.argmax(q_values, axis=1)
            # return selected_action
            return self.sample_from_action_probability(q_values)
        elif policy in ['ucb_max', 'optimistic_max']:
            q_mean_values = tf.reduce_mean(particles, axis=2) #(bs,na)
            q_values = q_mean_values + tf.reduce_max(particles, axis=-1)
            return self.sample_from_action_probability(q_values)

        elif policy in ['ps', 'posterior']:
            # A head is sampled at each time step, as opposed to bootrapped policy where a head is sampled at each episode. 
            p_shape = particles.shape.as_list()
            logits = tf.ones( [p_shape[0] * p_shape[1], p_shape[2]], dtype=tf.float32) 
            indices = tf.reshape(tf.random.categorical(logits, num_samples=1), [p_shape[0], p_shape[1]]) # (bs,na)
            mask = tf.one_hot(indices, depth=p_shape[2]) #(bs,na,n) where the last dim is one-hot 
            q_values = tf.reduce_sum(tf.multiply( particles, mask ), axis=2) 
            return self.sample_from_action_probability(q_values)
        elif policy in ['ps2', 'posterior2']:
            # Equation (3) in B. O'Donoghue et al. "The Uncertainty Bellman Equation and Exploration". 
            p_shape = particles.shape.as_list()
            q_mean_values = tf.reduce_mean(particles, axis=2) #(bs,na)
            q_std_values = tf.math.reduce_std(particles, axis=2) #(bs,na)
            beta = tf.random.normal((p_shape[0],p_shape[1])) #(bs,na)
            q_values = q_mean_values + beta * q_std_values 
            # selected_action = tf.argmax(q_values, axis=1)
            # return selected_action
            return self.sample_from_action_probability(q_values)
        elif policy in ['ps3', 'posterior3']:
            # Random weight of the heads into a randomized Q-function
            p_shape = particles.shape.as_list()
            random_ensemble_weights = tf.random.normal((p_shape[-1],)) #(bs,na)
            random_ensemble_weights = random_ensemble_weights / tf.reduce_sum(random_ensemble_weights) 
            q_values = tf.reduce_mean(tf.multiply(particles, random_ensemble_weights[None, None, :]), axis=-1)
            # selected_action = tf.argmax(q_values, axis=1)
            # return selected_action
            return self.sample_from_action_probability(q_values)
        elif policy == 'boot':
            # Select a head uniformaly at random at the start of the episode and follow this choice for an entire episode. 
            # Q: How about in evaluation?
            q_values = particles[:,:,head_index] #(bs,na)
            # print('DEBUGGGG')
            # print(q_values)
            # selected_action = tf.argmax(q_values, axis=1)
            # return selected_action
            return self.sample_from_action_probability(q_values)

        elif policy == 'rem':
            # Inspired by https://arxiv.org/abs/1907.04543v3, randomly combine the heads into a randomized head. 
            q_values = tf.reduce_sum(tf.multiply( particles, random_weights[None, None,:] ), axis=-1)
            return self.sample_from_action_probability(q_values)

        elif policy == 'ensemble':
            # Choose action based on the majority vote across heads
            # Q: episode-based or step-based? Seems step-based is more natural. 
            argmax_ensemble = tf.math.argmax(particles, axis=1) #(bs,n)
            assert argmax_ensemble.shape.as_list()[0] == 1 
            argmax_ensemble = tf.squeeze(argmax_ensemble) #(n,)
            with tf.device('/cpu:0'): # tf.unique_with_counts is not supported in GPU. 
                y,idx,count = tf.unique_with_counts(argmax_ensemble) 
                max_count_idx = tf.math.argmax(count) 
                # print('DEBUGGG')
                # print(y)
                # print(max_count_idx)
                # print(argmax_ensemble)
                return tf.gather(y, max_count_idx)[None]
        else:
            raise ValueError('Unrecognized policy: {}'.format(policy))

    def compute_target(self, targets, estimator='mean', random_weights=np.array([1])):
        """
        Args:
            targets: (bs,na,n) 
            estimator: str, [mean/optimistic/posterior/head_wise]

        Returns:
            action_targets: (bs,n)
        """
        if estimator == 'mean':
            q_values = tf.reduce_mean(targets, axis=2) #(bs,na)
            action_prob = tf.cast(tf.equal(tf.reduce_max(q_values, axis=1)[:,None], q_values), tf.float32) #(bs,na)
            action_prob = tf.div(action_prob, tf.reduce_sum(action_prob, axis=1, keepdims=True))
            action_targets = tf.reduce_sum(tf.multiply(targets, action_prob[:,:,None]), axis=1)
            return action_targets 
        elif estimator == 'optimistic':
            q_mean_values = tf.reduce_mean(targets, axis=2) #(bs,na)
            if self.quantile_index is None:
                q_std_values = tf.math.reduce_std(targets, axis=2) #(bs,na)
                q_values = q_mean_values + self.beta * q_std_values 
            else:
                q_values = q_mean_values + targets[:,:, self.quantile_index]
            
            action_prob = tf.cast(tf.equal(tf.reduce_max(q_values, axis=1)[:,None], q_values), tf.float32) #(bs,na)
            action_prob = tf.div(action_prob, tf.reduce_sum(action_prob, axis=1, keepdims=True))
            action_targets = tf.reduce_sum(tf.multiply(targets, action_prob[:,:,None]), axis=1)
            return action_targets  
        elif estimator == 'optimistic_max':
            q_mean_values = tf.reduce_mean(targets, axis=2) #(bs,na)
            q_values = q_mean_values + tf.reduce_max(targets, axis=-1)
            
            action_prob = tf.cast(tf.equal(tf.reduce_max(q_values, axis=1)[:,None], q_values), tf.float32) #(bs,na)
            action_prob = tf.div(action_prob, tf.reduce_sum(action_prob, axis=1, keepdims=True))
            action_targets = tf.reduce_sum(tf.multiply(targets, action_prob[:,:,None]), axis=1)
            return action_targets  
        elif estimator == 'posterior':
            action_prob = self.compute_thompson_matrix(targets)
            action_prob = tf.div(action_prob, tf.reduce_sum(action_prob, axis=1, keepdims=True))
            action_targets = tf.reduce_sum(tf.multiply(targets, action_prob[:,:,None]), axis=1)
            return action_targets  
        elif estimator == 'head_wise':
            action_targets = tf.reduce_max(targets, axis=1)
            return action_targets
        elif estimator == 'posterior3':
            p_shape = targets.shape.as_list()
            random_ensemble_weights = tf.random.normal((p_shape[-1],)) #(bs,na)
            random_ensemble_weights = random_ensemble_weights / tf.reduce_sum(random_ensemble_weights) 
            q_values = tf.reduce_mean(tf.multiply(targets, random_ensemble_weights[None, None, :]), axis=-1)
            action_prob = tf.cast(tf.equal(tf.reduce_max(q_values, axis=1)[:,None], q_values), tf.float32) #(bs,na)
            action_prob = tf.div(action_prob, tf.reduce_sum(action_prob, axis=1, keepdims=True))
            action_targets = tf.reduce_sum(tf.multiply(targets, action_prob[:,:,None]), axis=1)
            return action_targets 
        else:
            raise ValueError('Unrecognized estimator: {}.'.format(estimator))
        