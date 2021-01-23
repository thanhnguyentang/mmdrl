import sys, os 
from dopamine.utils import atari_lib, checkpointer
from extra_utils import iteration_statistics, logger

import numpy as np 
import time 
import tensorflow as tf 
import gin.tf 

@gin.configurable 
class CreateRunner(object):
    def __init__(self, 
            base_dir, 
            create_agent_fn,
            create_environment_fn=atari_lib.create_atari_environment,
            checkpoint_file_prefix='ckpt',
            logging_file_prefix='log',
            num_iterations=200, 
            training_steps=1000,
            evaluation_steps=200,
            max_steps_per_episode=10,
            reward_shaping_fn=None, #reward_shaping.identity_fn, 
            report_reward_shaping_fn=None, #reward_shaping.identity_fn, 
            reward_clip=False, 
            log_period = 1):
        self._num_iterations = num_iterations 
        self._training_steps = training_steps 
        self._evaluation_steps = evaluation_steps 
        self._max_steps_per_episode = max_steps_per_episode 
        self._logging_file_prefix = logging_file_prefix
        self._reward_clip = reward_clip
        self._reward_shaping_fn = reward_shaping_fn
        self._report_reward_shaping_fn = report_reward_shaping_fn

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True 
        self._sess = tf.Session(config=config)
        self._environment = create_environment_fn() 
        self._agent = create_agent_fn(self._sess, self._environment)  
        self._sess.run(tf.global_variables_initializer()) 
        self._base_dir = base_dir 
        self._summary_writer = tf.summary.FileWriter(self._base_dir)
        self._summary_writer.add_graph(graph=tf.get_default_graph())
        self._log_period = log_period 
        self._create_directories()
        self._initialize_checkpointer_and_maybe_resume(checkpoint_file_prefix)
    
    def _create_directories(self):
        self._checkpoint_dir = os.path.join(self._base_dir, 'checkpoints') 
        self._logger = logger.Logger(os.path.join(self._base_dir, 'logs'), self._logging_file_prefix) 

    def _initialize_checkpointer_and_maybe_resume(self, checkpoint_file_prefix):
        """Reloads the latest checkpoint if it exists.

        This method will first create a `Checkpointer` object and then call
        `checkpointer.get_latest_checkpoint_number` to determine if there is a valid
        checkpoint in self._checkpoint_dir, and what the largest file number is.
        If a valid checkpoint file is found, it will load the bundled data from this
        file and will pass it to the agent for it to reload its data.
        If the agent is able to successfully unbundle, this method will verify that
        the unbundled data contains the keys,'logs' and 'current_iteration'. It will
        then load the `Logger`'s data from the bundle, and will return the iteration
        number keyed by 'current_iteration' as one of the return values (along with
        the `Checkpointer` object).

        Args:
            checkpoint_file_prefix: str, the checkpoint file prefix.

        Returns:
            start_iteration: int, the iteration number to start the experiment from.
            experiment_checkpointer: `Checkpointer` object for the experiment.
        """
        self._checkpointer = checkpointer.Checkpointer(self._checkpoint_dir, checkpoint_file_prefix)
        self._start_iteration = 0
        # Check if checkpoint exists. Note that the existence of checkpoint 0 means
        # that we have finished iteration 0 (so we will start from iteration 1).
        latest_checkpoint_version = checkpointer.get_latest_checkpoint_number(self._checkpoint_dir)
        experiment_data = None 
        if latest_checkpoint_version >= 0:
            experiment_data = self._checkpointer.load_checkpoint(latest_checkpoint_version)
        if self._agent.unbundle(self._checkpoint_dir, latest_checkpoint_version, experiment_data):
            if experiment_data is not None:
                assert 'logs' in experiment_data
                assert 'current_iteration' in experiment_data
                self._logger.data = experiment_data['logs']
                self._start_iteration = experiment_data['current_iteration'] + 1
                tf.logging.info('Reloaded checkpoint and will start from iteration %d',self._start_iteration)

    def _run_one_step(self, action):
        observation, reward, is_terminal, _ = self._environment.step(action)
        return observation, reward, is_terminal 

    def _end_episode(self, reward):
        self._agent.end_episode(reward)

    def _run_one_episode(self):
        """Run one episode. 
        """
        step_number = 0
        total_reward = 0.

        action = self._initialize_episode()
        is_terminal = False

        # Keep interacting until we reach a terminal state.
        while True:
            observation, reward, is_terminal = self._run_one_step(action)

            total_reward += self._report_reward_shaping_fn(reward, is_terminal) if self._report_reward_shaping_fn is not None else reward 
            step_number += 1

            # print(reward)

            # Perform reward clipping.
            if self._reward_clip:
                reward = np.clip(reward, -1, 1)

            if self._reward_shaping_fn is not None:
                reward = self._reward_shaping_fn(reward, is_terminal)

            if (self._environment.game_over or step_number == self._max_steps_per_episode):
                # Stop the run loop once we reach the true end of episode.
                break
            elif is_terminal:
                # If we lose a life but the episode is not over, signal an artificial
                # end of episode to the agent.
                self._agent.end_episode(reward)
                action = self._agent.begin_episode(observation)
            else:
                action = self._agent.step(reward, observation) 

        self._end_episode(reward)

        return step_number, total_reward

    def _initialize_episode(self):
        initial_observation = self._environment.reset()
        return self._agent.begin_episode(initial_observation)

    def _run_one_phase(self, min_steps, statistics, run_mode_str):
        """Run `min_steps` steps (and count how many episodes for that number of steps). 
        It always run a least one episode. 
        Args:
            statistics: `IterationStatistics`.
        """
        step_count = 0
        num_episodes = 0
        sum_returns = 0.

        while step_count < min_steps:
            # start_time = time.time()
            episode_length, episode_return = self._run_one_episode()
            # end_time = time.time()
            statistics.append({
                '{}_episode_lengths'.format(run_mode_str): episode_length,
                '{}_episode_returns'.format(run_mode_str): episode_return
            })
            step_count += episode_length
            sum_returns += episode_return
            num_episodes += 1
            # We use sys.stdout.write instead of tf.logging so as to flush frequently
            # without generating a line break.
            sys.stdout.write('Steps executed: {} '.format(step_count) +
                       'Episode length: {} '.format(episode_length) +
                       'Return: {}\r'.format(episode_return))
                    #    'Time/Step: {}\r'.format((end_time - start_time) /episode_length))
        sys.stdout.flush()
        return step_count, sum_returns, num_episodes

    def _run_train_phase(self, statistics):
        """Run `self._training_steps` steps. 
        """
        self._agent.eval_mode = False
        start_time = time.time()
        number_steps, sum_returns, num_episodes = self._run_one_phase(self._training_steps, statistics, 'train')
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        statistics.append({'train_average_return': average_return})
        time_delta = time.time() - start_time
        tf.logging.info('Average undiscounted return per training episode: %.2f', average_return)
        tf.logging.info('Average training steps per second: %.2f', number_steps / time_delta)
        return num_episodes, average_return

    def _run_eval_phase(self, statistics):
        """Run evaluation phase.

        Args:
        statistics: `IterationStatistics` object which records the experimental
            results. Note - This object is modified by this method.

        Returns:
        num_episodes: int, The number of episodes run in this phase.
        average_reward: float, The average reward generated in this phase.
        """
        # Perform the evaluation phase -- no learning.
        self._agent.eval_mode = True
        _, sum_returns, num_episodes = self._run_one_phase(self._evaluation_steps, statistics, 'eval')
        average_return = sum_returns / num_episodes if num_episodes > 0 else 0.0
        tf.logging.info('Average undiscounted return per evaluation episode: %.2f',average_return)
        statistics.append({'eval_average_return': average_return})
        return num_episodes, average_return

    def _run_one_iteration(self, iteration):
        """Runs one iteration of agent/environment interaction.

        An iteration involves running several episodes until a certain number of
        steps are obtained. The interleaving of train/eval phases implemented here
        are to match the implementation of (Mnih et al., 2015).

        Args:
            iteration: int, current iteration number, used as a global_step for saving
            Tensorboard summaries.

        Returns:
            A dict containing summary statistics for this iteration.
        """
        
        # train_episode_length: length of each episode during training.
        # train_episode_returns: total return for each episode run during training. 
        # train_average_return: Average episodic return after running for `training_steps`.   
        # eval_average_return: Average episodic return after running for `evaluation_steps` for evaluation. 
        statistics = iteration_statistics.IterationStatistics()
        tf.logging.info('Starting iteration %d', iteration)
        num_episodes_train, average_reward_train = self._run_train_phase(statistics)
        num_episodes_eval, average_reward_eval = self._run_eval_phase(statistics)

        # debug 
        try:
            if self._agent._debug:
                self._monitor_statistics() 

                # Print avg training error 
                avg_loss = np.mean(np.array(self._agent.statistics_collection), axis=0) 
                tf.logging.info('DEBUG VAR: {}'.format(avg_loss))
                self._agent.statistics_collection = []  
        except: # Currently on IDN has debug flag. 
            pass 

        self._save_tensorboard_summaries(iteration, num_episodes_train,
                                     average_reward_train, num_episodes_eval,
                                     average_reward_eval, monitor_statistics=1)
        return statistics.data_lists

    def _monitor_statistics(self):
        # cdfs = self._agent._replay_net_outputs.cdf_values 
        # zs = self._agent._replay_net_outputs.z_values 
        # cdfs_, zs_ = self._sess.run([cdfs, zs])
        # z = zs_[:,0]
        # y = cdfs_[:,0,0] 
        # i1 = np.argmin(z) 
        # i2 = np.argmax(z) 
        # print('(z_min, y_min) = ({},{})'.format(z[i1], y[i1]))
        # print('(z_max, y_max) = ({},{})'.format(z[i2], y[i2]))
        z_debug = np.array([-100., -10.,-5.,  0, 5., 10., 100., ])
        try:
            y, a, ss = self._agent._debug_op(z_debug) 
            print(z_debug)
            print(y[:,0,0])
            print(y[:,0,1])
            print(a)
            print(np.mean(ss[0], axis=-1))
            print(np.mean(ss[1], axis=-1))
        except:
            pass 
    def _save_tensorboard_summaries(self, iteration,
                                    num_episodes_train,
                                    average_reward_train,
                                    num_episodes_eval,
                                    average_reward_eval, 
                                    monitor_statistics):
        """Save statistics as tensorboard summaries.

        Args:
        iteration: int, The current iteration number.
        num_episodes_train: int, number of training episodes run.
        average_reward_train: float, The average training reward.
        num_episodes_eval: int, number of evaluation episodes run.
        average_reward_eval: float, The average evaluation reward.
        """
        summary = tf.Summary(value=[
            tf.Summary.Value(tag='Train/NumEpisodes',
                            simple_value=num_episodes_train),
            tf.Summary.Value(tag='Train/AverageReturns',
                            simple_value=average_reward_train),
            tf.Summary.Value(tag='Eval/NumEpisodes',
                            simple_value=num_episodes_eval),
            tf.Summary.Value(tag='Eval/AverageReturns',
                            simple_value=average_reward_eval), 
            tf.Summary.Value(tag='Monitor/Statistics', 
                            simple_value=monitor_statistics)
        ])
        self._summary_writer.add_summary(summary, iteration)
    
    def _log_experiment(self, iteration, statistics):
        self._logger.append('iteration_%d'%(iteration), statistics) 
        if iteration % self._log_period == 0:
            self._logger.log_to_file(iteration)

    def _checkpoint_experiment(self, iteration):
        """Checkpoint experiment data.

        Args:
        iteration: int, iteration number for checkpointing.
        """
        experiment_data = self._agent.bundle_and_checkpoint(self._checkpoint_dir, iteration)
        if experiment_data:
            experiment_data['current_iteration'] = iteration
            experiment_data['logs'] = self._logger.data
            self._checkpointer.save_checkpoint(iteration, experiment_data)

    def run_experiment(self):
        """Runs a full experiment, spread over multiple iterations."""
        tf.logging.info('Beginning training...')
        if self._num_iterations <= self._start_iteration:
            tf.logging.warning('num_iterations (%d) < start_iteration(%d)',self._num_iterations, self._start_iteration)
            return

        for iteration in range(self._start_iteration, self._num_iterations):
            statistics = self._run_one_iteration(iteration)
            self._log_experiment(iteration, statistics)
            self._checkpoint_experiment(iteration) 
    
    # For visualizing the learnt agent
    def evaluate_policy(self, fig_dir):
        SAVE=True 
        VISUALIZE=False 
        PLOT_FREQ=1

        if not os.path.exists(fig_dir):
            os.makedirs(fig_dir)

        n_actions = self._environment.environment.action_space.n 
        print(n_actions)

        plt.ion()
        
        # BeamRider: [-7,7]
        # Qbert, Asterix: [-30,30]
        # Breakout: [-14,14]
        vmin = -30
        vmax = 30 
        n_bins = 17

        bins = np.linspace(vmin,vmax,n_bins)
        xtick_locs = np.linspace(vmin,vmax,5)

        ytick_locs = np.linspace(0, 1, 5) 
        xtick_labels = ['%.1f'%(l) for l in xtick_locs ]
        ytick_labels = ['%.1f'%(l) for l in ytick_locs ]

        cmap = plt.cm.get_cmap('jet')
        colors = cmap(np.arange(cmap.N))
        cinv = cmap.N / (1. * n_actions)

        fig = plt.figure(figsize=(2.9*2, 3), constrained_layout=True)
        fig.subplots_adjust(wspace=0.37)

        ax0 = fig.add_subplot(1,2,1)
        ax1 = fig.add_subplot(1,2,2)

        action = self._initialize_episode()
        game_over = False 
        total_reward = 0. 
        step_count = 0 
        while True:
            print('STEP: ', step_count)

            obs, r, is_terminal, infos = self._environment.step(action)
            screen_buffer = self._environment.environment.ale.getScreenRGB2()

            total_reward += r 
            step_count += 1         

            if self._environment.game_over: 
                break 
            elif is_terminal:
                self._agent.end_episode(r) 
                action = self._agent.begin_episode(obs)
            else:
                self._agent._last_observation = self._agent._observation
                self._agent._record_observation(obs)
                action, particles = self._agent._sess.run([self._agent._q_argmax, self._agent._net_outputs.particles],{self._agent.state_ph: self._agent.state})
                print(action)
                if VISUALIZE or (SAVE and step_count % PLOT_FREQ == 0): 
                    ax1.cla()
                    for i in range(n_actions):
                        data = particles[0,i,:]
                        label = self._environment.environment.unwrapped.get_action_meanings()[i]
                        color=colors[int( (i+0.1)*cinv  )]
                        
                        ax1.hist(data, bins=bins,density=False, label=label, weights=np.ones(len(data)) / len(data), \
                            color=color, edgecolor='black', linewidth=1.2, alpha=0.9)

                    ax1.legend(loc='center left', bbox_to_anchor=(1,0.5), fontsize=5 )
                    ax1.set_xlabel('Return') 
                    ax1.set_ylabel('Probability')

                    ax1.set_yticks(ytick_locs)
                    ax1.set_yticklabels(ytick_labels)
                    ax1.set_ylim([0,1])

                    ax1.set_xticks(xtick_locs)
                    ax1.set_xticklabels(xtick_labels)
                    
                    ax0.cla()
                    ax0.imshow(screen_buffer,aspect='auto') 
                    ax0.axis('off')

                    if VISUALIZE:
                        fig.canvas.draw()
                        fig.canvas.flush_events()
                    else:
                        filename = os.path.join(fig_dir,'screen_it_%.5d.png'%(step_count))
                        fig.savefig(filename, dpi=500, bbox_inches='tight') 
                        print(filename)
        self._end_episode(r) 
        print('Total rewards: ', total_reward)