import sys 
import tensorflow as tf 
import numpy as np 
import gin.tf 
from dopamine.agents.dqn import dqn_agent 
from dopamine.agents.implicit_quantile.implicit_quantile_agent import ImplicitQuantileAgent
from mmd_agent import MMDAgent 
from quantile_agent import QuantileRegressionAgent
from create_runner import CreateRunner

import gin.tf 

AGENTS = ['mmd', 'quantile', 'iqn'] 

def load_gin_configs(gin_files, gin_bindings):
    gin.parse_config_files_and_bindings(gin_files,
                                      bindings=gin_bindings,
                                      skip_unknown=False)


@gin.configurable
def create_agent(sess, environment, agent_name=None, summary_writer=None, debug_mode=False):
    assert agent_name in AGENTS
    if not debug_mode:
        summary_writer = None 
    if agent_name == 'mmd':
        return MMDAgent(sess, num_actions=environment.action_space.n, summary_writer=summary_writer)
    elif agent_name == 'quantile':
        return QuantileRegressionAgent(sess, num_actions=environment.action_space.n, summary_writer=summary_writer)
    elif agent_name == 'iqn':
        return ImplicitQuantileAgent(sess, num_actions=environment.action_space.n, summary_writer=summary_writer)
    else:
        raise ValueError('Unrecognized agent: {}.'.format(agent_name)) 


@gin.configurable
def create_agent_runner(base_dir, agent_name):
    def create_agent_fn(sess, environment, summary_writer=None, debug_mode=False):
        return create_agent(sess, environment, agent_name, summary_writer, debug_mode) 
    return CreateRunner(base_dir, create_agent_fn) 
