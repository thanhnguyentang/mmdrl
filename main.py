import sys,os 
import numpy as np 
import tensorflow as tf 
import gin.tf 
import time 

import experiment_prototype 

from absl import app 
from absl import flags 
import argparse 

flags.DEFINE_string('env', 'cartpole', 'Env name') 
flags.DEFINE_string('base_dir', './results', 'Base dir') 
flags.DEFINE_string('agent_id', 'mmd', 'Agent id')
flags.DEFINE_string('agent_name', None, 'Agent name')

flags.DEFINE_integer('run_id', 0, 'run id')
flags.DEFINE_multi_string('gin_files', './configs/mmd_atari.gin', 'List of paths to gin configuration files')
flags.DEFINE_multi_string('gin_bindings', [], 'Gin bindings to override the values in the config files')
flags.DEFINE_bool('debug', 0, 'Debug')
FLAGS = flags.FLAGS 


def main(unused_argv):
    agent_name = FLAGS.agent_id if FLAGS.agent_name is None else FLAGS.agent_name 

    if FLAGS.debug:
        base_dir = os.path.join('./results/tmp',str(time.time())) 
    else:
        base_dir = os.path.join(FLAGS.base_dir, FLAGS.env, agent_name, 'run_%d'%(FLAGS.run_id))  

    tf.logging.set_verbosity(tf.logging.INFO)
    experiment_prototype.load_gin_configs(FLAGS.gin_files, FLAGS.gin_bindings)
    runner = experiment_prototype.create_agent_runner(base_dir, FLAGS.agent_id)
    runner.run_experiment()

if __name__ == '__main__':
    app.run(main)