import tensorflow as tf 
import numpy as np 
import cv2 
import collections 
import gin.tf 
import math 

ParticleDQNType = collections.namedtuple('ParticleDQN', ['particles', 'q_values'])

@gin.configurable
class ParticleDQNet(tf.keras.Model):
    def __init__(self, num_actions, num_atoms, name=None):
        super(ParticleDQNet, self).__init__(name=name)
        activation_fn = tf.keras.activations.relu
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.kernel_initializer = tf.keras.initializers.VarianceScaling(
            scale=1.0 / np.sqrt(3.0), mode='fan_in', distribution='uniform')
        # Defining layers.
        self.conv1 = tf.keras.layers.Conv2D(
            32, [8, 8], strides=4, padding='same', activation=activation_fn,
            kernel_initializer=self.kernel_initializer, name='Conv')
        self.conv2 = tf.keras.layers.Conv2D(
            64, [4, 4], strides=2, padding='same', activation=activation_fn,
            kernel_initializer=self.kernel_initializer, name='Conv')
        self.conv3 = tf.keras.layers.Conv2D(
            64, [3, 3], strides=1, padding='same', activation=activation_fn,
            kernel_initializer=self.kernel_initializer, name='Conv')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(
            512, activation=activation_fn,
            kernel_initializer=self.kernel_initializer, name='fully_connected')
        self.dense2 = tf.keras.layers.Dense(
            num_actions * num_atoms, kernel_initializer=self.kernel_initializer,
            name='fully_connected')
    def call(self, state):
        x = tf.cast(state, tf.float32)
        x = tf.div(x, 255.)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        x = self.dense1(x)
        x = self.dense2(x)
        particles = tf.reshape(x, [-1, self.num_actions, self.num_atoms]) #(b,a,n)
        q_values = tf.reduce_mean(particles, axis=2) # (b,a)
        return ParticleDQNType(particles, q_values)