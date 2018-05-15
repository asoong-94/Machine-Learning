import gym
from gym.wrappers import Monitor
import itertools
import numpy as np
import os
import random
import sys
import tensorflow as tf

from collections import deque, namedtuple

env = gym.envs.make("Breakout-v0")

"""
0 = no op 
1 = fire
2 = left
3 = right
"""
VALID_ACTIONS = [0, 1, 2, 3]

class stateProcessor():
	"""
	process a raw atari image. 
	convert to grey scale. 
	resize before feeding to network 
	"""
	def __init__(self):
		# build the tf graph
		with tf.variable_scope("state_processor"):
			self.input_state = tf.placeholder(shape=[210,160,3], dtype=tf.uint8)
			self.output = tf.image.rgb_to_grayscale(self.input_state)
			self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
			self.output = tf.image.resize_images(
				self.output, [84, 84], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
			self.output = tf.squeeze(self.output)

	def process(self, sess, state):
		"""
		Arguments:
			sess: a tf session object
			state: a [210, 160, 3] atari rgb state
		Returns:
			processed [84, 84, 1] grayscale state 
		"""
		return sess.run(self.output, feed_dict={self.input_state : state})


class Estimator():
	"""Q-Value Estimator Network"""

	def __init__(self, scope="estimator", summaries_dir=None):
		self.scope = scope
		self.summary_writer = None
		with tf.variable_scope(scope):
			# Build the graph
			self._build_model()
			if summaries_dir:
				summary_dir = os.path.join(summaries_dir, "summaries_{}".format(scope))
			if not os.path.exists(summary_dir):
				os.makedirs(summary_dir)
			self.summary_writer = tf.summary.FileWriter(summary_dir)


	def _build_model(self):
		"""
		build tf graph 
		"""

		# placeholders 
		# input is 4 rgb frames 
		self.X_pl = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
		# target TD values
		self.Y_pl = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
		# Integer ID of action selected
		self.actions_pl = tf.placeholder(shape=[None], dtype=tf.int32, name="actions")

		X = tf.to_float(self.X_pl) / 255.0
		batch_size = tf.shape(self.X_pl)[0]

		# 3 convolutional layers 
		# (input, num_output, kernel_size, stride, activation)
		conv1 = tf.contrib.layers.conv2d(X, 32, 8, 3, activation_fn=tf.nn.relu)
		conv2 = tf.contrib.layers.conv2d(conv1, 64, 4, 2, activation_fn=tf.nn.relu)
		conv3 = tf.contrib.layers.conv2d(conv2, 64, 3, 1, activation_fn=tf.nn.relu)

		# flatten into a fully connected layer 
		flattened = tf.contrib.layers.flatten(conv3)
		fc1 = tf.contrib.layers.fully_connected(flattened, 512)
		# predict a action 
		self.predictions = tf.contrib.layers.fully_connected(fc1, len(VALID_ACTIONS))

		# get predictions for chosen action 
		gather_indices = tf.range(batch_size) * tf.shape(self.predictions)[1] + self.actions_pl
		self.action_predictions = tf.gather(tf.reshape(self.predictions, [-1]), gather_indices)









