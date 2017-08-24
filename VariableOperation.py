import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

state = tf.Variable(0, name='counter')
# print(state.name)
one = tf.constant(1)

new_value = tf.add(state, one)
# update = tf.assign(state, new_value)

print(state.name)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	for __ in range(3):
		# sess.run(update)
		sess.run(tf.assign(state, new_value))
		print(sess.run(state))

# input1 = tf.placeholder(tf.float32)
# input2 = tf.placeholder(tf.float32)
input1 = np.float32(7.0)
input2 = np.float32(2.0)

output = tf.multiply(input1, input2)

with tf.Session() as sess:
	# print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
	print(sess.run(output))