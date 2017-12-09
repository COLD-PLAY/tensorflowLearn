import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# create data
# x_data = np.random.rand(100).astype(np.float32)
# y_data = x_data*0.1 + 0.3
x_data = [1, 2, 3, 4, 5, 6]
y_data = [37.9, 39.8, 40.4, 42.7, 44.1, 47.1]

# plt.plot(x_data, y_data, 'bo')
# plt.show()

### create tensorflow structure start ###
# weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
weight = tf.Variable(1.)
bias = tf.Variable(tf.zeros([1]))

y = weight*x_data + bias

loss = tf.reduce_mean(tf.square(y -  y_data))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
### create tensorflow structure end ###

sess = tf.Session()
sess.run(init)

for step in range(10001):
	sess.run(train)
	print(step, sess.run(weight), sess.run(bias), sess.run(loss))
	# if step % 20 == 0:
	# 	print(step, sess.run(weight), sess.run(bias))