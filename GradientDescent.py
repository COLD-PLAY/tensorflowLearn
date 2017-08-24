import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

### create tensorflow structure start ###
weight = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.zeros([1]))

y = weight*x_data + bias

loss = tf.reduce_mean(tf.square(y -  y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
### create tensorflow structure end ###

sess = tf.Session()
sess.run(init)

for step in range(201):
	sess.run(train)
	print(step, sess.run(weight), sess.run(bias))
	# if step % 20 == 0:
	# 	print(step, sess.run(weight), sess.run(bias))