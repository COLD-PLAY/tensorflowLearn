import tensorflow as tf
import numpy as np

# create data
# x_data = np.random.rand(100).astype(np.float32)
# y_data = x_data*0.1 + 0.3

x_data = tf.float32(np.random.rand(2, 100))
y_data = np.dot([0.100, 0.200], x_data) + 0.300

### create tensorflow structure start ###
bias = tf.Variable(tf.zeros([1]))
weight = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))

# y = weight*x_data + bias
y = tf.matmul(x_data, weight) + bias

loss = tf.reduce_mean(tf.square(y -  y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
### create tensorflow structure end ###

sess = tf.Session()
sess.run(init)

for step in range(201):
	sess.run(train)
	if step % 20 == 0:
		print(step, sess.run(weight), sess.run(bias))