from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)

x = tf.placeholder('float', [None, 784])

# weight and biases
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
# y = tf.matmul(x, W) + b
y_ = tf.placeholder('float', [None, 10])

# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
tf.summary.scalar('cross_entropy', cross_entropy)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

merged = tf.summary.merge_all()
init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	
	writer = tf.summary.FileWriter('logs/', sess.graph)
	for step in range(1000):
		# print(step)
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

		result = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys})
		writer.add_summary(result, step)

	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
	tf.summary.scalar('accuracy', accuracy)	

	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))