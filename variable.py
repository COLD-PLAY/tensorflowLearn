import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

state = tf.Variable(0, name = 'counter')
one = tf.constant(1)

new_value = tf.add(state, one)
update = tf.assign(state, new_value)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)

	print(sess.run(state))

	for _ in range(3):
		sess.run(update)
		print(sess.run(state))

input1 = tf.constant(3.)
input2 = tf.constant(2.)
input3 = tf.constant(5.)

intermed = tf.add(input3, input2)
mul = tf.multiply(intermed, input1)

with tf.Session() as sess:
	# fetch 可以传入一个包含多个参数的list 来对多个op 节点运算
	result = sess.run([mul, intermed])
	print(result)

input1 = tf.placeholder(tf.types.float32)
input2 = tf.placeholder(tf.types.float32)

