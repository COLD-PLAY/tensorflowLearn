import tensorflow as tf
import numpy as np

from layer import add_layer

from tensorflow.examples.tutorials.mnist import input_data

# training with GPU
# tf.device('/gpu:1')

# this is our mnist_data, if you havn't MNIST_data, it'll download the data, else it'll load data.
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28 * 28
ys = tf.placeholder(tf.float32, [None, 10]) # 10 numbers

# add output layer
prediction = add_layer(xs, 784, 10, 1, activation_function=tf.nn.softmax)

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
tf.summary.scalar('loss', cross_entropy)

# cross_entropy = -tf.reduce_sum(ys * tf.log(prediction))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

writer = tf.summary.FileWriter('logs/', sess.graph)
merged = tf.summary.merge_all()

for i in range(1001):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})

    if i % 50 == 0:
        result = sess.run(merged, feed_dict={xs: mnist.test.images, ys: mnist.test.labels})
        writer.add_summary(result, i)

        correct_prediction = tf.equal(tf.argmax(ys, 1), tf.argmax(prediction, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

        # print(i, sess.run(accuracy, feed_dict={xs: mnist.test.images, ys: mnist.test.labels}));
    
# result = sess.run(prediction, feed_dict={xs: mnist.test.images, ys: mnist.test.labels})

# with open('result.txt', 'a') as fr:
#     for line in result:
#         line_str = map(lambda x: str(x), line)
#         fr.write('\t'.join(line_str))
#         fr.write('\n')

sess.close()