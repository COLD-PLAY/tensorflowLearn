import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1.})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1.})

    return result

def weight_variable(shape):
    # Weights = tf.Variable(tf.random_normal(shape))
    # return Weights
    initial = tf.

def bias_variable(shape):
    bias = tf.Variable(tf.random_normal(tf.float32, shape) + 0.1)
    return bias

def conv2d(x, W):
    pass

def max_pool_2x2(x):
    pass

# define