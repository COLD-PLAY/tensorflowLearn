import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# define the function to add layer to NN and return the outputs of this layer
def add_layer(inputs, in_size, out_size, n_layer=None, activation_function=None):
    layer_name = 'layer%s' % n_layer
    
    with tf.name_scope('layer'):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.random_normal([in_size, out_size]))
            tf.summary.histogram(layer_name + '/weights', W)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
            tf.summary.histogram(layer_name + '/biases', b)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, W) + b
            
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        tf.summary.histogram(layer_name + '/outputs', outputs)
        return outputs