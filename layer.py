import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# define the function to add layer to NN and return the outputs of this layer
def add_layer(inputs, in_size, out_size, n_layer=None, activation_function=None):
    layer_name = 'layer%d' % n_layer
    
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
        return outputs, W, b

# define the x_data and y_data and noise variables
# x_data = tf.linspace(-1., 1., 300)[:, np.newaxis]
x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) + noise - 0.5

with tf.name_scope('inputs'):
    # define the placeholder variables, make it easy to change the inputs_data
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# add the hiden layer to NN
l1, WW1, bb1 = add_layer(xs, 1, 10, n_layer=0, activation_function=tf.nn.relu)
# add the output layer to NN
prediction, WW2, bb2 = add_layer(l1, 10, 1, n_layer=1, activation_function=None)

with tf.name_scope('loss'):
    # define the loss function
    loss = tf.reduce_mean(tf.square(ys - prediction))
    tf.summary.scalar('loss', loss)

with tf.name_scope('train'):
    # define the train_step(A Gradient Descent Optimizer) to minimize the loss function
    # optimizer = tf.train.GradientDescentOptimizer(0.5)
    # train_step = optimizer.minimize(loss)
    train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# define the initializer for all variables
init = tf.global_variables_initializer()

sess = tf.Session()
merged = tf.summary.merge_all()
writer = tf.summary.FileWriter("logs/", sess.graph)

sess.run(init)

for step in range(1000):
    if step % 50 == 0:
        result = sess.run(merged, feed_dict={xs: x_data, ys: y_data})
        writer.add_summary(result, step)
        # print(step, 'loss: ', sess.run(loss, feed_dict={xs: x_data, ys: y_data}), 'prediction: ', sess.run(prediction, feed_dict={xs: x_data}))

    sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    
sess.close()





# # training station
# with tf.Session() as sess:
#     sess.run(init)

#     figure = plt.figure(1)
#     ax = figure.add_subplot(111)
#     ax.scatter(x_data, y_data)
#     plt.axis([-1., 1., -0.6, 0.6])
#     # plt.ion()
#     # plt.show()

#     for step in range(1001):
#         sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        
#         if step % 50 == 0:
#             print(step, sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
            
#             try:
#                 ax.lines.remove(lines[0])
#             except Exception as e:
#                 pass
#             # hide_layer_output = sess.run(tf.nn.relu(tf.matmul(x_data, WW1) + bb1))
#             # prediction_output = sess.run(tf.matmul(hide_layer_output, WW2) + bb2)
#             prediction_output = sess.run(prediction, feed_dict={xs: x_data})
#             lines = ax.plot(x_data, prediction_output, 'r-')

#             plt.pause(0.1)    
#     prediction_output = sess.run(prediction, feed_dict={xs: x_data})
#     lines = ax.plot(x_data, prediction_output, 'r-')
#     plt.show()