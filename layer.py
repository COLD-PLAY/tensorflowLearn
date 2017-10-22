import tensorflow as tf
import numpy as np

from matplotlib import pyplot as plt

def add_layer(inputs, in_size, out_size, activation_function=None):
    # W = tf.Variable(tf.random_normal([out_size, in_size]))
    # b = tf.Variable(tf.zeros([out_size, 1]) + 0.1)
    # Wx_plus_b = tf.matmul(W, inputs) + b

    W = tf.Variable(tf.random_normal([in_size, out_size]))
    b = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, W) + b

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs

# x_data = tf.linspace(-1., 1., 300)[:, np.newaxis]
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) + noise - 0.5

plt.plot(x_data, y_data, 'bo')
plt.show()

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# hiden layer
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# output layer
prediction = add_layer(l1, 10, 1, activation_function=None)

loss = tf.reduce_mean(tf.square(ys - prediction))
# optimizer = tf.train.GradientDescentOptimizer(0.5)
# train_step = optimizer.minimize(loss)

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(1001):
        if step % 50 == 0:
            print(step, sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
    
    