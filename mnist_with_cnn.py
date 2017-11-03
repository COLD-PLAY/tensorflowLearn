import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

import os  
os.environ['CUDA_VISIBLE_DEVICES']='2'

sess = tf.Session()

class Trainer(object):
    # define placeholder for inputs to CNN
    xs = tf.placeholder(tf.float32, [None, 784]) # 28 X 28
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)
    prediction = None
    accuracy_total = []

    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    def __init__(self, prob):
        self.prob = prob
            
    def compute_accuracy(self, v_xs, v_ys):
        global prediction
        y_pre = sess.run(self.prediction, feed_dict={self.xs: v_xs, self.keep_prob: 1.})
        correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        result = sess.run(accuracy, feed_dict={self.xs: v_xs, self.ys: v_ys, self.keep_prob: 1.})

        return result

    def weight_variable(self, shape):
        # Weights = tf.Variable(tf.random_normal(shape))
        # return Weights
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        # bias = tf.Variable(tf.random_normal(tf.float32, shape) + 0.1)
        # return bias
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        # strides [1, x_movement, y_movement, 1]
        # padding 'SAME' or 'VALID'
        # MUST HAVE strides[0] = strides[3] = 1
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        
        # strides [1, x_movement, y_movement, 1]
        # padding 'SAME' or 'VALID'
        # MUST HAVE strides[0] = strides[3] = 1
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def train(self):
        # the last 1 meams the height of the image, here is 1 because it's gray image
        # the first -1 means didn't care how many samples has
        x_image = tf.reshape(self.xs, [-1, 28, 28, 1])

        # print(x_image.shape) # [n_samples, 28, 28, 1]

        ## covn1 layer ##
        W_conv1 = self.weight_variable([5, 5, 1, 32]) # patch 5x5, in size 1, out size 32
        b_conv1 = self.bias_variable([32])
        h_conv1 = tf.nn.relu(self.conv2d(x_image, W_conv1) + b_conv1) # output size 28x28x32
        h_pool1 = self.max_pool_2x2(h_conv1)                          # output size 14x14x32

        ## covn2 layer ##
        W_conv2 = self.weight_variable([5, 5, 32, 64]) # patch 5x5, in size 32, out size 64
        b_conv2 = self.bias_variable([64])
        h_conv2 = tf.nn.relu(self.conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
        h_pool2 = self.max_pool_2x2(h_conv2)                          # output size 7x7x64

        ## func1 layer ##
        W_fc1 = self.weight_variable([7*7*64, 1024])
        b_fc1 = self.bias_variable([1024])
        # [n_samples, 7, 7, 64] ->> [n_samples, 7*7*64]
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        ## func2 layer ##
        W_fc2 = self.weight_variable([1024, 10])
        b_fc2 = self.bias_variable([10])
        # output the prediction, get the prediction
        self.prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

        cross_entropy = -tf.reduce_mean(self.ys * tf.log(self.prediction))
        # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

        sess.run(tf.global_variables_initializer())

        print(self.prob, '...')
        this_accuracy = []
        for _ in range(301):
            batch_xs, batch_ys = self.mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={self.xs: batch_xs, self.ys: batch_ys, self.keep_prob: self.prob})
            
            if _ % 10 == 0:
                accuracy = self.compute_accuracy(self.mnist.test.images[:100], self.mnist.test.labels[:100])
                print(_, accuracy)
                this_accuracy.append(accuracy)
        print(this_accuracy)

        self.accuracy_total.append(this_accuracy)

    def save(self, filename='2333.pkl'):
        import pickle

        with open(filename, 'wb') as fr:
            pickle.dump(self.accuracy_total, fr)

    def restore(filename='2333.pkl'):
        import pickle
        self.accuracy_total = pickle.load(open(filename, 'rb'))

    def plot(accuracy_total):
        import matplotlib.pyplot as plt

    def main():
        accuracy_total = restore(filename='2333.pkl')
        print(accuracy_total)
    


def main():
    # for prob in np.linspace(0.4, 1.0, 7):
    #     trainer = Trainer(prob)
    #     trainer.train()
    #     trainer.save(filename=str(prob) + '_2333.pkl')
    
    import pickle
    import matplotlib.pyplot as plt

    accuracy = pickle.load(open('accuracy_with_kind_of_keep_prob.pkl', 'rb'))

    for i in np.linspace(0.1, 1.0, 10):
        plt.plot(np.linspace(1, 301, 31), accuracy[str(i)], label=str(i))

    plt.legend(loc='best')
    plt.xlabel('data')
    plt.ylabel('accuracy')
    plt.show()

if __name__ == '__main__':
    main()