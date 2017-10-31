import tensorflow as tf
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)

    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    outputs = tf.nn.dropout(Wx_plus_b, keep_prob)

    # outputs = Wx_plus_b
    if activation_function is not None:
        outputs = activation_function(outputs)
    
    return outputs

def train():
    # load data
    digits = datasets.load_digits()
    X, y = digits.data, digits.target

    y = LabelBinarizer().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    xs = tf.placeholder(tf.float32, [None, 64])
    ys = tf.placeholder(tf.float32, [None, 10])
    keep_prob = tf.placeholder(tf.float32)

    hidden_layer = add_layer(xs, 64, 50, 'Hidden Layer', activation_function=tf.nn.tanh)
    prediction = add_layer(hidden_layer, 50, 10, 'Output Layer', activation_function=tf.nn.softmax)

    # cross_entropy = tf.reduce_mean(tf.square(output_layer - ys))
    # train_step = tf.train.GradientDescentOptimizer(0.05).minimize(cross_entropy)

    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction), reduction_indices=[1]))
    cross_entropy = -tf.reduce_mean(ys * tf.log(prediction))
    train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    ############ tensorboard scalar
    tf.summary.scalar('loss', cross_entropy)
    train_writer = tf.summary.FileWriter('logs/dropout/train/', sess.graph)
    test_writer = tf.summary.FileWriter('logs/dropout/test/', sess.graph)
    merged = tf.summary.merge_all()

    train_cross = []
    test_cross = []
    for _ in range(1, 1001):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: .3})
        train_cross_entropy = sess.run(cross_entropy, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
        test_cross_entropy = sess.run(cross_entropy, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})

        train_cross.append(train_cross_entropy)
        test_cross.append(test_cross_entropy)
        
        if _ % 20 == 0:
            print(_, train_cross_entropy, end='\t')
            print(test_cross_entropy)

            train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train, keep_prob: 1})
            test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test, keep_prob: 1})
            train_writer.add_summary(train_result, _)
            test_writer.add_summary(test_result, _)

    return train_cross, test_cross

def save(filename=None):
    import pickle
    with open('dropout/train_cross.pickle', 'wb') as fr:
        pickle.dump(train_cross, fr)

    with open('dropout/test_cross.pickle', 'wb') as fr:
        pickle.dump(test_cross, fr)

def plot(train_cross, test_cross):
    import matplotlib.pyplot as plt

    # plt.subplot(121)
    # plt.title('train_cross_entropy')
    # plt.plot(range(1000), train_cross, 'r-')

    # plt.subplot(122)
    # plt.title('test_cross_entropy')
    # plt.plot(range(1000), train_cross, 'b-')

    plt.plot(range(1000), train_cross, 'r-', label='Train Cross Entropy')
    plt.plot(range(1000), test_cross, 'b-', label='Test Cross Entropy')

    plt.xlabel('Training Examples')
    plt.ylabel('Cross Entropy')
    plt.legend(loc='best')
    plt.show()

def restore(filename=None):
    import pickle
    train_cross = pickle.load(open('dropout/train_cross.pickle', 'rb'))
    test_cross = pickle.load(open('dropout/test_cross.pickle', 'rb'))

    plot(train_cross, test_cross)

def plot_compare(train_cross_1, test_cross_1, train_cross_2, test_cross_2):
    import matplotlib.pyplot as plt

    plt.subplot(121)
    plt.plot(range(1000), train_cross_1, 'r-', label='Train Cross Entropy')
    plt.plot(range(1000), test_cross_1, 'b-', label='Test Cross Entropy')

    plt.xlabel('Training Examples')
    plt.ylabel('Cross Entropy')
    plt.legend(loc='best')

    plt.subplot(122)
    plt.plot(range(1000), train_cross_2, 'r-', label='Train Cross Entropy')
    plt.plot(range(1000), test_cross_2, 'b-', label='Test Cross Entropy')

    plt.xlabel('Training Examples')
    plt.ylabel('Cross Entropy')
    plt.legend(loc='best')
    
    plt.show()

def restore_compare(filename=None):
    import pickle
    train_cross_1 = pickle.load(open('dropout/train_cross.pkl', 'rb'))
    test_cross_1 = pickle.load(open('dropout/test_cross.pkl', 'rb'))

    train_cross_2 = pickle.load(open('dropout/train_cross.pickle', 'rb'))
    test_cross_2 = pickle.load(open('dropout/test_cross.pickle', 'rb'))
    
    plot_compare(train_cross_1, test_cross_1, train_cross_2, test_cross_2)

def main():
    restore_compare()

if __name__ == '__main__':
    main()