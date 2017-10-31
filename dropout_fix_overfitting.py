import tensorflow as tf
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros(out_size))

    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    outputs = Wx_plus_b
    if activation_function is not None:
        outputs = activation_function(Wx_plus_b)
    
    return outputs

# load data
digits = datasets.load_digits()
X, y = digits.data, digits.target

y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

xs = tf.placeholder(tf.float32, [None, 64])
ys = tf.placeholder(tf.float32, [None, 10])

hidden_layer = add_layer(xs, 64, 10, 'Hidden Layer', activation_function=tf.nn.relu)
output_layer = add_layer(hidden_layer, 10, 10, 'Output Layer', activation_function=None)

loss = tf.reduce_mean(tf.square(output_layer - ys))
train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

############ tensorboard scalar
tf.summary.scalar('loss', loss)
writer = tf.summary.FileWriter('logs/dropout/', sess.graph)
merged = tf.summary.merge_all()

for _ in range(1001):
    sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
    if _ % 20 == 0:
        print(_, sess.run(loss, feed_dict={xs: X_train, ys: y_train}), end='\t')
        print(sess.run(loss, feed_dict={xs: X_test, ys: y_test}))

        result = sess.run(merged, feed_dict={xs: X_train, ys: y_train})
        writer.add_summary(result, _)