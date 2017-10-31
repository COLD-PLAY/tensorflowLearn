import tensorflow as tf
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

# load data
digits = datasets.load_digits()
X, y = digits.data, digits.target

y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros())

    Wx_plus_b = tf.matmul(inputs, Weights) + biases

    if activation_function is not None:
        outputs = activation_function(Wx_plus_b)
    
    return outputs