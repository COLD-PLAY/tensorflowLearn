import tensorflow as tf
import numpy as np

class GAN(object):
    log_path = 'tf_writer'

    def __init__(self, lenOfTimeSeq, lr_g=0.001, lr_d=0.001, useGPU=True):
        self.useGPU = useGPU
        self.seq_len = lenOfTimeSeq
        self.lr_g = lr_g
        self.lr_d = lr_d
        self.n_classes = 2

        # define common parameter, it's a scalar
        self.batch_size_t = tf.placeholder(tf.int32, shape=[])

        # g-network data flow
        self.g_inputTensor = tf.placeholder(tf.float32, shape=(None, self.seq_len))
        self.g_inputLabel = tf.placeholder(tf.float32, shape=(None, self.seq_len))
        g_logit = self.generator(self.g_inputTensor)
        tf.summary.histogram('g_net_input', self.g_inputTensor)

        # d-network data flow
        self.groundTruthTensor = tf.placeholder(tf.float32, shape=(None, self.seq_len, 1), name='gndTruth')
        self.sum_gnd_truth = tf.summary.tensor_summary('gnd_truth', self.groundTruthTensor)
        tf.summary.histogram('gnd_truth', self.groundTruthTensor)

        