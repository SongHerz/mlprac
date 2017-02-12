#!/usr/bin/env python3

import tensorflow as tf
import numpy as np


def add_layer(inputs, in_size, out_size, activation_funciton=None):
    with tf.name_scope('layer'):
        Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='Weights')
        # biases = tf.Variable(initial_value=tf.zeros([1, out_size])) + 0.1
        # I think the following biases code is more compact
        biases = tf.Variable(initial_value=tf.constant(value=0.1, dtype=tf.float32, shape=[1, out_size]), name='biases')
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs, Weights) + biases

        if activation_funciton is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_funciton(Wx_plus_b)

        return outputs


x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
# FIXME: May control random seed to make data stable
# in order to compare a lot of parameters
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

with tf.name_scope('inputs'):
    # None means any sample can be given.
    xs = tf.placeholder(tf.float32, [None, 1], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

# hidden layer
l1 = add_layer(xs, 1, 10, activation_funciton=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_funciton=None)

with tf.name_scope('train'):
    with tf.name_scope('loss'):
        # tf.reduce_sum(..., axis=[1]) convert
        # 2-d array from tf.square(...) to
        # 1-d array
        # The key is `axis=[1]`
        #
        # tf.reshpe can replace tf.reduce_sum here. (Not tested yet)
        loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), axis=[1]))

    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    writer = tf.summary.FileWriter('logs', sess.graph)

    # sess.run(init)
    # for i in range(1000):
    #     sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
