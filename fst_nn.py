#!/usr/bin/env python3

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class Vis:
    def __init__(self, x_data, y_data):
        self.__fig = plt.figure()
        self.__ax = self.__fig.add_subplot(1, 1, 1)
        self.__ax.scatter(x_data, y_data)
        plt.ion()
        plt.show()

        self.__last_lines = []
        pass

    def update(self, x_data, y_predictions):
        if self.__last_lines:
            assert len(self.__last_lines) == 1
            self.__ax.lines.remove(self.__last_lines[0])

        self.__last_lines = self.__ax.plot(x_data, y_predictions, 'r-', lw=5)
        pass

    @staticmethod
    def pause_a_tick():
        plt.pause(0.1)
        pass


def add_layer(inputs, in_size, out_size, activation_funciton=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    # biases = tf.Variable(tf.zeros([1, out_size])) + 0.1
    # The following code should be more compact
    biases = tf.Variable(initial_value=tf.constant(0.1, dtype=tf.float32, shape=(1, out_size)))
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

# None means any sample can be given.
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# hidden layer
l1 = add_layer(xs, 1, 10, activation_funciton=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_funciton=None)

# tf.reduce_sum(..., axis=[1]) convert
# 2-d array from tf.square(...) to
# 1-d array
# The key is `axis=[1]`
#
# tf.reshpe can replace tf.reduce_sum here. (Not tested yet)
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction), axis=[1]))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()

vis = Vis(x_data, y_data)

with tf.Session() as sess:
    sess.run(init)
    for i in range(1000):
        sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
        if i % 50 == 0:
            print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))

            prediction_value = sess.run(prediction, feed_dict={xs: x_data})
            vis.update(x_data, prediction_value)
            vis.pause_a_tick()

    # Check some intermediate result
    # print(sess.run(tf.square(ys - prediction), feed_dict={xs: x_data, ys: y_data}))
    # print(sess.run(tf.reduce_sum(tf.square(ys - prediction), axis=[1]), feed_dict={xs: x_data, ys: y_data}))
