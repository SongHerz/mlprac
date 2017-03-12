#!/usr/bin/env python3
# vim: fileencoding=utf-8


# Reference: 莫烦 Tensorflow 17 dropout 解决 overfitting 问题
# 具体的实现和视频的有点小不同

# With 500 train steps:
# 1. Without dropout, prediction accuracy is around 20%
# 2. With dropout keep probability 0.5, prediction accuracy is over 97%

import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3)


def add_layer(inputs, keep_prob, in_size, out_size, activation_function=None):
    # Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    Weights = tf.Variable(tf.zeros([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)

    return outputs


def get_loss(ys, prediction):
    # loss
    # Call this "naive cross_entropy"
    # cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
    #                                                reduction_indices=[1]))
    # FIXME: It looks convergence speed is much slower with softmax_cross_entropy_with_logits
    # than the above manually write corssentropy
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys, logits=prediction))


def compute_accuracy(sess, prediction, v_xs, v_ys):
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1.0})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1.0})
    return result


IN_SIZE = 8 * 8
INNER_SIZE = 100
OUT_SIZE = 10
xs = tf.placeholder(tf.float32, (None, IN_SIZE))
ys = tf.placeholder(tf.float32, (None, OUT_SIZE))
keep_prob = tf.placeholder(tf.float32)

l1 = add_layer(xs, keep_prob, IN_SIZE, INNER_SIZE, activation_function=tf.nn.relu)
prediction = add_layer(l1, keep_prob, INNER_SIZE, OUT_SIZE, activation_function=tf.nn.softmax)

loss = get_loss(ys, prediction)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(loss)

tf.summary.scalar(name='loss', tensor=loss)
all_summ = tf.summary.merge_all()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)
    test_writer = tf.summary.FileWriter('logs/test', sess.graph)

    sess.run(tf.global_variables_initializer())
    for i in range(500):
        sess.run(train_step, feed_dict={xs: X_train, ys: y_train, keep_prob: 0.5})
        if i % 50 == 0:
            print('Step:', i,
                  'Accuracy:', compute_accuracy(
                      sess, prediction, X_test, y_test))

            train_writer.add_summary(sess.run(all_summ, feed_dict={xs: X_train, ys: y_train, keep_prob: 1.0}), i)
            test_writer.add_summary(sess.run(all_summ, feed_dict={xs: X_test, ys: y_test, keep_prob: 1.0}), i)
