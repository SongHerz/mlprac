#!/usr/bin/env python3

# I use this script to play with some configurations.
# It is interesting that with 2 fully connected layers, the prediction result is bad.
# One layer can get 91% accuracy, but 2 will not exceed 58% with configurations I tried.

# Accuracy with differnt activation function at different steps, 3 times data.
# +--------------------------------------------------------------------------------------------------+
# | step |        None          |       sigmoid        |         tanh         |         relu         |
# +--------------------------------------------------------------------------------------------------+
# | 500  | 0.8989 0.8975 0.9027 | 0.8506 0.8020 0.7376 | 0.9074 0.9095 0.9087 | 0.9098 0.9066 0.9034 |
# | 1000 | 0.9083 0.9116 0.9059 | 0.9072 0.9057 0.9071 | 0.9249 0.9203 0.9205 | 0.9220 0.9219 0.9212 |
# | 1500 | 0.9117 0.9126 0.9101 | 0.9159 0.9179 0.9144 | 0.9279 0.9227 0.9260 | 0.9298 0.9311 0.9286 |
# | 2000 | 0.9117 0.9145 0.9151 | 0.9190 0.9224 0.9206 | 0.9304 0.9277 0.9278 | 0.9355 0.9343 0.9360 |
# +--------------------------------------------------------------------------------------------------+

# From the table above:
# Activation function can be ordered:
# relu > tanh > sigmoid > None

# Another reference:
# https://www.tensorflow.org/get_started/mnist/beginners

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


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


IN_SIZE = 28 * 28
INNER_SIZE = 28 * 4
OUT_SIZE = 10
xs = tf.placeholder(tf.float32, (None, IN_SIZE))
ys = tf.placeholder(tf.float32, (None, OUT_SIZE))
keep_prob = tf.placeholder(tf.float32)


def compute_accuracy(sess, prediction, v_xs, v_ys):
    global xs
    global ys
    global keep_prob
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1.0})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1.0})
    return result


def run(active_func):
    l1 = add_layer(xs, keep_prob, IN_SIZE, INNER_SIZE, activation_function=active_func)
    prediction = add_layer(l1, keep_prob, INNER_SIZE, OUT_SIZE, activation_function=tf.nn.softmax)

    loss = get_loss(ys, prediction)
    # train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
    # train_step = tf.train.AdamOptimizer(0.003).minimize(loss)
    # train_step = tf.train.AdadeltaOptimizer().minimize(loss)
    # train_step = tf.train.RMSPropOptimizer(0.8).minimize(loss)

    tf.summary.scalar(name='loss', tensor=loss)
    all_summ = tf.summary.merge_all()

    with tf.Session() as sess:
        train_writer = tf.summary.FileWriter('logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('logs/test', sess.graph)

        sess.run(tf.global_variables_initializer())

        # If minst data does not exists, it will be downloaded automatically
        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        for i in range(2001):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 0.5})
            if i % 50 == 0:
                print('Step:', i,
                      'Accuracy:', compute_accuracy(
                          sess, prediction,
                          mnist.test.images, mnist.test.labels))

                train_writer.add_summary(
                    sess.run(all_summ, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob: 1.0}), i)
                test_writer.add_summary(
                    sess.run(all_summ, feed_dict={xs: mnist.test.images, ys: mnist.test.labels, keep_prob: 1.0}), i)


if __name__ == '__main__':
    import argparse

    DESCRIPTION = 'Play with a nerual network'
    ap = argparse.ArgumentParser(description=DESCRIPTION)
    ap.add_argument('--activation', required=True, choices=['None', 'relu', 'sigmoid', 'tanh'],
                    help='Activation function for the inner layer')
    args = ap.parse_args()

    if args.activation == 'relu':
        active_func = tf.nn.relu
    elif args.activation == 'sigmoid':
        active_func = tf.nn.sigmoid
    elif args.activation == 'tanh':
        active_func = tf.nn.tanh
    else:
        assert args.activation == 'None'
        active_func = None

    run(active_func)
