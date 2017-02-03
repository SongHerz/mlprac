#!/usr/bin/env python3

import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

output = tf.mul(input1, input2)

with tf.Session() as sess:
    # No variable, no variable initialization required
    result = sess.run(output, feed_dict={input1: [7.], input2: [2.]})
    print(result)
