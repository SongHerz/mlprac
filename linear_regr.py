#!/usr/bin/env python3

# Initial code from 莫烦 Tensorflow 5 例子2 (神经网络 教学教

import tensorflow as tf
import numpy as np

# Manually create dots of a line
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3


# Creation
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()

# Train
sess = tf.Session()
sess.run(init)  # Very important

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))

# Inspect types
print('Weights', type(Weights), Weights)
print('Biases', type(biases), biases)

w_ = sess.run(Weights)
b_ = sess.run(biases)
l_ = sess.run(loss)
print('run(Weights)', type(w_), w_, w_.shape, w_.dtype)
print('run(Biases)', type(b_), b_, b_.shape, b_.dtype)
print('run(loss)', type(l_), l_, l_.shape, l_.dtype)

print('type(init)', type(init))
print('type(train)', type(train))
print('type(loss)', type(loss))
