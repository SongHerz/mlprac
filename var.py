#!/usr/bin/env python3

import tensorflow as tf


def counter1():
    """
    Return (update operation, counter variable)
    """
    counter = tf.Variable(0, name='counter_1')
    one = tf.constant(1)
    update = tf.assign_add(counter, one)
    return (update, counter)


def counter2():
    """
    Return (update operation, couner variable)
    """
    counter = tf.Variable(0, name='counter_2')
    one = tf.constant(1)
    add_1 = tf.add(counter, one)
    update = tf.assign(counter, add_1)
    return (update, counter)


def update(sess, updater, counter):
    print('# Before updating: counter', sess.run(counter))
    upd_r = sess.run(updater)
    print('# After updating', sess.run(counter))
    print('# Updater value', type(upd_r), upd_r)
    print()

update_1, counter_1 = counter1()
update_2, counter_2 = counter2()


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    print('## Updating counter 1')
    print(type(counter_1), counter_1.name)
    print(type(update_1), update_1.name)
    for _ in range(3):
        update(sess, update_1, counter_1)

    print('## Updating counter 2')
    print(type(counter_2), counter_2.name)
    print(type(update_2), update_2.name)
    for _ in range(3):
        update(sess, update_2, counter_2)
