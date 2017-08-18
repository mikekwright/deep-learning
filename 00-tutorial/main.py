import tensorflow as tf
import numpy as np


def np_example():
    X = np.ones(5)
    Y = np.ones(5) * 3
    print(X)
    print(Y)
    print(np.dot(X, Y))


def tf_example():
    X = tf.constant(np.ones((1,5)))
    Y = tf.constant(np.ones((1,5)) * 3)
    print(X)
    print(Y)
    Z = tf.matmul(X, Y, transpose_b=True)
    print(Z)

    sess = tf.Session()
    W = sess.run(Z)
    print(W, W.shape)
    print(sess.run(X))

def tf_simple_graph():
    sess = tf.Session()

    x = tf.get_variable('my_tensor6', shape=(1, 5), initializer=tf.constant_initializer(1))
    y = x * 2
    print(x)
    print(y)

    target = tf.placeholder(tf.float32, (1,5), 'my_target')
    loss = tf.reduce_sum(tf.square(target - y))
    grads = tf.gradients(loss, x)
    print(grads)

    sess.run(tf.global_variables_initializer())
    feed_dict = {target: [[1, 2, 3, 4, 5]]}
    print(sess.run(loss, feed_dict))
    print(sess.run(grads, feed_dict))

    opt = tf.train.GradientDescentOptimizer(1e-1)
    print(opt)

    train_step = opt.minimize(loss)
    print(train_step)
    print(sess.run(train_step, feed_dict))
    print(sess.run([x, loss], feed_dict))  # After one run

    for i in range(10):
        sess.run(train_step, feed_dict)
        print(sess.run([x, loss], feed_dict))


if __name__ == '__main__':
    # np_example()
    # tf_example()
    tf_simple_graph()
