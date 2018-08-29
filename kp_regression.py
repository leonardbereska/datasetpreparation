"""
Ground truth keypoint (gt kp) regression and evaluation
require:
- gt kp txt file (train+test)
- pred kp txt file (train+test)
"""
import os
from helpers import read_table
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
rng = np.random

path_to_root = '/Users/leonardbereska/myroot/'
path = path_to_root + 'test_regression/'
assert os.path.exists(path)


# make artificial data
# n = 10000  # sample size
dim_x = 15  # n keypoints predicted
dim_y = 15  # n keypoints ground truth
d = 2  # x, y
b = 12

lr = 0.001
n_epochs = 5000

# x_train = rng.rand(n, dim_x, d)
# A = rng.rand(dim_y, dim_x)
# y_train = np.einsum('jk,ikl->ijl', A, x_train)


def read_kp_data(txt_path, n_kp):  # todo kp visible
    """
    read in text file with columns (img_idx, kp_idx, kp_x, kp_y, kp_visible)
    :param txt_path: where to read data
    :param n_kp: number of keypoints
    :return: matrix (n_samples, n_kp, d), d=2 for images
    """
    global d

    table = read_table(txt_path)
    assert float(int(len(table) / n_kp)) == (len(table) / n_kp), 'n_kp wrong or txt file corrupted'
    n_samples = int(len(table) / n_kp)
    kp = table[:, 2:4]
    kp = np.reshape(kp, newshape=(n_samples, n_kp, d))
    return kp, n_samples  # x, y, visible


# data = np.load(path + 'yuting_train_predictions.npy')


# y = W * x (y: ground truth, x: predictions)
# read in txt files to tables
y_train, n_train = read_kp_data(path + 'y_train.txt', n_kp=dim_y)
y_test, n_test = read_kp_data(path + 'y_test.txt', n_kp=dim_y)
x_train, _ = read_kp_data(path + 'y_train.txt', n_kp=dim_x)
x_test, _ = read_kp_data(path + 'y_test.txt', n_kp=dim_x)

# Train regressor:
# define linear layer (regressor)
X = tf.placeholder(dtype=tf.float64, shape=(b, dim_x, d))
Y = tf.placeholder(dtype=tf.float64, shape=(b, dim_y, d))
W = tf.Variable(rng.rand(dim_y, dim_x), name="weight")
Y_ = tf.einsum('jk,ikl->ijl', W, X)


def sample(x_train, y_train, b):
    n = x_train.shape[0]
    select = rng.randint(low=0, high=n-1, size=b)
    x_s = np.take(x_train, select, axis=0)
    y_s = np.take(y_train, select, axis=0)
    return x_s, y_s


# define L1 distance loss
def loss(gt, pred, n_b):
    assert gt.shape == (n_b, dim_y, d)
    x = tf.gather(gt, 0, axis=2)
    y = tf.gather(gt, 1, axis=2)
    x_ = tf.gather(pred, 0, axis=2)
    y_ = tf.gather(pred, 1, axis=2)
    return tf.reduce_mean(tf.sqrt(tf.pow(x-x_, 2)+tf.pow(y-y_, 2)))

# todo loss for human


loss = loss(Y, Y_, b)

# define optimizer
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)  # run the initializer

    # fit all training data
    for epoch in range(n_epochs):
        # for (x, y) in zip(x_train, y_train):
        x_s, y_s = sample(x_train, y_train, b)
        sess.run(optimizer, feed_dict={X: x_s, Y: y_s})

        # Display logs per epoch step
        if (epoch + 1) % int(n_epochs / 10) == 0:
            c = sess.run(loss, feed_dict={X: x_s, Y: y_s})
            print("Epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(c))

    print("Optimization finished!")
    x_s, y_s = sample(x_train, y_train, b)
    training_loss = sess.run(loss, feed_dict={X: x_s, Y: y_s})
    print("Training loss=", training_loss, "W=", np.round(sess.run(W), 2), '\n')
    # todo save W

    # todo visualize
    # # Graphic display
    # plt.plot(x_train, y_train, 'ro', label='Original data')
    # plt.plot(x_train, sess.run(W) * x_train, label='Fitted line')
    # plt.legend()
    # plt.show()

    print("Testing... (Mean square loss comparison)")
    testing_loss = 0
    length = int(len(x_test)/b)-1
    for i in range(length):  # go through test set in batch-size piecewise fashion, then average
        x_s, y_s = sample(x_train, y_train, b)
        n = x_test.shape[0]
        select = np.arange(i*b, (i+1)*b)  # same batch size as before
        x_s = np.take(x_test, select, axis=0)
        y_s = np.take(y_test, select, axis=0)
        testing_loss += sess.run(loss, feed_dict={X: x_s, Y: x_s})
    testing_loss /= length
    print("Testing loss=", testing_loss)
    print("Absolute mean square loss difference:", abs(training_loss - testing_loss))

    # plt.plot(test_X, test_Y, 'bo', label='Testing data')
    # plt.plot(x_train, sess.run(W) * x_train + sess.run(b), label='Fitted line')
    # plt.legend()
    # plt.show()
