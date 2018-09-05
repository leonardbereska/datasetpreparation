"""
Ground truth keypoint (gt kp) regression and evaluation
require:
- gt kp txt file (train+test)
- pred kp txt file (train+test)
"""
import os
from ops_general import read_table
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from ops_general import show_kp, read_kp_data
rng = np.random

path_to_harddrive = '/Volumes/Uni/'
path_to_root = '/Users/leonardbereska/myroot/'
# path = path_to_root + 'test_regression/'
# path_to_data = path_to_root + 'birds/birds_kp/'

path = path_to_harddrive + 'bbcpose/'
path_to_data = path
assert os.path.exists(path)


# name = 'yuting_birds'
# dim_x = 10  # n keypoints predicted yuting
# x_test = (np.load(path + 'yuting_predictions_test.npy') * 80 + 10) * 6 + 60
# x_train = (np.load(path + 'yuting_predictions_train.npy') * 80 + 10) * 6 + 60
# test loss: 32.15

# name = 'leo_birds'
# dim_x = 15  # n keypoints predicted leo
# x_test = (np.load(path + 'leo_test_predicted_kp.npy') + 1) * 200 + 160
# x_train = (np.load(path + 'leo_train_predicted_kp.npy') + 1) * 200 + 160
# test loss: 27.58

# name = 'leo_birds_new'
# dim_x = 15  # n keypoints predicted leo
# x_test = (np.load(path + 'leo_test_predicted_kp_new.npy') + 1) * 300 + 60
# x_train = (np.load(path + 'leo_train_predicted_kp_new.npy') + 1) * 300 + 60
# # test loss: 23.51

# name = 'leo_par'
# dim_x = 8  # n keypoints predicted leo
# x_test = (np.load(path + 'leo_test_predicted_kp_par.npy') + 1) * 200 + 160
# x_train = (np.load(path + 'leo_train_predicted_kp_par.npy') + 1) * 200 + 160
# test loss: 30.08


dim_x = 7
dim_y = 7  # n keypoints ground truth
d = 2  # x, y
b = 12
lr = 0.0002
n_epochs = 10000


# make artificial data
# n = 10000  # sample size
# x_train = rng.rand(n, dim_x, d)
# A = rng.rand(dim_y, dim_x)
# y_train = np.einsum('jk,ikl->ijl', A, x_train)

# y = W * x (y: ground truth, x: predictions)
# read in txt files to tables
name = 'testy'
y_train, n_train = read_kp_data(path + 'y_train.txt', n_kp=dim_y, d=d)
y_test, n_test = read_kp_data(path + 'y_train.txt', n_kp=dim_y, d=d)
x_train, _ = read_kp_data(path + 'y_train.txt', n_kp=dim_x, d=d)
x_test, _ = read_kp_data(path + 'y_train.txt', n_kp=dim_x, d=d)
x_train = x_train[:, :, 0:2]
x_test = x_test[:, :, 0:2]
# Train regressor:
# define linear layer (regressor)
X = tf.placeholder(dtype=tf.float64, shape=(b, dim_x, d))
Y = tf.placeholder(dtype=tf.float64, shape=(b, dim_y, d+1))  # +1 for kp visibility
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
    assert gt.shape == (n_b, dim_y, d+1)
    x = tf.gather(gt, 0, axis=2)
    y = tf.gather(gt, 1, axis=2)
    mask = tf.gather(gt, 2, axis=2)
    x_ = tf.gather(pred, 0, axis=2)
    y_ = tf.gather(pred, 1, axis=2)
    result = tf.sqrt(tf.pow(x - x_, 2) + tf.pow(y - y_, 2))
    result = mask * result  # mask out non-visible kp
    result = tf.reduce_mean(result)
    return result

# todo loss for human


loss = loss(Y, Y_, b)

# define optimizer
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
init = tf.global_variables_initializer()
saver = tf.train.Saver()


def train(from_scratch=True):
    with tf.Session() as sess:
        if from_scratch:
            sess.run(init)  # run the initializer
        else:
            saver.restore(sess, "../saved/" + name + ".ckpt")

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
        saver.save(sess, "../saved/" + name + ".ckpt")


def test():
    with tf.Session() as sess:
        saver.restore(sess, "../saved/" + name + ".ckpt")

        print("Testing... (Mean square loss comparison)")
        testing_loss = 0
        length = int(len(x_test)/b)-1
        for i in range(length):  # go through test set in batch-size piecewise fashion, then average
            # x_s, y_s = sample(x_train, y_train, b)
            # n = x_test.shape[0]
            select = np.arange(i*b, (i+1)*b)  # same batch size as before
            x_s = np.take(x_test, select, axis=0)
            y_s = np.take(y_test, select, axis=0)
            testing_loss += sess.run(loss, feed_dict={X: x_s, Y: y_s})
        testing_loss /= length
        print("Testing loss=", testing_loss)
        # print("Absolute mean square loss difference:", abs(training_loss - testing_loss))


def visualize():
    with tf.Session() as sess:
        saver.restore(sess, "../saved/" + name + ".ckpt")

        def show_regress(idx, original=False):
            plt.figure(idx)

            def regress(kp, regress):
                if regress:
                    out = kp
                else:
                    kp = kp.astype(np.float64)
                    kp = tf.expand_dims(kp, axis=1)
                    out = tf.matmul(W, kp).eval()
                return out

            kp_x = regress(x_test[idx][:, 0], original)
            kp_y = regress(x_test[idx][:, 1], original)
            # kp = kp_y, kp_x
            kp = kp_x, kp_y


            kp_x = y_test[idx][:, 0]
            kp_y = y_test[idx][:, 1]
            # kp_gt = kp_y, kp_x
            kp_gt = kp_x, kp_y

            path_img = read_table(path_to_data + 'test_img.txt', type_float=False)[idx][1]
            img = cv2.imread(path_to_data + path_img)
            show_kp(img, kp)
            show_kp(img, kp_gt)

        for j in range(10):
            # show_original(j)
            show_regress(j, original=True)
        block=True

# train(from_scratch=True)
# test()
visualize()