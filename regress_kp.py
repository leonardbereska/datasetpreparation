import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from ops_general import read_table, read_kp_data, show_kp_regr, show_kp_orig


class Regressor(object):
    def __init__(self, dataset, name, dim_y, dim_x, b, lr, n_epochs,
                 path_to_files, path_to_data, testpath, trainpath, process_kp, img_size, regularize=False):
        self.dataset = dataset
        self.name = name
        self.path_to_data = path_to_data
        self.path_to_files = path_to_files
        self.testpath = testpath
        self.trainpath = trainpath
        self.regularize = regularize

        # parameters
        self.dim_y = dim_y
        self.dim_x = dim_x
        self.b = b
        self.lr = lr
        self.n_epochs = n_epochs
        self.img_size = img_size
        self.d_pix = 6 / 128. * self.img_size

        # data
        self.y_train, n_train = read_kp_data(path_to_files + 'y_train.txt', n_kp=self.dim_y, d=2)
        self.y_test, n_test = read_kp_data(path_to_files + 'y_test.txt', n_kp=self.dim_y, d=2)  # self.y_train
        if self.dataset == 'birds':
            self.y_train = swap_yx(self.y_train)
            self.y_test = swap_yx(self.y_test)
        self.x_test = process_kp(np.load(path_to_files + self.testpath))
        self.x_train = process_kp(np.load(path_to_files + self.trainpath))

        # variables
        self.X = tf.placeholder(dtype=tf.float64, shape=(b, dim_x, 2))
        self.Y = tf.placeholder(dtype=tf.float64, shape=(b, dim_y, 2 + 1))  # +1 for kp visibility mask
        self.W = tf.Variable(np.random.rand(dim_y * 2, dim_x * 2), name="weight")
        if self.regularize:
            tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES, self.W)
        self.X_ = tf.reshape(self.X, [b, dim_x * 2])
        self.Y_ = tf.einsum('jk,ik->ij', self.W, self.X_)
        self.Y_ = tf.reshape(self.Y_, [b, dim_y, 2])
        if self.regularize:
            self.regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
        self.loss = self.loss_l1(self.Y, self.Y_)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def loss_l1(self, labels, pred):
        assert labels.shape == (self.b, self.dim_y, 2 + 1)
        dist = dist_l1(labels, pred)
        mask = tf.gather(labels, 2, axis=2)
        dist_batch_kp = mask * dist  # mask out non-visible kp
        dist_batch = tf.reduce_sum(dist_batch_kp, axis=1) / tf.reduce_sum(mask, axis=1)  # do not count invisible kp
        result = tf.reduce_mean(dist_batch)
        if self.regularize:
            reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            reg_term = tf.contrib.layers.apply_regularization(self.regularizer, reg_variables)
            result = result + reg_term
        return result

    def pck(self, x_s, y_s):
        x_s = x_s.astype(np.float64)
        x_s = tf.constant(x_s)
        x_s = tf.reshape(x_s, [self.b, self.dim_x * 2])

        y_ = tf.einsum('jk,ik->ij', self.W, x_s)
        y_ = tf.reshape(y_, [self.b, self.dim_y, 2])

        diff = dist_l1(y_s, y_) - self.d_pix
        is_correct = diff <= 0
        is_correct = tf.cast(is_correct, tf.float32)
        pck = tf.reduce_mean(is_correct).eval()
        pck_per_kp = tf.reduce_mean(is_correct, axis=0).eval()
        return pck, pck_per_kp

    def train(self, from_scratch=True):
        with tf.Session() as sess:
            if from_scratch:
                sess.run(self.init)
            else:
                self.saver.restore(sess, "../saved/" + self.name + ".ckpt")

            for epoch in range(self.n_epochs):

                x_s, y_s = random_sample(self.x_train, self.y_train, self.b)
                sess.run(self.optimizer, feed_dict={self.X: x_s, self.Y: y_s})

                if (epoch + 1) % int(self.n_epochs / 10) == 0:
                    c = sess.run(self.loss, feed_dict={self.X: x_s, self.Y: y_s})
                    print("epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(c))

            x_s, y_s = random_sample(self.x_train, self.y_train, self.b)
            training_loss = sess.run(self.loss, feed_dict={self.X: x_s, self.Y: y_s})
            # training_loss = sess.run(tf.reduce_mean(dist_l1(x_s, y_s)))
            # train_reg = sess.run
            print("train loss =", training_loss, '\n')
            # print("Training loss=", training_loss, "W=", np.round(sess.run(self.W), 2), '\n')
            self.saver.save(sess, "../saved/" + self.name + ".ckpt")

    def test(self, testset=True):
        pred, labels, testname = self.choose_set(testset)

        with tf.Session() as sess:
            self.saver.restore(sess, "../saved/" + self.name + ".ckpt")

            testing_loss = 0
            testing_pck = 0
            testing_pck_per_kp = 0

            if testset:
                length = int(len(pred) / self.b) - 1
            else:
                length = 100
            for i in range(length):  # go through test set in batch-size piecewise fashion, then average
                if testset:
                    select = np.arange(i * self.b, (i + 1) * self.b)  # same batch size as before
                    x_s = np.take(pred, select, axis=0)
                    y_s = np.take(labels, select, axis=0)
                else:
                    x_s, y_s = random_sample(self.x_train, self.y_train, self.b)

                batch_pck, batch_pck_per_kp = self.pck(x_s, y_s)

                batch_loss = sess.run(self.loss, feed_dict={self.X: x_s, self.Y: y_s})
                # batch_loss = sess.run(tf.reduce_mean(dist_l1(x_s, y_s)))

                testing_loss += batch_loss
                testing_pck += batch_pck
                testing_pck_per_kp += batch_pck_per_kp

            testing_loss /= length
            testing_pck /= length
            testing_pck_per_kp /= length

            print('{} {} = {}'.format(testname, 'loss', testing_loss))
            print('{} {} = {}'.format(testname, 'pck', testing_pck))
            print('{} {} = {}'.format(testname, 'pck_per_kp', testing_pck_per_kp))

    def choose_set(self, testset):
        if testset:
            pred = self.x_test
            labels = self.y_test
            testname = 'test'
        else:
            pred = self.x_train
            labels = self.y_train
            testname = 'train'
        return pred, labels, testname

    def get_pred(self, kp_orig):
        kp_orig = kp_orig.astype(np.float64)
        kp_orig = tf.reshape(kp_orig, [2 * self.dim_x, 1])
        kp_pred = tf.matmul(self.W, kp_orig)
        kp_pred = tf.reshape(kp_pred, [self.dim_y, 2]).eval()
        return kp_pred

    def show_regress(self, original, testset, verbose):
        pred, labels, testname = self.choose_set(testset)
        n = pred.shape[0]
        idx = np.random.randint(low=0, high=n - 1, size=1)[0]
        plt.figure(idx)

        path_img = read_table(self.path_to_files + '{}_img.txt'.format(testname), type_float=False)[idx][1]
        img = cv2.imread(self.path_to_data + path_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        kp_pred = pred[idx]
        kp_gt = labels[idx][:, 0:2]
        mask = labels[idx][:, 2]

        def to_tuple(x):
            x0 = x[:, 0]
            x1 = x[:, 1]
            x = x0, x1
            return x
        if original:
            show_kp_orig(img, to_tuple(kp_pred))
        else:
            kp_pred = self.get_pred(kp_pred)  # y = W * x
            error_kp = np_dist_l1(kp_gt, kp_pred)
            error_kp *= mask
            error_mean = np.sum(error_kp) / np.sum(mask)
            pck_kp = np_pck_per_kp(kp_gt, kp_pred, thresh=self.d_pix)
            pck = np.mean(pck_kp)
            show_kp_regr(img, to_tuple(kp_gt), to_tuple(kp_pred),
                         error_kp, error_mean, pck_kp, pck, color_scale=self.d_pix, verbose=verbose)

    def visualize(self, testset=True, n_show=5, original=False, save=False, savepath=None):
        with tf.Session() as sess:
            self.saver.restore(sess, "../saved/" + self.name + ".ckpt")

            for j in range(n_show):
                if save:
                    verbose = 0
                else:
                    verbose = 1
                self.show_regress(original=original, testset=testset, verbose=verbose)
                if save:
                    assert os.path.exists(savepath)
                    plt.savefig('{}{}.png'.format(savepath, str(j).zfill(3)), bbox_inches='tight',
                                pad_inches=0, format='png', dpi=100)
                if not save:
                    plt.show(block=True)


def random_sample(x_train, y_train, sample_size):
    n = x_train.shape[0]
    select = np.random.randint(low=0, high=n - 1, size=sample_size)
    x_s = np.take(x_train, select, axis=0)
    y_s = np.take(y_train, select, axis=0)
    return x_s, y_s


def swap_yx(x):
    x1 = x[:, :, 2]  # kp vis
    x0 = x[:, :, 0:2]  # kp y, x
    x0 = x0[:, :, ::-1]  # -> x, y
    x1 = np.expand_dims(x1, axis=2)  # (n_s, n_kp, 1)
    x = np.concatenate([x0, x1], axis=2)  # x, y, vis
    return x


def dist_l1(labels, pred):
    x = tf.gather(labels, 0, axis=2)
    y = tf.gather(labels, 1, axis=2)
    x_ = tf.gather(pred, 0, axis=2)
    y_ = tf.gather(pred, 1, axis=2)
    result = tf.sqrt(tf.pow(x - x_, 2) + tf.pow(y - y_, 2))
    return result


def np_dist_l1(labels, pred):
    x = np.take(labels, 0, axis=1)
    y = np.take(labels, 1, axis=1)
    x_ = np.take(pred, 0, axis=1)
    y_ = np.take(pred, 1, axis=1)
    result = np.sqrt(np.power(x - x_, 2) + np.power(y - y_, 2))
    return result


def np_pck_per_kp(labels, pred, thresh):
    diff = np_dist_l1(labels, pred) - thresh
    is_correct = diff <= 0
    return is_correct
