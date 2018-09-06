import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from ops_general import show_kp, read_table, read_kp_data


# def artificial_data(dim_x, dim_y, d):
#     n = 10000  # sample size
#     x_train = np.random.rand(n, dim_x, d)
#     A = np.random.rand(dim_y, dim_x)
#     y_train = np.einsum('jk,ikl->ijl', A, x_train)


# y = W * x (y: ground truth, x: predictions)
# read in txt files to tables

class Regressor(object):
    def __init__(self, name, dim_y, dim_x, b, lr, n_epochs, path_to_files, path_to_data, testpath, trainpath, process_kp, img_size):
        self.name = name
        self.path_to_data = path_to_data
        self.path_to_files = path_to_files
        self.testpath = testpath
        self.trainpath = trainpath

        # parameters
        self.dim_y = dim_y
        self.dim_x = dim_x
        self.b = b
        self.lr = lr
        self.n_epochs = n_epochs
        self.img_size = img_size

        # data
        self.y_train, n_train = read_kp_data(path_to_files + 'y_train.txt', n_kp=self.dim_y, d=2)
        self.y_test, n_test = read_kp_data(path_to_files + 'y_test.txt', n_kp=self.dim_y, d=2)  # self.y_train
        self.x_test = process_kp(np.load(path_to_files + self.testpath))
        self.x_train = process_kp(np.load(path_to_files + self.testpath))

        # variables
        self.X = tf.placeholder(dtype=tf.float64, shape=(b, dim_x, 2))
        self.Y = tf.placeholder(dtype=tf.float64, shape=(b, dim_y, 2 + 1))  # +1 for kp visibility mask
        self.W = tf.Variable(np.random.rand(dim_y * 2, dim_x * 2), name="weight")
        self.X_ = tf.reshape(self.X, [b, dim_x * 2])
        self.Y_ = tf.einsum('jk,ik->ij', self.W, self.X_)
        self.Y_ = tf.reshape(self.Y_, [b, dim_y, 2])

        self.loss = self.loss_l1(self.Y, self.Y_)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    def loss_l1(self, labels, pred):
        assert labels.shape == (self.b, self.dim_y, 2 + 1)
        dist = dist_l1(labels, pred)
        mask = tf.gather(labels, 2, axis=2)
        dist = mask * dist  # mask out non-visible kp
        result = tf.reduce_mean(dist)
        return result

    def pck(self, x_s, y_s, d_=6):
        d_pix = d_ / 128. * self.img_size
        x_s = x_s.astype(np.float64)
        x_s = tf.constant(x_s)
        x_s = tf.reshape(x_s, [self.b, self.dim_x * 2])

        y_ = tf.einsum('jk,ik->ij', self.W, x_s)
        y_ = tf.reshape(y_, [self.b, self.dim_y, 2])

        diff = dist_l1(y_s, y_) - d_pix
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
                    print("epoch:", '%04d' % (epoch + 1), "loss=", "{:.9f}".format(c))  # todo also pck

            x_s, y_s = random_sample(self.x_train, self.y_train, self.b)
            training_loss = sess.run(self.loss, feed_dict={self.X: x_s, self.Y: y_s})
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
                length = 10
            for i in range(length):  # go through test set in batch-size piecewise fashion, then average
                if testset:
                    select = np.arange(i * self.b, (i + 1) * self.b)  # same batch size as before
                    x_s = np.take(pred, select, axis=0)
                    y_s = np.take(labels, select, axis=0)
                else:
                    x_s, y_s = random_sample(self.x_train, self.y_train, self.b)

                batch_pck, batch_pck_per_kp = self.pck(x_s, y_s)
                batch_loss = sess.run(self.loss, feed_dict={self.X: x_s, self.Y: y_s})

                testing_loss += batch_loss
                testing_pck += batch_pck
                testing_pck_per_kp += batch_pck_per_kp

            testing_loss /= length
            testing_pck /= length
            testing_pck_per_kp /= length

            print('{} {} = {}'.format(testname, 'loss', testing_loss))
            print('{} {} = {}'.format(testname, 'pck', testing_pck))
            print('{} {} = {}'.format(testname, 'pck_per_kp', testing_pck_per_kp))

            # print("Absolute mean square loss difference:", abs(training_loss - testing_loss))  # todo implement

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

    def show_regress(self, original, testset):
        pred, labels, testname = self.choose_set(testset)
        n = pred.shape[0]
        idx = np.random.randint(low=0, high=n - 1, size=1)[0]
        plt.figure(idx)
        kp_in = pred[idx]
        if original:
            out = kp_in
        else:
            kp = kp_in.astype(np.float64)
            kp = tf.reshape(kp, [2 * self.dim_x, 1])
            out = tf.matmul(self.W, kp)
            out = tf.reshape(out, [self.dim_y, 2]).eval()
        kp = out
        kp_x = kp[:, 0]
        kp_y = kp[:, 1]
        kp = kp_x, kp_y

        kp_x = labels[idx][:, 0]
        kp_y = labels[idx][:, 1]
        kp_gt = kp_x, kp_y

        path_img = read_table(self.path_to_files + '{}_img.txt'.format(testname), type_float=False)[idx][1]
        img = cv2.imread(self.path_to_data + path_img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        show_kp(img, kp)
        show_kp(img, kp_gt)

    def visualize(self, testset=True, n_show=5, original=False):
        with tf.Session() as sess:
            self.saver.restore(sess, "../saved/" + self.name + ".ckpt")

            for j in range(n_show):  # todo print settings
                self.show_regress(original=original, testset=testset)  # todo implement show error, pck
                plt.show(block=True)


def random_sample(x_train, y_train, sample_size):  # todo rewrite
    n = x_train.shape[0]
    select = np.random.randint(low=0, high=n - 1, size=sample_size)  # todo optimize speed: not take from whole dataset
    x_s = np.take(x_train, select, axis=0)
    y_s = np.take(y_train, select, axis=0)
    return x_s, y_s


def dist_l1(labels, pred):
    x = tf.gather(labels, 0, axis=2)
    y = tf.gather(labels, 1, axis=2)
    x_ = tf.gather(pred, 0, axis=2)
    y_ = tf.gather(pred, 1, axis=2)
    result = tf.sqrt(tf.pow(x - x_, 2) + tf.pow(y - y_, 2))
    return result
