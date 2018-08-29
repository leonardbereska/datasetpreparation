import numpy as np
import glob
import cv2
import tensorflow as tf
import os
import scipy.io as sio
from scipy.misc import imresize
from ops import to_tfrecords, make_dir
from ops_img import save_img, resize_img


class Olympic(object):
    def __init__(self, path_to_dataset):
        self.path = path_to_dataset
        self.n_kp = 13  # todo clean
        # self.keypoints = self.get_kp()  # comment out for speed
        # self.bboxes = self.get_bb()
        # self.n_images
        self.res = 600
        self.shape = (self.res, self.res, 3)
        self.path_data = path_to_dataset + 'frames/'
        self.path_label = path_to_dataset + 'labels/'
        self.make_train_test_lists()

        self.all_actions = ['basketball_layup', 'bowling', 'clean_and_jerk', 'discus_throw', 'diving_platform_10m',
                            'diving_springboard_3m', 'hammer_throw', 'high_jump', 'javelin_throw', 'long_jump',
                            'pole_vault', 'shot_put', 'snatch', 'tennis_serve', 'triple_jump', 'vault']

        self.selected_actions = ['long_jump']

    def make_train_test_lists(self):
        path = self.path + 'train_test_split/'
        test_path = path + 'test/*'
        all_txt = glob.glob(test_path)
        self.test_list = []
        for txt in all_txt:
            with open(txt) as file:
                data = file.read()
                self.test_list += data.split('\n')

        train_path = path + 'train/*'
        all_txt = glob.glob(train_path)
        self.train_list = []
        for txt in all_txt:
            with open(txt) as file:
                data = file.read()
                self.train_list += data.split('\n')

    def is_testset(self, video_path):
        video_ID = video_path.split('/').pop()
        if video_ID in self.test_list:
            test = True
        elif video_ID in self.train_list:
            test = False
        else:
            raise NotImplementedError
        return test

    def process(self, make_trainset=True):
        dir = 'processed/'
        make_dir(self.path + dir)
        make_dir(self.path + dir + 'train/')
        make_dir(self.path + dir + 'test/')
        make_dir(self.path + 'tfrecords/')

        # actions = glob.glob(self.path_data + '*')
        # actions = [a.split('/').pop() for a in actions]
        for action in self.selected_actions:
            assert action in self.all_actions
            # make_dir(self.path+dir+'train/'+action + '/')
            # make_dir(self.path+dir+'test/'+action+ '/')

            video_paths = glob.glob(self.path_data + action + '/*')
            for video_idx, video_path in enumerate(video_paths):

                # video_path = video_paths[0]

                if make_trainset:
                    if self.is_testset(video_path):
                        continue
                else:
                    if not self.is_testset(video_path):
                        continue

                save_path = video_path.replace('frames/', dir)
                if os.path.exists(save_path):
                    print('video {} already done, continue..'.format(video_idx))
                    continue
                make_dir(save_path + '/')

                list_imgpaths = []

                frames = sorted(glob.glob(video_path + '/*'))
                for frame_idx, frame in enumerate(frames):

                    image = cv2.imread(frame)

                    image = resize_img(image, self.shape)

                    image_path = frame.replace('frames/', dir)
                    save_img(img=image, img_path=image_path, img_shape=self.shape)
                    list_imgpaths.append(image_path)

                to_tfrecords(self.path, action + "_" + str(video_idx + 1).zfill(4), video_idx,
                             list_imgpaths, list_keypoints=None, list_masks=None)

                print('{}'.format(video_idx))


path = '../../../../myroot/olympic/'
assert os.path.exists(path)
olymp = Olympic(path_to_dataset=path)
olymp.process()
