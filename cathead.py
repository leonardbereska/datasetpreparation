import numpy as np
import glob
import cv2
import tensorflow as tf
import os
from ops import to_tfrecords, make_dir
from ops_img import pad_img, crop_img, resize_img, save_img


class CatHead(object):
    def __init__(self, path_to_dataset):
        self.path = path_to_dataset
        self.n_kp = 9
        self.res = 600  # maximum in dataset: w 952 , h 919
        self.shape = (self.res, self.res, 3)
    # Number of points(default is 9): Left Eye, Right Eye, Mouth,
    # Left Ear - 1 Left Ear - 2 Left Ear - 3 Right Ear - 1 Right Ear - 2 - Right Ear - 3

    def make_videos(self, orig_path='frames/', traintest=False):
        """
        Create short sequences (videos) of the 200.000 images
        :return:
        """
        make_dir(self.path + 'processed/')
        make_dir(self.path + 'tfrecords/')

        video_paths = glob.glob(self.path + orig_path + '*')
        for video_idx, video in enumerate(video_paths):

            img_paths = glob.glob(video + '/*.jpg')
            if not traintest:
                kp_paths = glob.glob(video + '/*.jpg.cat')

            save_dir = 'processed/'
            save_path = video.replace(orig_path, save_dir)

            if not make_dir(save_path):
                print('video {} done already, skipping..'.format(video_idx))
                continue

            list_imgpaths = []
            list_keypoints = []

            for image_idx, image_path in enumerate(img_paths):

                image = cv2.imread(image_path)

                # get kp
                if not traintest:
                    kp_path = kp_paths[image_idx]
                else:
                    kp_path = image_path.replace(orig_path, 'frames/') + '.cat'
                with open(kp_path) as file:
                    data = file.read()
                    kp_list = data.split(' ')
                    kp_list.pop(0)  # n kp
                    kp = []
                    for k in kp_list:
                        try:
                            kp.append(int(k))
                        except ValueError:
                            pass
                    assert len(kp) == self.n_kp * 2
                kp = np.array(kp)  # x, y, x, y ...

                # get bbox (from kp)
                kp_x = kp[::2]
                kp_y = kp[1::2]
                kp_bb_x = np.delete(kp_x, [4, 7], axis=0)
                kp_bb_y = np.delete(kp_y, [4, 7], axis=0)
                bb = [min(kp_bb_x), min(kp_bb_y), max(kp_bb_x), max(kp_bb_y)]
                bb_w = int(bb[2] - bb[0]) + 1
                bb_h = int(bb[3] - bb[1]) + 1
                center = [int((bb[2] + bb[0]) / 2), int((bb[3] + bb[1]) / 2)]  # x, y

                max_bbox = np.max([bb_w, bb_h])
                kp = [kp_x, kp_y]

                # pad
                image, kp, center = pad_img(image, kp=kp, center=center)

                # crop around center
                crop = (int(max_bbox * 2), int(max_bbox * 2))
                image, kp, center = crop_img(image, crop, kp, center)

                # resize to final size
                image, kp, center = resize_img(image, self.shape, kp, center)

                kp_x = kp[0]
                kp_y = kp[1]
                kp = np.concatenate([kp_x, kp_y], axis=0)

                image_path = image_path.replace(orig_path, save_dir)
                save_img(image, image_path, self.shape)

                list_imgpaths.append(image_path)
                list_keypoints.append(kp)

            to_tfrecords(self.path, str(video_idx).zfill(2), video_idx, list_imgpaths, list_keypoints, None)

    def get_test_train_split(self):
        """
        Save test/train split of Zhang et. al. in two folders ('train' and 'test')
        """
        def path_to_list(path):
            with open(path) as file:
                data = file.read()
                path_list = data.split('\n')
                path_list.pop()  # "
                out_list = []
                for i, image in enumerate(path_list):
                    out_list.append(image.replace('output_yixin/', ''))
            return out_list
        test_path = self.path + 'cat_data/testing_yixin.txt'
        train_path = self.path + 'cat_data/training_yixin.txt'

        test_list = path_to_list(test_path)
        train_list = path_to_list(train_path)

        def img_to_dir(img_path_list, part):
            make_dir(self.path + part)
            for image_path in img_path_list:
                image = cv2.imread(self.path + 'frames/' + image_path)
                save_path = self.path + part + image_path
                folder, image_id = image_path.split('/')
                make_dir(self.path + part + folder + '/')
                cv2.imwrite(save_path, image)

        img_to_dir(test_list, 'test/')
        img_to_dir(train_list, 'train/')


path = '../../../../myroot/cats2/'
assert os.path.exists(path)
cats = CatHead(path_to_dataset=path)
cats.make_videos('train/', traintest=True)

# cats.get_test_train_split()
