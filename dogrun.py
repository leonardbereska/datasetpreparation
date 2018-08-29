import numpy as np
import glob
import cv2
import tensorflow as tf
import os
from scipy.misc import imresize
from helpers import to_tfrecords, make_dir, extract_frames


class DogRun(object):
    def __init__(self, path_to_dataset):
        self.path = path_to_dataset
        self.img_size = [600, 600]  # todo find maximum in dataset

    def make_videos(self, orig_path='raw/'):
        """
        :return:
        """
        make_dir(self.path + 'tfrecords/')
        make_dir(self.path + 'processed/')

        video_paths = glob.glob(self.path + orig_path + '*')
        for video_idx, video in enumerate(video_paths):

            img_paths = glob.glob(video + '/*.jpg')

            save_dir = 'processed/'
            save_path = video.replace(orig_path, save_dir)
            if os.path.exists(save_path):
                print('video {} done already, skipping..'.format(video_idx))
                continue
            make_dir(save_path)

            list_imgpaths = []
            list_keypoints = []
            list_masks = []

            for image_idx, image_path in enumerate(img_paths):

                image = cv2.imread(image_path)
                try:
                    h, w, _ = image.shape
                except AttributeError:
                    print('skipping empty image {} in video {}'.format(image_idx, video_idx))
                    continue

                # resize to quadratic

                center = [int(w / 2), int(h / 2)]  # x, y

                # pad
                pad_x = 1000  # self.max_dim[0]
                pad_y = 1000  # self.max_dim[1]

                center[0] += pad_x
                center[1] += pad_y
                image = np.lib.pad(image, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'symmetric')

                # crop around center
                c = max(w, h)
                crop_w = c
                crop_h = c
                crop_x = int(center[0] - crop_w / 2)
                crop_y = int(center[1] - crop_h / 2)
                center[0] -= crop_x
                center[1] -= crop_y
                image = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

                # resize to bbox
                out_shape = (self.img_size[0], self.img_size[1], 3)
                image = imresize(image, out_shape)
                center[1] = center[1] / crop_h * out_shape[1]
                center[0] = center[0] / crop_w * out_shape[0]

                # # visualize
                # from matplotlib import pyplot as plt
                # i = 7
                # plt.imshow(image)
                # plt.scatter(kp_x[i], kp_y[i])
                # plt.scatter(center[0], center[1], c='r')  # 7, 4

                image_path = image_path.replace(orig_path, save_dir)
                dim_correct = (image.shape == (self.img_size[0], self.img_size[0], 3))
                assert dim_correct, '{}'.format(image.shape)  # must be rectangular and same size
                assert (image.dtype == np.uint8)
                cv2.imwrite(image_path, image)
                list_imgpaths.append(image_path)

                kp = np.array([0])
                list_keypoints.append(kp)

                # max_bbox = int(min(w, h) / c * out_shape[0])  # smaller of image size scaled to out size
                # list_max_bbox.append(max_bbox) # todo delete
                # mask # todo create
                mask = kp * 0. + 1. # todo check
                list_masks.append(mask)

            # save tfrecords
            make_dir(self.path + 'tfrecords/')
            out_path = os.path.join(self.path + 'tfrecords/' + "_" + str(video_idx).zfill(2) + ".tfrecords")
            to_tfrecords(out_path, video_idx, list_imgpaths, list_keypoints, list_masks)


path = '../../../../myroot/'
assert os.path.exists(path)

dogs = DogRun(path_to_dataset=path)

# extract_frames(path, from_dir='exported', to_dir='raw', video_format='mov', img_format='jpg', frame_rate=2)

dogs.make_videos()

# for video in ['002', '003', '004', '005', '006', '007', '008', '009']:
# dogs.left_to_right('007')
