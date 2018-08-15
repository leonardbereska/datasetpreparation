import numpy as np
import glob
import cv2
import tensorflow as tf
import os
from scipy.misc import imresize

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_float32(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))  # wtf only float


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class CatHead(object):
    def __init__(self, path_to_dataset):
        self.path = path_to_dataset
        self.n_kp = 9  # todo only use 7
        # self.keypoints = self.get_kp()  # comment out for speed
        # self.bboxes = self.get_bb()
        # self.n_images
        # self.max_dim = [960]
        self.img_size = [600, 600]  # maximum in dataset: w 952 , h 919
    # Number
    # of
    # points(default is 9)
    # Left
    # Eye
    # Right
    # Eye
    # Mouth
    # Left
    # Ear - 1
    # Left
    # Ear - 2
    # Left
    # Ear - 3
    # Right
    # Ear - 1
    # Right
    # Ear - 2 - Right
    # Ear - 3

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
            if os.path.exists(save_path):
                print('video {} done already, skipping..'.format(video_idx))
                continue
            make_dir(save_path)

            list_imgpaths = []
            list_keypoints = []
            list_max_bbox = []

            for image_idx, image_path in enumerate(img_paths):

                # image_idx = 4
                # image_path = img_paths[image_idx]
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
                # if bb_w > self.max_dim[0] or bb_h > self.max_dim[1]:   # take all frames now
                #     print('size ({}, {}) too big'.format(bb_w, bb_h))
                #     continue
                center = [int((bb[2] + bb[0]) / 2), int((bb[3] + bb[1]) / 2)]  # x, y

                # pad
                pad_x = 1000  # self.max_dim[0]
                pad_y = 1000  # self.max_dim[1]
                kp_x += pad_x
                kp_y += pad_y
                center[0] += pad_x
                center[1] += pad_y
                image = np.lib.pad(image, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'symmetric')

                # crop around center
                crop_w = int(bb_w * 2)
                crop_h = int(bb_h * 2)
                crop_x = int(center[0] - crop_w / 2)
                crop_y = int(center[1] - crop_h / 2)
                kp_x -= crop_x
                kp_y -= crop_y
                center[0] -= crop_x
                center[1] -= crop_y
                image = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

                # resize to bbox
                out_shape = (self.img_size[0], self.img_size[1], 3)
                image = imresize(image, out_shape)
                kp_y = kp_y / crop_h * out_shape[1]
                kp_x = kp_x / crop_w * out_shape[0]
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

                kp = np.concatenate([kp_x, kp_y], axis=0)
                list_keypoints.append(kp)

                max_bbox = np.max([bb_w, bb_h])
                list_max_bbox.append(max_bbox)

            self.to_tf_records(video_idx, list_imgpaths, list_keypoints, list_max_bbox)
        #     print('{}/{}'.format(video_idx, n_videos))
        # # except IndexError:
        # #     print('Finished conversion')

    def to_tf_records(self, video_idx, list_imgpaths, list_keypoints, list_max_bbox, save_dir='tfrecords/'):
        """
        Create tfrecords file from video
        :param video_idx: int index of video
        :param list_imgpaths: all image paths in video
        :param list_keypoints: all keypoints as numpy array
        :param list_max_bbox: list of maxima of bounding box width/height
        :return:
        """
        save_path = self.path + save_dir
        make_dir(save_path)
        out_path = os.path.join(save_path + "_" + str(video_idx).zfill(2) + ".tfrecords")

        print("Converting: " + out_path)
        with tf.python_io.TFRecordWriter(out_path) as writer:
            # Iterate over all the image-paths and class-labels.
            for i, (img_path, keypoints, max_bbox) in enumerate(zip(list_imgpaths, list_keypoints, list_max_bbox)):
                with open(img_path, 'rb') as f:
                    img_raw = f.read()

                # Convert the image to raw bytes.
                # img_bytes = img_raw.tostring()
                # Create a dict with the data we want to save in the
                # TFRecords file. You can add more relevant data here.
                data = {'image': wrap_bytes(img_raw),
                        'video': wrap_int64(video_idx),
                        'keypoints': wrap_float32(keypoints),
                        'bbox_max': wrap_int64(max_bbox)

                }

                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=data)

                # Wrap again as a TensorFlow Example.
                example = tf.train.Example(features=feature)

                # Serialize the data.
                serialized = example.SerializeToString()

                # Write the serialized data to the TFRecords file.

                writer.write(serialized)

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
