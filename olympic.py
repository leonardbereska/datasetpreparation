import numpy as np
import glob
import cv2
import tensorflow as tf
import os
import scipy.io as sio
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


class Olympic(object):
    def __init__(self, path_to_dataset):
        self.path = path_to_dataset
        self.n_kp = 13  # todo clean
        # self.keypoints = self.get_kp()  # comment out for speed
        # self.bboxes = self.get_bb()
        self.path_data = path_to_dataset + 'frames/'
        self.path_label = path_to_dataset + 'labels/'
        self.make_train_test_lists()
        # todo train/test split
        # todo visibility of kp ask domili
        # self.n_images
        self.max_dim = [600, 600]
        self.all_actions = ['basketball_layup', 'bowling', 'clean_and_jerk', 'discus_throw', 'diving_platform_10m',
                            'diving_springboard_3m', 'hammer_throw', 'high_jump', 'javelin_throw', 'long_jump',
                            'pole_vault', 'shot_put', 'snatch', 'tennis_serve', 'triple_jump', 'vault']

        self.selected_actions = ['long_jump']
        self.bbox_factor = 2.

    def video_to_tfrecords(self, video_idx, list_imgpaths, list_keypoints, list_max_bbox, list_masks, action, save_path):
        """
        Create tfrecords file from video
        :param video_idx: int index of video
        :param list_imgpaths: all image paths in video
        :param list_keypoints: all keypoints as numpy array
        :param list_max_bbox: list of maxima of bounding box width/height
        :return:
        """
        out_path = os.path.join(save_path + action + "_" + str(video_idx+1).zfill(4) + ".tfrecords")

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


    def process(self):
        dir = 'processed/'
        make_dir(self.path + dir)
        make_dir(self.path + dir + 'train/')
        make_dir(self.path + dir + 'test/')
        make_dir(self.path + 'tfrecords/')


        # actions = glob.glob(self.path_data + '*')
        # actions = [a.split('/').pop() for a in actions]
        for action in self.selected_actions:
            assert action in self.all_actions
            make_dir(self.path+dir+'train/'+action + '/')
            make_dir(self.path+dir+'test/'+action+ '/')



            video_paths = glob.glob(self.path_data + action + '/*')
            for video_idx, video_path in enumerate(video_paths):

                # video_path = video_paths[0]
                video_ID = video_path.split('/').pop()
                if video_ID in self.test_list:
                    # save_dir = self.path + dir + 'train/' + video_ID + '/'
                    # make_dir(save_dir)
                    train = 'test/'
                elif video_ID in self.train_list:
                    train = 'train/'
                else:
                    raise NotImplementedError

                save_path = video_path.replace('frames/', dir + train)
                if os.path.exists(save_path):
                    print('video {} already done, continue..'.format(video_idx))
                    continue
                make_dir(save_path + '/')

                list_imgpaths = []
                list_keypoints = []
                list_max_bbox = []
                list_masks = []

                frames = sorted(glob.glob(video_path + '/*'))
                for frame_idx, frame in enumerate(frames):

                    # try: x.shape
                    # kp_x = x[frame_idx]
                    # kp_y = y[frame_idx]
                    # mask = visibility[frame_idx]
                    image = cv2.imread(frame)
                    # bb = bbox[frame_idx]
                    # # except IndexError:
                    # #     print('warning Indexerror')
                    # #     continue
                    # bb_w = int(bb[2] - bb[0])
                    # bb_h = int(bb[3] - bb[1])
                    # if not (bb_w > 0 and bb_h > 0):
                    #     print('bbox_w {}, bbox_h {} in frame {}, video {}, continue..'.format(bb_w, bb_h, frame_idx, video_idx))
                    #     continue
                    # center = [int((bb[2] + bb[0]) / 2), int((bb[3] + bb[1]) / 2)]  # x, y


                    # pad
                    # pad_x = self.max_dim[0]x((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'symmetric')


                    # crop around center
                    # crop0 = int(center[0] - self.max_dim[0] / 2)
                    # crop1 = int(center[1] - self.max_dim[1] / 2)
                    # kp_x -= crop0
                    # kp_y -= crop1
                    # center[0] -= crop0
                    # center[1] -= crop1
                    # image = image[crop1:crop1+480, crop0:crop0+480]



                    # crop around center
                    # bbox_factor = self.bbox_factor
                    # max_bbox = np.max([bb_w, bb_h])
                    # crop_w = int(max_bbox * bbox_factor)
                    # crop_h = int(max_bbox * bbox_factor)
                    # crop_x = int(center[0] - crop_w / 2)
                    # crop_y = int(center[1] - crop_h / 2)
                    # kp_x -= crop_x
                    # kp_y -= crop_y
                    # center[0] -= crop_x
                    # center[1] -= crop_y
                    # image = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]



                    # resize to bbox
                    out_shape = (self.max_dim[0], self.max_dim[1], 3)
                    image = imresize(image, out_shape)
                    # kp_y = kp_y / crop_h * out_shape[1]
                    # kp_x = kp_x / crop_w * out_shape[0]
                    # center[1] = center[1] / crop_h * out_shape[1]
                    # center[0] = center[0] / crop_w * out_shape[0]

                    # from matplotlib import pyplot as plt
                    # fig = plt.figure()
                    # plt.imshow(image)
                    # kp_x *= mask
                    # kp_y *= mask
                    # plt.scatter(kp_x, kp_y)
                    # plt.scatter(center[0], center[1], c='r')
                    # # plt.scatter(bb[0], bb[1], c='k')
                    # # plt.scatter(bb[2], bb[3], c='k')
                    # make_dir(self.path + 'matplot/')
                    # plt.savefig(self.path + 'matplot/video{}image{}.png'.format(video_idx, frame_idx), format='png')
                    # plt.close(fig)

                    # # visualize
                    # from matplotlib import pyplot as plt
                    # fig = plt.figure()
                    # plt.imshow(image)
                    # # plt.scatter(kp_x, kp_y)
                    # # plt.scatter(center[0], center[1], c='r')
                    # make_dir(self.path + 'matplot/')
                    # plt.savefig(self.path + 'matplot/image{}.png'.format(frame_idx), format='png')
                    # plt.close(fig)

                    image_path = frame.replace('frames/', dir+train)
                    dim_correct = (image.shape == (self.max_dim[0], self.max_dim[0], 3))
                    assert dim_correct, '{}'.format(image.shape)  # must be rectangular and same size
                    assert (image.dtype == np.uint8)
                    cv2.imwrite(image_path, image)
                    list_imgpaths.append(image_path)

                    # kp = np.concatenate([kp_x, kp_y], axis=0)
                    # assert 0 <= kp.all() < self.max_dim[0], 'kp not in image'
                    kp = np.array([0])
                    list_keypoints.append(kp)

                    max_bbox = out_shape[0]  # smaller of image size scaled to out size
                    list_max_bbox.append(max_bbox)

                save_path = self.path + 'tfrecords/' + train
                make_dir(save_path)
                self.video_to_tfrecords(video_idx, list_imgpaths, list_keypoints, list_max_bbox, list_masks, action, save_path)  # todo tfrecords train test
                print('{}'.format(video_idx))


path = '../../../../myroot/olympic/'
assert os.path.exists(path)
olymp = Olympic(path_to_dataset=path)
olymp.process()
