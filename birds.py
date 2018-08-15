import cv2
import os
import glob
import re
# import time
# from matplotlib.pyplot import imread
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import glob
import scipy.io as sio
import h5py
# import numpy as np
import sys
import os
import re
# import py
# import pysatCDF
from scipy.misc import imresize


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s


def alphanum_key(s):
    return [tryint(c) for c in re.split('([0-9]+)', s)]


def show_img(img):
    plt.imshow(img)
    plt.show(block=False)


def print_progress(count, total):
    # Percentage completion.
    pct_complete = float(count) / total

    # Status-message.
    # Note the \r which means the line should overwrite itself.
    msg = "\r- Progress: {0:.1%}".format(pct_complete)

    # Print it.
    sys.stdout.write(msg)
    sys.stdout.flush()


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_float32(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))  # wtf only float


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


# def convert(data, activ, save_path):
#     # Args:
#     # image_paths   List of file-paths for the images.
#     # labels        Class-labels for the images.
#     # out_path      File-path for the TFRecords output file.
#
#     list_img_path, list_video_nr, list_frame_idx, list_n_frames = data["img_paths"], data["video_idxs"], data["frame_idxs"], data["max_frame_idxs"]
#
#     for img_path, video_nr, frame_idx, n_frames in zip(list_img_path, list_video_nr, list_frame_idx, list_n_frames):
#         # Number of images. Used
#         # s when printing the progress.
#         num_images = len(img_path)
#
#         # Open a TFRecordWriter for the output-file.
#
#         out_path = os.path.join(save_path + activ + "_" + str(video_nr[0]).zfill(2) + ".tfrecords")
#         print("Converting: " + out_path)
#         with tf.python_io.TFRecordWriter(out_path) as writer:
#             # Iterate over all the image-paths and class-labels.
#             for i, (path, video_nr, frame_idx, n_frames) in enumerate(zip(img_path, video_nr, frame_idx, n_frames)):
#                 # Print the percentage-progress.
#                 print_progress(count=i, total=num_images - 1)
#
#                 with open(path, 'rb') as f:
#                     img_raw = f.read()
#
#                 # Convert the image to raw bytes.
#                 # img_bytes = img_raw.tostring()
#
#                 # Create a dict with the data we want to save in the
#                 # TFRecords file. You can add more relevant data here.
#                 data = \
#                     {
#                         'image': wrap_bytes(img_raw),
#                         'video': wrap_int64(video_nr),
#                         'frame': wrap_int64(frame_idx),
#                         'n_frames': wrap_int64(n_frames),
#                     }
#
#                 # Wrap the data as TensorFlow Features.
#                 feature = tf.train.Features(feature=data)
#
#                 # Wrap again as a TensorFlow Example.
#                 example = tf.train.Example(features=feature)
#
#                 # Serialize the data.
#                 serialized = example.SerializeToString()
#
#                 # Write the serialized data to the TFRecords file.
#
#                 writer.write(serialized)

def read_table(txt_path):
    with open(txt_path) as file:
        data = file.read()
        string_list = data.split('\n')
        if string_list[-1] == '':
            string_list.pop()
        string_list_list = [string.split(' ') for string in string_list]
        # return [int(x) for x in string_list_list]
        return string_list_list


def get_bb(c, bb_ratio, bb_w, bb_h, w, h):
    y_min = max(int(c[0] - bb_ratio * bb_h / 2), 0)
    x_min = max(int(c[1] - bb_ratio * bb_w / 2), 0)
    y_max = min(int(c[0] + bb_ratio * bb_h / 2), h)
    x_max = min(int(c[1] + bb_ratio * bb_w / 2), w)
    too_big = y_min == 0 or x_min == 0 or y_max == h or x_max == w
    return too_big, y_min, y_max, x_min, x_max


class Birds(object):
    def __init__(self, path_to_dataset):

        # ground truth key points:
        # 1 back, 2 beak, 3 belly, 4 breast, 5 crown, 6 forehead, 7 left eye, 8 left leg, 9 left wing, 10 nape
        # 11 right  eye, 12  right leg, 13 right wing, 14 tail, 15 throat

        self.path = path_to_dataset
        self.classes = read_table(self.path + 'classes.txt')
        self.train_test = read_table(self.path + 'train_test_split.txt')
        self.bboxes = read_table(self.path + 'bounding_boxes.txt')
        self.kp = read_table(self.path + 'parts/part_locs.txt')
        self.img_class = read_table(self.path + 'image_class_labels.txt')
        self.img_path = read_table(self.path + 'images.txt')
        self.bad_categories = read_table(self.path + 'classes_difficult.txt')  # sb: sea bird, bb: big bird, fb: flying bird, cm: camouflage

    def is_test_set(self, img_idx):
        return bool(self.train_test[img_idx - 1][1])

    def to_tf_records(self, video_name, video_idx, list_imgpaths, list_keypoints, list_max_bbox, save_dir='tfrecords/'):
        """
        Create tfrecords file from video
        :param video_name: int index of video
        :param list_imgpaths: all image paths in video
        :param list_keypoints: all keypoints as numpy array
        :param list_max_bbox: list of maxima of bounding box width/height
        :return:
        """
        save_path = self.path + save_dir
        make_dir(save_path)
        out_path = os.path.join(save_path + video_name + ".tfrecords")

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

    def process(self, save_in_dir='processed/', skip_if_video_exists=True, use_mask=True, out_shape=(300, 300, 3), bb_ratio=1.5, only_simple_classes=True, visible_thresh=0.8):

        save_path = self.path + save_in_dir
        make_dir(save_path)

        categories = glob.glob(self.path + 'images/' + '*')
        for cat_idx, cat in enumerate(categories):
            cat_name = self.classes[cat_idx][1]
            cat_dir = save_path + cat_name

            # Select only good categories:
            bad_category = False
            for bad_cat in self.bad_categories:
                if int(bad_cat[0]) == cat_idx+1 and only_simple_classes:  # is it a bad category?
                    # if bad_cat[1] == 'fb':  # except flying birds
                    #     break
                    # else:
                    #     continue
                    print('skip category {}, because {}'.format(cat_name, bad_cat[1]))
                    bad_category = True
            if bad_category:
                continue

            # make dir
            if not os.path.exists(cat_dir):
                os.mkdir(cat_dir)
            else:
                if skip_if_video_exists:
                    print('warning: video {}, {} exists, skipping conversion..'.format(cat_idx, cat_name))
                    continue
                else:
                    print('warning: video {}, {} exists, overwriting existing video'.format(cat_idx, cat_name))

            images = [int(img_class[0]) for img_class in self.img_class if int(img_class[1]) == cat_idx+1]
            list_kp = []
            list_max_bbox = []
            list_imgpaths = []
            for i in images:
                i -= 1
                image_path = self.img_path[i][1]
                image_path = self.path + 'images/' + image_path
                img = cv2.imread(image_path)
                bbox = self.bboxes[i][1:5]
                kp = self.kp[i*15:(i+1)*15]
                kp_xy = [k[2:5] for k in kp]   # x, y, visible

                # select only good images:
                kp_visible = [int(k[4]) for k in kp]
                visible_ratio = np.mean(kp_visible)
                if visible_ratio < visible_thresh:
                    print('skip image {}, because kp visible ratio < {}'.format(i, visible_ratio.round(2)))
                    continue


                bbox = [float(bb) for bb in bbox]
                bbox = [int(bb) for bb in bbox]

                # kp = [float(k) for k in kp

                if img is None:
                    print('warning: image does not exist for image {}'.format(i))
                    continue
                if bbox is None:
                    print('warning: bbox does not exist for image {}'.format(i))
                    continue
                bb_x, bb_y, bb_w, bb_h = bbox
                (h, w, _) = img.shape


                # x = kp[0]
                # y = kp[1]

                c = [int(bb_y+bb_h/2), int(bb_x+bb_w/2)]  # center of bbox

                # exclude images with bbox too close to edges
                margin = 0.125  # margin to both sides
                min_bb_ratio = 1. + margin * 2
                too_big, y_min, y_max, x_min, x_max = get_bb(c, min_bb_ratio, bb_w, bb_h, w, h)
                if too_big:
                    print('skip image {}, because bird (bbox) too big in image'.format(i))
                    continue
                real_bb_ratio = min(w/bb_w, h/bb_h)
                assert real_bb_ratio >= min_bb_ratio

                # mirror edges to both sides
                img_length = max(w, h)

                def double_margin(img_, new_margin, c):
                    # pad = 1000  # max(new_h, new_w)
                    pad = int((real_bb_ratio-1)/2 * img_length)
                    c[0] += pad
                    c[1] += pad
                    img_ = np.lib.pad(img_, ((pad, pad), (pad, pad), (0, 0)), 'symmetric')  # pad with a lot (mirror)
                    bb_ratio_crit = 1. + new_margin * 2
                    (new_h, new_w, _) = img_.shape
                    _, y_min, y_max, x_min, x_max = get_bb(c, bb_ratio_crit, bb_w, bb_h, new_w, new_h)  # get box
                    img_ = img_[y_min:y_max, x_min:x_max]  # crop to box
                    (new_h, new_w, _) = img_.shape
                    c = [int(new_h/2), int(new_w/2)]  # center of image
                    return img_, c

                img, c = double_margin(img, new_margin=0.5, c=c)
                img, c = double_margin(img, new_margin=1., c=c)
                # bb_ratio_crit = 1. + 1. * 2
                # (new_h, new_w, _) = img.shape
                # new_l = min(new_w, new_h)
                # _, y_min, y_max, x_min, x_max = get_bb(c, bb_ratio_crit, bb_w, bb_h, new_l, new_l)  # get box
                # img = img[y_min:y_max, x_min:x_max]  # crop to box
                # (new_h, new_w, _) = img.shape
                # c = [int(new_h/2), int(new_w/2)]  # center of image
                # (new_h, new_w, _) = img.shape
                # (out_w, out_h, _) = out_shape
                # _, y_min, y_max, x_min, x_max = get_bb(c, 3., new_w, new_h, out_w, out_h)  # get box

                # invert image to have same parity (birds looking in one direction)
                left_eye = kp_visible[6]
                right_eye = kp_visible[10]
                if left_eye and right_eye:  # exclude frontal view
                    print('skip image {}, because both eyes visible'.format(i))
                    continue  # both visible -> exclude
                if not left_eye and right_eye:
                    img = img[:, ::-1, :]  # parity transform if only right eye visible

                img = imresize(img, out_shape)
                # show_img(img)

                # Padding
                (bb_h, bb_w, _) = img.shape  # height and width of bbox
                max_bbox = np.max([bb_h, bb_w])  # parameter for size of person in image -> later used for zooming


                img = img.astype(int)
                # keypoint_arr = np.concatenate([kp_y, kp_x], axis=0)
                kp = np.array([0])

                # Save image
                assert(img.shape == out_shape)  # must be rectangular and same size
                assert (img.dtype == int)  # save as int to reduce disk usage
                frame_path = image_path.replace('images/', save_in_dir)
                cv2.imwrite(frame_path, img)  # only if well-tested

                list_imgpaths.append(frame_path)
                list_max_bbox.append(max_bbox)
                list_kp.append(kp)

            print('video {}'.format(cat_name))
            self.to_tf_records(cat_name, cat_idx, list_imgpaths, list_kp, list_max_bbox)


path_to_root = '../../../../myroot/'
path = path_to_root + 'birds/birds_simple/'
# path_to_harddrive = '/Volumes/Uni/human_backgr/'

# path = path_to_harddrive + ID + '/'

mybirds = Birds(path_to_dataset=path)
# build in the following order
# myhuman.convert_mg4_to_img()  # 1. extracts every nth frame of the mp4
mybirds.process(save_in_dir='processed/', use_mask=False, out_shape=(720, 720, 3), bb_ratio=3.0, only_simple_classes=True)  # 2. masks out background


# Pre-processing Protocol
# excluded classes: seabirds, flying birds, camouflage and bigger birds
# mirrored edges (used bbox)
# exclude frontal view (used eye kp)
# align parity (used eye kp)  -> most important
# exclude hidden (used all kp visibility)
