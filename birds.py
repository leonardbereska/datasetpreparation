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


def show_kp(img, kp_x, kp_y):
    plt.imshow(img)
    plt.scatter(kp_x, kp_y)
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
        self.n_kp = 15

    def is_test_set(self, img_idx):
        return bool(self.train_test[img_idx - 1][1])

    def to_tfrecords(self, video_name, video_idx, list_imgpaths, list_keypoints, list_max_bbox, list_masks, save_dir='tfrecords/'):
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
            for i, (img_path, keypoints, max_bbox, mask) in enumerate(zip(list_imgpaths, list_keypoints, list_max_bbox, list_masks)):
                with open(img_path, 'rb') as f:
                    img_raw = f.read()

                # Convert the image to raw bytes.
                # img_bytes = img_raw.tostring()
                # Create a dict with the data we want to save in the
                # TFRecords file. You can add more relevant data here.
                data = {'image': wrap_bytes(img_raw),
                        'video': wrap_int64(video_idx),
                        'keypoints': wrap_float32(keypoints),
                        'bbox_max': wrap_int64(max_bbox),
                        'masks': wrap_float32(mask),
                        }

                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=data)

                # Wrap again as a TensorFlow Example.
                example = tf.train.Example(features=feature)

                # Serialize the data.
                serialized = example.SerializeToString()

                # Write the serialized data to the TFRecords file.
                writer.write(serialized)

    def process(self, save_in_dir='processed/', skip_if_video_exists=True, out_shape=(720, 720, 3),
                only_simple_classes=True, visible_thresh=None, align_parity=True, bb_margin=0.125, exclude_big=True, make_trainset=True, make_tfrecords=False):

        save_path = self.path + save_in_dir
        make_dir(save_path)

        categories = glob.glob(self.path + 'images/' + '*')
        for cat_idx, cat in enumerate(categories):
            cat_name = self.classes[cat_idx][1]
            cat_dir = save_path + cat_name

            # exclude certain bird categories (e.g. sea birds, big birds..)
            if only_simple_classes:  # select only good categories:
                bad_category = False
                for bad_cat in self.bad_categories:
                    if int(bad_cat[0]) == cat_idx+1: # is it a bad category?
                        print('skip category {}, because {}'.format(cat_name, bad_cat[1]))
                        bad_category = True
                if bad_category:
                    continue

            # make directory
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
            list_masks = []

            for i in images:
                i -= 1
                image_path = self.img_path[i][1]
                image_path = self.path + 'images/' + image_path
                img = cv2.imread(image_path)

                # train-test split
                img_in_trainset = bool(int(self.train_test[i][1]))
                if make_trainset:
                    if not img_in_trainset:
                        # print('exclude test set')
                        continue  # img in test set -> skip for making train set
                elif not make_trainset:
                    if img_in_trainset:
                        # print('exclude train set')
                        continue  # img in train set -> skip for making test set
                else:
                    raise NotImplementedError

                # get bbox
                bbox = self.bboxes[i][1:5]
                bbox = [float(bb) for bb in bbox]
                bbox = [int(bb) for bb in bbox]

                # get keypoints
                kp = self.kp[i*15:(i+1)*15]
                kp_xy = [k[2:4] for k in kp]   # x, y, visible; dtype in strings
                kp_x = [kp[0] for kp in kp_xy]
                kp_y = [kp[1] for kp in kp_xy]
                kp_x = [float(k) for k in kp_x]  # convert strings to float
                kp_y = [float(k) for k in kp_y]
                kp_x = np.array(kp_x)  # convert to array
                kp_y = np.array(kp_y)

                # create visibility mask for hidden kp
                kp_visible = [int(k[4]) for k in kp]
                kp_visible = np.array(kp_visible)
                kp_visible = kp_visible.astype(np.float32)

                # select only images with visible kp:
                if visible_thresh is not None:
                    visible_ratio = np.mean(kp_visible)
                    if visible_ratio < visible_thresh:
                        print('skip image {}, because kp visible ratio < {}'.format(i, visible_ratio.round(2)))
                        continue

                if img is None:
                    print('warning: image does not exist for image {}'.format(i))
                    continue
                if bbox is None:
                    print('warning: bbox does not exist for image {}'.format(i))
                    continue

                # exclude images with bbox too close to edges
                bb_x, bb_y, bb_w, bb_h = bbox
                max_bbox = np.max([bb_h, bb_w])  # parameter for size of object in image -> later used for zooming
                (h, w, _) = img.shape
                c = [int(bb_y + bb_h / 2), int(bb_x + bb_w / 2)]  # center of bbox
                min_bb_ratio = 1. + bb_margin * 2  # margin to both sides, need complete bird in image
                too_big, y_min, y_max, x_min, x_max = get_bb(c, min_bb_ratio, bb_w, bb_h, w, h)
                if too_big and exclude_big:
                    print('skip image {}, because bird (bbox) too big in image'.format(i))
                    continue
                real_bb_ratio = min(w/bb_w, h/bb_h)
                assert real_bb_ratio >= min_bb_ratio

                # padding sides: mirror edges to both sides
                img_length = max(w, h)

                def double_margin(img_, new_margin, c, kp_x, kp_y):
                    # pad = 1000  # max(new_h, new_w)
                    pad = int((real_bb_ratio-1)/2 * img_length)
                    c[0] += pad
                    c[1] += pad
                    kp_x += float(pad)
                    kp_y += float(pad)
                    img_ = np.lib.pad(img_, ((pad, pad), (pad, pad), (0, 0)), 'symmetric')  # pad with a lot (mirror)
                    bb_ratio_crit = 1. + new_margin * 2
                    (new_h, new_w, _) = img_.shape
                    _, y_min, y_max, x_min, x_max = get_bb(c, bb_ratio_crit, bb_w, bb_h, new_w, new_h)  # get box
                    img_ = img_[y_min:y_max, x_min:x_max]  # crop to box
                    kp_x -= float(x_min)
                    kp_y -= float(y_min)
                    (new_h, new_w, _) = img_.shape
                    c = [int(new_h/2), int(new_w/2)]  # center of image
                    return img_, c, kp_x, kp_y

                img, c, kp_x, kp_y = double_margin(img, new_margin=0.5, c=c, kp_x=kp_x, kp_y=kp_y)
                img, c, kp_x, kp_y = double_margin(img, new_margin=1., c=c, kp_x=kp_x, kp_y=kp_y)

                # invert image to have same parity (birds looking in one direction)
                if align_parity:
                    left_eye = kp_visible[6]
                    right_eye = kp_visible[10]
                    if left_eye and right_eye:  # exclude frontal view
                        print('skip image {}, because both eyes visible'.format(i))
                        continue  # both visible -> exclude
                    if not left_eye and right_eye:
                        img = img[:, ::-1, :]  # parity transform if only right eye visible
                        _, w, _ = img.shape
                        kp_x = w - kp_x  # invert kp x location

                # resize image to intended final resolution
                (h_in, w_in, _) = img.shape
                (h_out, w_out, _) = out_shape
                img = imresize(img, out_shape)  # resize image
                kp_x = kp_x / w_in * w_out  # resize kp
                kp_y = kp_y / h_in * h_out

                kp_x = [kp * kp_visible[k] for k, kp in enumerate(kp_x)]  # mask out hidden kp: set to zero
                kp_y = [kp * kp_visible[k] for k, kp in enumerate(kp_y)]

                img = img.astype(int)
                kp = np.concatenate([kp_y, kp_x], axis=0)
                kp = kp.astype(np.float32)

                # save image (for visual debugging)
                assert(img.shape == out_shape)  # must be rectangular and same size
                assert (img.dtype == int)  # save as int to reduce disk usage
                assert kp.dtype == np.float32, print('kp dtype: {}'.format(kp.dtype))
                assert max(kp_x) <= w_out
                if max(kp_y) >= h_out:
                    print('warning: max(kp_y): {} > h_out: {}'.format(max(kp_y), h_out))
                assert min(kp) >= 0.
                assert kp_visible.dtype == np.float32
                frame_path = image_path.replace('images/', save_in_dir)
                cv2.imwrite(frame_path, img)  # only if well-tested

                # add relevant info to lists (for tfrecords)
                list_imgpaths.append(frame_path)
                list_max_bbox.append(max_bbox)
                list_kp.append(kp)
                list_masks.append(kp_visible)

                # kp to txt file:
                testtrain = ['test', 'train']
                with open(self.path + 'y_{}.txt'.format(testtrain[make_trainset]), 'a') as f:
                    shape = (self.n_kp, 1)
                    img_idx = np.full(shape, fill_value=i+1)   # +1 for starting at 1
                    kp_idx = np.reshape(np.arange(self.n_kp)+1, shape)  # +1 for starting at 1
                    kp_out = np.transpose(np.reshape(kp, (2, self.n_kp)))
                    kp_vis = np.reshape(kp_visible, newshape=shape)
                    out = np.concatenate((img_idx, kp_idx, kp_out, kp_vis), axis=1)
                    f.write("\n".join(" ".join(map(str, x)) for x in out)+'\n')

                # train/test split text file
                with open(self.path + '{}_img.txt'.format(testtrain[make_trainset]), 'a') as f:
                    relative_path = os.path.relpath(frame_path, self.path)
                    f.write('{} {}\n'.format(i+1, relative_path))

            print('video {}'.format(cat_name))
            if make_tfrecords:
                self.to_tfrecords(cat_name, cat_idx, list_imgpaths, list_kp, list_max_bbox, list_masks)  # save in tfrecords


path_to_root = '/Users/leonardbereska/myroot/'
path = path_to_root + 'birds/birds/'
# path_to_harddrive = '/Volumes/Uni/human_backgr/'
assert os.path.exists(path)
# path = path_to_harddrive + ID + '/'

mybirds = Birds(path_to_dataset=path)

mybirds.process(save_in_dir='processed/', out_shape=(720, 720, 3),
                only_simple_classes=True, visible_thresh=0.8, align_parity=True, bb_margin=0.0, exclude_big=True, make_trainset=False)


# Pre-processing Protocol
# birds_big
# - resize to quadratic
# - center on bbox
# - excluded classes: seabirds, flying birds, camouflage and bigger birds
# - mirrored edges (used bbox), excluded too big birds (is this not included in excluding hidden)  # which bb_margin?
# - exclude frontal view (used eye kp)
# - align parity (used eye kp)  -> most important step
# - exclude hidden (used all kp visibility: threshold 0.8)

# bird_parity: like birds_big but without aligning parity and frontal-view exclusion + exclude test_set

# birds_complete
# - resize
# - center on bbox, mirror edges multiple times
# - exclude test set

