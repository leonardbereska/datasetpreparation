from matplotlib import pyplot as plt
import tensorflow as tf
import sys
import os
import re
import numpy as np
import glob
import cv2

"""
Helpers for pre-processing videos and images
- General operations: converting to tfrecords file, visualization, reading/writing to txt files
- Video operations
"""


#######################################################################################################################
# GENERAL OPS
#######################################################################################################################

def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)
        return True
    else:
        return False


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


def show_kp(img, kp):
    kp_x, kp_y = kp
    plt.imshow(img)
    plt.scatter(kp_x, kp_y)
    plt.show(block=False)


def show_kp_mask(image, kp, center, mask=None):
    from matplotlib import pyplot as plt
    fig = plt.figure()
    plt.imshow(image)
    kp_x, kp_y = kp
    if mask:
        kp_x *= mask
        kp_y *= mask
    plt.scatter(kp_x, kp_y)
    plt.scatter(center[0], center[1], c='r')
    # make_dir(self.path + 'matplot/')
    # plt.savefig(self.path + 'matplot/image{}.png'.format(frame_idx), format='png')
    plt.close(fig)


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_float32(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))  # wtf only float


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def read_table(txt_path, type_float=True):
    with open(txt_path) as file:
        data = file.read()
        string_list = data.split('\n')
        if string_list[-1] == '':
            string_list.pop()
        string_list_list = [string.split(' ') for string in string_list]
        # return [float(x) for x in string_list_list]
        if type_float:
            out = np.array(string_list_list).astype(float)
        else:
            out = string_list_list
        return out


def read_kp_data(txt_path, n_kp, d):  # todo kp visible
    """
    read in text file with columns (img_idx, kp_idx, kp_x, kp_y, kp_visible)
    :param txt_path: where to read data
    :param n_kp: number of keypoints
    :return: matrix (n_samples, n_kp, d), d=2 for images
    """

    table = read_table(txt_path)
    assert float(int(len(table) / n_kp)) == (len(table) / n_kp), 'n_kp wrong or txt file corrupted'
    n_samples = int(len(table) / n_kp)
    kp = table[:, 2:5]
    kp = np.reshape(kp, newshape=(n_samples, n_kp, d+1))  # +1 for kp_visible
    # kp_vis = table[:, 4]
    # kp_vis = np.reshape(kp_vis, newshape=(n_samples, n_kp))
    # kp[:, :, 2]
    # kp, kp_vis
    return kp, n_samples  # x, y, visible

def write_table(string_list_list, txt_path):
    with open(txt_path, 'a') as f:
        f.write("\n".join(" ".join(map(str, x)) for x in string_list_list) + '\n')


def to_tfrecords(root_path, video_name, video_idx, list_imgpaths, list_keypoints, list_masks):
    """
    Create tfrecords file from video
    """
    make_dir(root_path + 'tfrecords/')
    out_path = os.path.join(root_path + 'tfrecords/' + video_name + ".tfrecords")
    if not list_masks:
        list_masks = []
        mask = np.array([0])
        for i in range(len(list_imgpaths)):
            list_masks.append(mask)
    if not list_keypoints:
        list_keypoints = []
        kp = np.array([0])
        for i in range(len(list_imgpaths)):
            list_keypoints.append(kp)
    print("Converting: " + video_name)
    with tf.python_io.TFRecordWriter(out_path) as writer:
        # Iterate over all the image-paths and class-labels.
        for i, (img_path, keypoints, mask) in enumerate(zip(list_imgpaths, list_keypoints, list_masks)):
            with open(img_path, 'rb') as f:
                img_raw = f.read()

            # Convert the image to raw bytes.
            # img_bytes = img_raw.tostring()
            # Create a dict with the data we want to save in the
            # TFRecords file. You can add more relevant data here.
            data = {'image': wrap_bytes(img_raw),  # todo add frame number
                    'video': wrap_int64(video_idx),
                    'keypoints': wrap_float32(keypoints),
                    # 'bbox_max': wrap_int64(max_bbox),
                    'masks': wrap_float32(mask), }

            # Wrap the data as TensorFlow Features.
            feature = tf.train.Features(feature=data)

            # Wrap again as a TensorFlow Example.
            example = tf.train.Example(features=feature)

            # Serialize the data.
            serialized = example.SerializeToString()

            # Write the serialized data to the TFRecords file.
            writer.write(serialized)


def kp_to_txt(root_path, n_kp, kp, kp_visible, test_or_train, i, frame_path):
    # kp to txt file:
    shape = (n_kp, 1)
    relative_path = os.path.relpath(frame_path, root_path)

    if kp_visible is None:  # kp always visible if no kp mask
        kp_visible = np.ones(shape=shape)
    testtrain = ['test', 'train']
    with open(root_path + 'y_{}.txt'.format(testtrain[test_or_train]), 'a') as f:
        img_path = np.full(shape, fill_value=relative_path)
        img_idx = np.full(shape, fill_value=i)  # todo solve +1 for starting at 1
        kp_idx = np.reshape(np.arange(n_kp), shape)  # +1 for starting at 1
        kp_out = np.transpose(np.reshape(kp, (2, n_kp)))
        kp_vis = np.reshape(kp_visible, newshape=shape)

        img_idx = (img_idx.astype(int).astype(str))
        kp_idx = (kp_idx.astype(int).astype(str))
        kp_out = (np.round(kp_out.astype(float), 2).astype(str))
        kp_vis = (kp_vis.astype(int).astype(str))
        # out = np.concatenate((img_path, img_idx, kp_idx, kp_out, kp_vis), axis=1)
        out = np.concatenate((img_idx, kp_idx, kp_out, kp_vis), axis=1)
        f.write("\n".join(" ".join(map(str, x)) for x in out) + '\n')

    # train/test split text file
    with open(root_path + '{}_img.txt'.format(testtrain[test_or_train]), 'a') as f:
        f.write('{} {}\n'.format(i, relative_path))


def img_to_video(img):
    # TODO does not work so far
    # TODO have to check
    image_folder = '../../../../myroot/new/0'
    video_name = 'video.avi'

    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, -1, 1, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_folder, image)))

    cv2.destroyAllWindows()
    video.release()


"""
Make train-test split
- Extracts all image paths from folder structure, 
- make a train-test-split 
- and write these paths to text files.
"""


def get_all_img_paths(path_to_folder):
    """
    Get all paths to images in the folder and subfolders
    :return: relative_paths
    """
    all_subfolders = glob.glob(path_to_folder + '*')
    all_paths = []
    for folder in all_subfolders:
        all_paths.extend(glob.glob(folder + '/*'))
    # get relative paths
    common_prefix = path_to_folder
    relative_paths = [os.path.relpath(path, common_prefix) for path in all_paths]
    return relative_paths


def make_train_test_split(all_paths, train_test_ratio):

    # random split
    np.random.seed(1)
    train_size = int(len(all_paths)*train_test_ratio)
    index_list = np.random.choice(range(len(all_paths)), size=train_size, replace=None)
    list_train = [all_paths[i] for i in index_list]
    list_test = sorted(list(set(all_paths) - set(list_train)))
    return list_train, list_test


def list_to_txt(list, name):
    with open(name, 'w+') as file:
        for i, img_path in enumerate(list):
            file.write(img_path + '\n')


# path_to_root = '/Users/leonardbereska/myroot/'
# path = path_to_root + 'birds_big_processed/'
#
# all_img_paths = get_all_img_paths(path)
# list_train, list_test = make_train_test_split(all_img_paths, train_test_ratio=0.8)
# list_to_txt(list_train, 'birds_train.txt')
# list_to_txt(list_test, 'birds_test.txt')


