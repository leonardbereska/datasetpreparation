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


def show_kp(img, kp_x, kp_y):
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
    testtrain = ['test', 'train']
    with open(root_path + 'y_{}.txt'.format(testtrain[test_or_train]), 'a') as f:
        shape = (n_kp, 1)
        img_idx = np.full(shape, fill_value=i)  # +1 for starting at 1
        kp_idx = np.reshape(np.arange(n_kp) + 1, shape)  # +1 for starting at 1
        kp_out = np.transpose(np.reshape(kp, (2, n_kp)))
        kp_vis = np.reshape(kp_visible, newshape=shape)
        out = np.concatenate((img_idx, kp_idx, kp_out, kp_vis), axis=1)
        f.write("\n".join(" ".join(map(str, x)) for x in out) + '\n')

    # train/test split text file
    with open(root_path + '{}_img.txt'.format(testtrain[test_or_train]), 'a') as f:
        relative_path = os.path.relpath(frame_path, root_path)
        f.write('{} {}\n'.format(i + 1, relative_path))


#######################################################################################################################
# VIDEO OPS
#######################################################################################################################

def down_sample(path_to_video, save_path):
    """Select specific video with a high frame rate, to downsample by factor of 2"""
    image_paths = glob.glob(path_to_video + '*')
    image_paths = image_paths[::2]
    make_dir(save_path)
    for i, image_path in enumerate(image_paths):
        cv2.imwrite(save_path + str(i + 1).zfill(4) + '.jpg', cv2.imread(image_path))


def left_to_right(path_to_video, save_path):
    image_paths = glob.glob(path_to_video + '/*')
    make_dir(save_path)
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        image_rev = image[:, ::-1, :]
        cv2.imwrite(save_path + str(i + 1).zfill(4) + '.jpg', image_rev)


def extract_frames(root_path, from_dir='exported', to_dir='raw', video_format='mp4', img_format='jpg', frame_rate=2,
                   deinterlace=True):
    """
    Converts video from mp4/mov to img frames
    assumes all videos in one directory
    """

    o_img = 4  # order of magnitude: maximum number of images for each activity: 10^img
    # o_vid = 2  # save for videos
    make_dir(root_path + to_dir)
    video_paths = glob.glob(root_path + from_dir + '/*')  # can use other format also e.g. .avi

    for video_idx, video_path in enumerate(video_paths):
        video_name = video_path.split('/').pop()  # e.g. afghan_hound.mp4
        video_name = video_name.replace('.{}'.format(video_format), '')  # e.g. afghan_hound
        save_path = root_path + to_dir + '/' + video_name + '/'

        if not make_dir(save_path):
            print('warning: {} folder exists, skipping...'.format(video_name))
            continue

        video_capture = cv2.VideoCapture(video_path)

        img_count = 0
        img_i = 0
        success = True
        while success:
            success, image = video_capture.read()
            try:
                if img_count % frame_rate == 0:
                    img_name = save_path + str(img_i).zfill(o_img) + '.' + img_format  # save images in video folder

                    if deinterlace:
                        # de-interlacing: delete every second row and column
                        image = image[::2, ::2, :]

                    cv2.imwrite(img_name, image)  # save frame as JPEG file
                    img_i += 1
            except:
                print('could not extract frame, continuing...')
                pass
            img_count += 1
        print('Finished extracting video {}/{}: {}'.format(video_idx, len(video_paths), video_name))


def rename_files(root_path, bg0=False, start_id=0):
    # files = glob.glob(path + 'tfrecords/*')
    files = glob.glob(root_path + 'rename/*')

    for i, file in enumerate(files):
        # os.rename(file, file.replace('exported/', 'exported/'+str.zfill(str(i), 2)))
        # os.rename(file, file.replace('.tfrecords', '-'+'.tfrecords'))
        # n, s, a, tf = file.split('-')
        # os.rename(file, n + '-' + a + '-' + s + '-' + tf)

        a = i % 5
        s = int(i / 5)
        new_name = str.zfill(str(i), 4) + '-' + str(a) + '-' + str.zfill(str(s), 3)

        os.rename(file, root_path + new_name)

    # # files = glob.glob(path + 'tfrecords/*')
    # subject_files = glob.glob(path + 'rename/' + '*')
    # i_b = 1  # start at one (0 is for other dataset)
    # for i_s, file_s in enumerate(subject_files):
    #     # os.rename(file, file.replace('exported/', 'exported/'+str.zfill(str(i), 2)))
    #     # os.rename(file, file.replace('.tfrecords', '-'+'.tfrecords'))
    #     # n, s, a, tf = file.split('-')
    #     # os.rename(file, n + '-' + a + '-' + s + '-' + tf)
    #     # subject_name = file.split('/')[-1]
    #     # os.rename(file_s, path + 's' + str(i_s))
    #     s = str.zfill(str(i_s+start_id), 3)
    #     backgr_files = glob.glob(file_s + '/*')
    #     for file_b in backgr_files:
    #         if file_b.split('/')[-1] == 'x':
    #             b = 'xxx'
    #         else:
    #             b = str.zfill(str(i_b), 3)
    #         i_b += 1
    #         videos = glob.glob(file_b + '/*')
    #         for i_v, file_v in enumerate(videos):
    #             v = str(i_v)
    #             if bg0:
    #                 b = '000'
    #             new_name = path + 'exported/' + 's' + s + '-b' + b + '-a' + v
    #             os.rename(file_v, new_name)

