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
from helpers import to_tfrecords


FRAME_RATE = 10
N_VID = 2  # maximum number of videos for each activity: 10^n_vid
N_IMG = 4  # maximum number of images for each video: 10^n_img


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


class Human(object):
    def __init__(self, path_to_dataset):
        # self.activities = ['directions', 'discussion', 'posing', 'waiting', 'greeting', 'walking', 'sitting', 'photo']
        self.activities = ['posing']
        self.path = path_to_dataset
        self.bboxes = None  # h and w of bounding boxes, list of activities of lists of videos of lists of frames
        self.keypoints = None  # keypoints

    def convert_mg4_to_img(self):
        """
        Converts video from mp4 to img frames
        """
        global N_IMG
        global N_VID
        global FRAME_RATE  # save only every nth frame

        # all_modes = ['mask', 'raw', 'bbox']
        # for mode in modes:
        #     assert mode in all_modes
            # extract Mask
        mode_path = self.path + 'raw' + '/'
        make_dir(mode_path)

        for activity in self.activities:
            dir_path = mode_path + activity + '/'
            make_dir(dir_path)

            # if mode == 'mask':
            #     vid_path = self.path + 'MySegmentsMat/ground_truth_bs/'  # path to masks
            # if mode == 'raw':
            vid_path = self.path + 'Videos/'  # path to all raw videos
            # elif mode == 'bbox':
            #     vid_path = self.path + 'MySegmentsMat/ground_truth_bb/'
            activity = (activity[:1]).upper() + activity[1:]  # first letter upper case
            vid_path += activity
            video_paths = glob.glob(vid_path + '*')  # can use other format also e.g. .avi

            for vid_count, path in enumerate(video_paths):
                save_path = dir_path + str(vid_count).zfill(N_VID) + '/'  # video folder path
                if not os.path.exists(save_path):
                    os.mkdir(save_path)  # create video folder
                else:
                    print('warning: {} folder for {} exists, skipping conversion..'.format(vid_count, activity))
                    continue

                # if mode == 'bbox':
                #     with h5py.File(path, 'r') as f:
                #         # print(f.keys())  # have to know which keys to select
                #         all_frames = np.array(f['Masks'])
                #         img_i = 0
                #         for img_count, frame in enumerate(all_frames):
                #             bb = np.array(f[frame[0]])  # bb : bounding box is one-hot array at bb locations
                #             bb = bb.transpose()  # change x and y
                #             img = np.empty(shape=(bb.shape[0], bb.shape[1], 3), dtype=int)  # create RGB image
                #             for i in range(3):
                #                 img[:, :, i] = bb*255  # all color channels have bounding box one-hot coding
                #             if img_count % FRAME_RATE == 0:
                #                 img_path = save_path + str(img_i).zfill(N_IMG) + ".jpg"
                #                 assert img.dtype == int
                #                 cv2.imwrite(img_path, img)
                #                 img_i += 1
                #             img_count += 1
                #
                # else:
                video_capture = cv2.VideoCapture(path)

                img_count = 0
                img_i = 0
                success = True
                while success:
                    success, image = video_capture.read()
                    try:
                        if img_count % FRAME_RATE == 0:
                            img_name = save_path + str(img_i).zfill(N_IMG) + ".jpg"  # save images in video folder
                            cv2.imwrite(img_name, image)     # save frame as JPEG file
                            img_i += 1
                    except:
                        pass
                    img_count += 1
                print('video {}'.format(vid_count))
            print('Finished {} for {}'.format(activity, 'raw'))

    def get_mask(self, activity, video_idx, frame_idx):
        """
        Get background mask for specific activity, video and frame
        :param activity: string of activity from self.activities
        :param video_idx: int for video index
        :param frame_idx: int for frame index
        :return:
        """
        global FRAME_RATE
        vid_path = self.path + 'MySegmentsMat/ground_truth_bs/'

        activity_upper = (activity[:1]).upper() + activity[1:]  # first letter upper case
        vid_path += activity_upper
        video_paths = glob.glob(vid_path + '*')  # can use other format also e.g. .avi
        vid_path = video_paths[video_idx]  # choose specific video

        with h5py.File(vid_path, 'r') as f:
            # print(f.keys())  # have to know which keys to select
            all_frames = np.array(f['Masks'])
            try:
                mask = np.array(f[all_frames[frame_idx * FRAME_RATE][0]])  # choose specific frame
                mask = mask.transpose()
            except:
                mask = None
        return mask

    def get_bbox(self, activity, video_idx, frame_idx):
        """
        Get bounding box for specific activity, video and frame
        :param activity: string of activity from self.activities
        :param video_idx: int for video index
        :param frame_idx: int for frame index
        :return:
        """

        global FRAME_RATE
        vid_path = self.path + 'MySegmentsMat/ground_truth_bb/'

        activity_upper = (activity[:1]).upper() + activity[1:]  # first letter upper case
        vid_path += activity_upper
        video_paths = glob.glob(vid_path + '*')  # can use other format also e.g. .avi
        vid_path = video_paths[video_idx]  # choose specific video

        with h5py.File(vid_path, 'r') as f:
            # print(f.keys())  # have to know which keys to select
            all_frames = np.array(f['Masks'])
            try:
                bbox = np.array(f[all_frames[frame_idx * FRAME_RATE][0]])  # choose specific frame
                bbox = bbox.transpose()
            except:
                bbox = None
        return bbox

    def get_img_raw(self, activity, video_idx, frame_idx):
        """
        Get image for specific activity, video and frame
        :param activity: string of activity from self.activities
        :param video_idx: int for video index
        :param frame_idx: int for frame index
        :return:
        """
        # global FRAME_RATE
        path_to_activity = self.path + 'raw/' + activity + '/*'
        video_paths = glob.glob(path_to_activity)  # can use other format also e.g. .avi
        path_to_video = video_paths[video_idx]  # choose specific video
        assert os.path.exists(path_to_video)

        frame_paths = glob.glob(path_to_video + '/*')
        path_to_frame = frame_paths[frame_idx]
        if not os.path.exists(path_to_frame):
            print('warning: path to img not existing, return None')
            img = None
        else:
            img = cv2.imread(path_to_frame)
        return img

    def get_kp(self, activity, video_idx, frame_idx):

        Activity = (activity[:1]).upper() + activity[1:]  # first letter upper case, e.g. 'Sitting '
        keypoint_path = self.path + 'MyPoseFeatures/D2_Positions/' + Activity + '*.mat'  # select only matlab files
        video_paths = glob.glob(keypoint_path)
        pose_mat = video_paths[video_idx]  # select video  # todo is this really the same video as the index indicates?
        assert os.path.exists(pose_mat), 'matlab file for keypoints does not exist'
        f = sio.loadmat(pose_mat)
        video_frames = f['video_frames']
        original_idx = frame_idx * FRAME_RATE
        try:
            keypoints = video_frames[original_idx, :]
            keypoints = np.array(keypoints).reshape((32, 2))  # get 32 keypoints with coordinates
            keypoints = keypoints.transpose()
        except:
            keypoints = None
        # x, y = keypoints[0], keypoints[1]
        # plt.figure()
        # plt.scatter(x, y)
        # plt.imshow(img)
        # plt.show()
        return keypoints

    def video_to_tfrecords(self,  activity, video_idx, list_keypoints, list_max_bbox, from_dir='processed/',
                           save_dir='tfrecords/'):

        global N_VID

        save_path = self.path + save_dir
        make_dir(save_path)
        video_path = self.path + from_dir + activity + '/' + str(video_idx).zfill(N_VID)
        os.path.exists(video_path)
        video_path += '/*.jpg'
        frame_paths = glob.glob(video_path)
        out_path = os.path.join(save_path + activity + "_" + str(video_idx).zfill(N_VID) + ".tfrecords")

        print("Converting: " + out_path)
        with tf.python_io.TFRecordWriter(out_path) as writer:
            # Iterate over all the image-paths and class-labels.
            for i, (img_path, keypoints, max_bbox) in enumerate(zip(frame_paths, list_keypoints, list_max_bbox)):
                # Print the percentage-progress.
                print_progress(count=i, total=len(frame_paths) - 1)
                # print(type(keypoints))
                with open(img_path, 'rb') as f:
                    img_raw = f.read()

                # Convert the image to raw bytes.
                # img_bytes = img_raw.tostring()
                # Create a dict with the data we want to save in the
                # TFRecords file. You can add more relevant data here.
                data = \
                    {
                        'image': wrap_bytes(img_raw),   # todo do I need to convert to jpg still
                        'video': wrap_int64(video_idx),
                        'keypoints': wrap_float32(keypoints),
                        'bbox_max': wrap_int64(max_bbox)
                        # 'frame': wrap_int64(frame_idx),
                        # 'n_frames': wrap_int64(n_frames),

                    }

                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=data)

                # Wrap again as a TensorFlow Example.
                example = tf.train.Example(features=feature)

                # Serialize the data.
                serialized = example.SerializeToString()

                # Write the serialized data to the TFRecords file.

                writer.write(serialized)

    def process(self, save_in_dir='processed/', skip_if_video_exists=True, use_mask=True):

        save_path = self.path + save_in_dir
        make_dir(save_path)

        for activity in self.activities:

            make_dir(save_path + activity)
            videos = glob.glob(self.path + 'raw/' + activity + '/*')

            for video_idx, video in enumerate(videos):
                frames = glob.glob(video+'/*')

                video_dir = video.replace('raw/', save_in_dir)
                if not os.path.exists(video_dir):
                    os.mkdir(video_dir)
                else:
                    if skip_if_video_exists:
                        print('warning: video {} for {} exists, skipping conversion..'.format(video_idx, activity))
                        continue
                    else:
                        print('warning: video {} for {} exists, overwriting existing video'.format(video_idx, activity))

                list_kp = []
                list_imgpaths = []
                for frame_idx, frame_path in enumerate(frames):

                    # get data for respective activity, video_idx, frame_idx
                    if use_mask:
                        mask = self.get_mask(activity, video_idx, frame_idx)
                    # img = self.get_img_raw(activity, video_idx, frame_idx)
                    img = cv2.imread(frame_path)  # for speed
                    bbox = self.get_bbox(activity, video_idx, frame_idx)
                    kp = self.get_kp(activity, video_idx, frame_idx)

                    # mask and video not exactly aligned sometimes for last frame, -> just ignore those
                    if img is None:
                        print('warning: image does not exist for {}, video {}, frame {}'.format(activity, video_idx, frame_idx))
                        continue
                    if use_mask and mask is None:
                        print('warning: mask does not exist for {}, video {}, frame {}'.format(activity, video_idx, frame_idx))
                        continue
                    if bbox is None:
                        print('warning: bbox does not exist for {}, video {}, frame {}'.format(activity, video_idx, frame_idx))
                        continue
                    if kp is None:
                        print('warning: keypoints do not exist for {}, video {}, frame {}'.format(activity, video_idx, frame_idx))
                        continue

                    if use_mask:
                        # mask = (mask/255).astype(np.int8)  # values either 0 or 1 as int
                        assert(pix == 0 or pix == 1 for pix in mask)  # mask has values 0 or 1
                    # bbox = (bbox / 255).astype(np.int8)  # values either 0 or 1 as int
                    assert (pix == 0 or pix == 1 for pix in bbox)  # bbox has values 0 or 1

                    img_size = 600  # afterwards

                    # get bounding box:
                    # crop image
                    i_big = 1000
                    img = img[0:i_big, 0:i_big, :]  # make sure all images same size
                    if use_mask:
                        mask = mask[0:i_big, 0:i_big]
                    bbox = bbox[0:i_big, 0:i_big]
                    bbox_idx = np.nonzero(bbox)
                    bb_ymin = min(bbox_idx[0])
                    bb_xmin = min(bbox_idx[1])
                    bb_ymax = max(bbox_idx[0])
                    bb_xmax = max(bbox_idx[1])

                    x = kp[0]
                    y = kp[1]

                    if use_mask:


                        # crop with bounding boxes
                        x = x - bb_xmin  # also crop keypoints
                        y = y - bb_ymin
                        img = img[bb_ymin:bb_ymax, bb_xmin:bb_xmax]
                        mask = mask[bb_ymin:bb_ymax, bb_xmin:bb_xmax]

                        bckgr = np.empty(img.shape)
                        bckgr.fill(255)  # 1 for white

                        for i in range(3):
                            img[:, :, i] = mask * img[:, :, i] + (1 - mask) * bckgr[:, :, i]  # mask out image background

                        # Padding
                        (h, w, _) = img.shape  # height and width of bbox
                        max_bbox = np.max([h, w])  # parameter for size of person in image -> later used for zooming

                        pad_x = int((img_size+2 - w) / 2)  # get padding values as ints, slightly bigger than img_size
                        pad_y = int((img_size+2 - h) / 2)  # to be able to later resize it exactly to img_size

                        pad_x = max([0, pad_x])  # make sure padding is positive, if img to big do not pad, but cut img
                        pad_y = max([0, pad_y])

                        img = np.pad(img, pad_width=((pad_y, pad_y), (pad_x, pad_x), (0, 0)),
                                     mode='constant', constant_values=((255, 255), (255, 255), (255, 255)))

                        # adjust keypoints to padding
                        x = x + pad_x
                        y = y + pad_y

                        img = img[0:img_size, 0:img_size, :]  # correct for wrong int rounding, cropping does not change kp

                    else:
                        cy = int((bb_ymin + bb_ymax) / 2)
                        cx = int((bb_xmin + bb_xmax) / 2)  # image center
                        # d = int(max(bb_xmax - bb_xmin, bb_ymax - bb_ymin) / 2) + 100
                        d = int(img_size / 2)

                        # horrible code, but it ensures that the image is always
                        # properly (not to much) cut at the borders
                        if cx - d < 0:
                            xmin = 0
                            xmax = img_size
                        elif cx + d > 1000:
                            xmin = 1000 - img_size
                            xmax = 1000
                        else:
                            xmin = cx - d
                            xmax = cx + d
                        if cy - d < 0:
                            ymin = 0
                            ymax = img_size
                        elif cy + d > 1000:
                            ymin = 1000 - img_size
                            ymax = 1000
                        else:
                            ymin = cy - d
                            ymax = cy + d

                        img = img[ymin:ymax, xmin:xmax, :]  # cut around center of bbox
                        x += xmin  # keypoints processing
                        y += ymin

                        img = img[0:img_size, 0:img_size, :]  # correct for wrong int rounding, cropping does not change kp
                        (h, w, _) = img.shape  # height and width of bbox

                    img = img.astype(int)
                    keypoint_arr = np.concatenate([y, x], axis=0)

                    # Save image
                    assert(img.shape == (img_size, img_size, 3))  # must be rectangular and same size
                    assert (img.dtype == int)  # save as int to reduce disk usage
                    frame_path = frame_path.replace('raw/', save_in_dir)
                    cv2.imwrite(frame_path, img)  # only if well-tested

                    list_imgpaths.append(frame_path)
                    list_kp.append(keypoint_arr)
                print('video {}'.format(video_idx))
                to_tfrecords()
                self.video_to_tfrecords(activity=activity, video_idx=video_idx, list_keypoints=list_kp, list_max_bbox=list_max_bbox)
            print('Finished {}'.format(activity))




ID_list = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']  # , 'S11']
# ID_list = ['S6']

for ID in ID_list:
    path_to_root = '../../../../myroot/'
    path_to_harddrive = '/Volumes/Uni/human_backgr/'

    path = path_to_harddrive + ID + '/'

    myhuman = Human(path_to_dataset=path)
    # build in the following order
    # myhuman.convert_mg4_to_img()  # 1. extracts every nth frame of the mp4
    myhuman.process(save_in_dir='processed/', use_mask=False)  # 2. masks out background


# TODO
# time and improve processing frames -> is ok now
# apply background mask from matlab file -> done
# get keypoint locations from matlab file
