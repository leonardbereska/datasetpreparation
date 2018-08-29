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
from ops import to_tfrecords, make_dir
from ops_img import save_img


FRAME_RATE = 10
N_VID = 2  # maximum number of videos for each activity: 10^n_vid
N_IMG = 4  # maximum number of images for each video: 10^n_img


class Human(object):
    def __init__(self, path_to_dataset):
        # self.activities = ['directions', 'discussion', 'posing', 'waiting', 'greeting', 'walking', 'sitting', 'photo']
        self.activities = ['posing']
        self.path = path_to_dataset
        self.bboxes = None  # h and w of bounding boxes, list of activities of lists of videos of lists of frames
        self.keypoints = None  # keypoints

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

                    kp_x = kp[0]
                    kp_y = kp[1]

                    if use_mask:


                        # crop with bounding boxes
                        kp_x = kp_x - bb_xmin  # also crop keypoints
                        kp_y = kp_y - bb_ymin
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
                        kp_x = kp_x + pad_x
                        kp_y = kp_y + pad_y

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
                        kp_x += xmin  # keypoints processing
                        kp_y += ymin

                        img = img[0:img_size, 0:img_size, :]  # correct for wrong int rounding, cropping does not change kp
                        (h, w, _) = img.shape  # height and width of bbox

                    img = img.astype(int)
                    keypoint_arr = np.concatenate([kp_y, kp_x], axis=0)

                    # Save image
                    img_path = frame_path.replace('raw/', save_in_dir)

                    save_img(img, img_path, (img_size, img_size, 3))

                    list_imgpaths.append(img_path)
                    list_kp.append(keypoint_arr)
                to_tfrecords(root_path=self.path, video_name=activity + "_" + str(video_idx).zfill(N_VID), video_idx=video_idx,
                             list_imgpaths=list_imgpaths, list_keypoints=list_kp, list_masks=None)
                print('video {}'.format(video_idx))
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
