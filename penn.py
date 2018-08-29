import numpy as np
import glob
import cv2
import os
import scipy.io as sio
from scipy.misc import imresize
from ops import to_tfrecords, make_dir
from ops_img import pad_img, crop_img, save_img, resize_img


class PennAction(object):
    def __init__(self, path_to_dataset):
        self.path = path_to_dataset
        self.n_kp = 13
        self.path_data = path_to_dataset + 'frames/'
        self.path_label = path_to_dataset + 'labels/'
        # todo train/test split
        # todo visibility of kp ask domili
        # self.n_images
        self.res = 600
        self.shape = (self.res, self.res, 3)
        self.all_actions = ['baseball_pitch', 'clean_and_jerk', 'pull_ups', 'strumming_guitar', 'baseball_swing',
                            'golf_swing', 'push_ups', 'tennis_forehand', 'bench_press', 'jumping_jacks', 'sit_ups',
                            'tennis_serve', 'bowling', 'jump_rope', 'squats']
        self.selected_actions = ["tennis_serve", "tennis_forehand", "baseball_pitch", "baseball_swing", 'jumping_jacks',
                           'golf_swing']
        self.c = 2.
        # keypoints
        # 1. head
        # 2. left_shoulder
        # 3. right_shoulder
        # 4. left_elbow
        # 5. right_elbow
        # 6. left_wrist
        # 7. right_wrist
        # 8. left_hip
        # 9. right_hip
        # 10. left_knee
        # 11. right_knee
        # 12. left_ankle
        # 13. right_ankle

    def process(self):
        dir = 'processed/'
        make_dir(self.path + dir)
        make_dir(self.path + dir + 'train/')  # todo make proper train-test split
        make_dir(self.path + dir + 'test/')
        make_dir(self.path + 'tfrecords/')

        videos = glob.glob(self.path_data + '*')
        for video_idx, video in enumerate(videos):

            # get meta data
            metadata_dir = video.replace('frames/', 'labels/') + '.mat'
            f = sio.loadmat(metadata_dir)
            action = f['action'][0]
            if not action in self.selected_actions:
                print('action {} not selected'.format(action))
                continue
            # pose = f['pose'][0]
            traintest = f['train'][0][0]
            if traintest == 1:
                traintest = 'train/'
                save_dir = dir + traintest
            elif traintest == -1:
                traintest = 'test/'
                save_dir = dir + traintest
            bbox = f['bbox']
            x = f['x']
            y = f['y']
            # dim = f['dimensions'][0]
            visibility = f['visibility']

            def get_frame(frame_idx):
                kp_x = x[frame_idx]
                kp_y = y[frame_idx]
                mask = visibility[frame_idx]
                image = cv2.imread(frame)
                bb = bbox[frame_idx]
                bb_w = int(bb[2] - bb[0])
                bb_h = int(bb[3] - bb[1])
                if not (bb_w > 0 and bb_h > 0):
                    print('bbox_w {}, bbox_h {} in frame {}, video {}, continue..'.format(bb_w, bb_h, frame_idx,
                                                                                          video_idx))
                    return False
                center = [int((bb[2] + bb[0]) / 2), int((bb[3] + bb[1]) / 2)]  # x, y
                kp = (kp_x, kp_y)
                max_bbox = np.max([bb_w, bb_h])
                return image, kp, mask, max_bbox, center

            video_path = video.replace('frames/', save_dir)

            if not make_dir(video_path):
                print('video {} already done, continue..'.format(video_idx))
                continue

            list_imgpaths = []
            list_keypoints = []
            list_masks = []

            frames = sorted(glob.glob(video + '/*'))
            for frame_idx, frame in enumerate(frames):

                if not get_frame(frame_idx):
                    continue
                image, kp, mask, max_bbox, center = get_frame(frame_idx)

                # pad
                pad = (self.res, self.res)
                image, kp, center = pad_img(image, pad, kp, center)

                # crop around center
                crop = (int(max_bbox * self.c), int(max_bbox * self.c))
                image, kp, center = crop_img(image, crop, kp, center)

                # resize image
                resize_img(image, self.shape, kp, center)

                # save image, make lists
                image_path = frame.replace('frames/', save_dir)
                save_img(image, image_path, self.shape)
                list_imgpaths.append(image_path)

                kp_x, kp_y = kp
                kp = np.concatenate([kp_x, kp_y], axis=0)
                assert 0 <= kp.all() < self.res, 'kp not in image'
                list_keypoints.append(kp)

                mask = mask.astype(np.float32)
                list_masks.append(mask)

            to_tfrecords(root_path=self.path, video_name=action + "_" + str(video_idx + 1).zfill(4), video_idx=video_idx,
                         list_imgpaths=list_imgpaths, list_keypoints=list_keypoints, list_masks=list_masks)


path = '../../../../myroot/penn/'
assert os.path.exists(path)
penn = PennAction(path_to_dataset=path)
penn.process()
