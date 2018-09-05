import glob
import cv2
import numpy as np
import os
import timeit
from ops_general import to_tfrecords, make_dir, kp_to_txt
from ops_image import save_img


class Dataset(object):
    def __init__(self, path_to_dataset, res, make_trainset=True, from_dir='raw', frame_rate=1):
        assert os.path.exists(path_to_dataset)
        self.path = path_to_dataset
        self.res = res
        self.shape = (res, res, 3)
        self.make_trainset = make_trainset
        self.from_dir = from_dir
        self.n_kp = 0
        self.all_actions = []
        self.selected_actions = []
        self.frame_rate = frame_rate
        self.video_key = lambda x: x
        self.image_key = lambda x: x

    def is_testset(self, video_path):
        return not self.make_trainset  # always make dataset by default

    def exclude_video(self, video_path):
        exclude = False
        if self.make_trainset:
            if self.is_testset(video_path):
                exclude = True
        else:
            if not self.is_testset(video_path):
                exclude = True
        return exclude

    def get_frame(self, image_idx, image_path, video_path, video_idx):
        return cv2.imread(image_path)

    def get_action(self, video_path):
        action = None
        if self.all_actions:
            action = video_path.split('/').pop(-2)  # assumes structure: raw/action/video_path
        return action

    def process(self, from_dir=None, to_dir='processed'):
        if from_dir is None:
            from_dir = self.from_dir
        if self.all_actions:
            for action in self.selected_actions:
                assert action in self.all_actions
                from_action_dir = from_dir + '/' + action
                self.process_vid(from_dir=from_action_dir, to_dir=to_dir, tfname='')  # tfname =  action + '_'
        else:
            self.process_vid(from_dir=from_dir, to_dir=to_dir, tfname='')

    def process_vid(self, from_dir=None, to_dir='processed', tfname=''):
        if from_dir is None:
            from_dir = self.from_dir

        make_dir(self.path + to_dir + '/')
        video_paths = sorted(glob.glob(self.path + from_dir + '/*'), key=self.video_key)
        for video_idx, video_path in enumerate(video_paths):

            img_paths = sorted(glob.glob(video_path + '/*'), key=self.image_key)  # todo check if sorted necessary
            save_path = video_path.replace(from_dir, to_dir)
            if self.all_actions:  # insert action: does not work
                action = self.get_action(video_path)
                split = save_path.split('/')
                split[-1] = action + '_' + split[-1]
                save_path = '/'.join(split)
            if self.exclude_video(video_path):
                continue

            if not make_dir(save_path):
                print('video {} done already, skipping..'.format(video_idx))
                continue

            list_imgpaths = []
            list_keypoints = []
            list_masks = []
            img_paths = img_paths[::10]

            for image_idx, image_path in enumerate(img_paths):

                if self.get_frame(image_idx, image_path, video_path, video_idx) is None:
                    continue
                frame = self.get_frame(image_idx, image_path, video_path, video_idx)
                if self.n_kp == 0 and not self.all_actions:  # e.g. Dogs
                    image = self.process_img(frame)
                    kp = None
                    kp_mask = None
                else:
                    image, kp, kp_mask, max_bbox, center = self.process_img(frame)

                action = self.get_action(video_path)

                # save image, make lists
                image_path = image_path.replace(from_dir, to_dir)
                if self.all_actions:  # insert action: does not work
                    action = self.get_action(video_path)
                    split = image_path.split('/')
                    split[-2] = action + '_' + split[-2]
                    image_path = '/'.join(split)
                save_img(image, image_path, self.shape)
                list_imgpaths.append(image_path)

                if self.n_kp != 0:
                    assert kp is not None
                    kp_x, kp_y = kp
                    kp = np.concatenate([kp_x, kp_y], axis=0)
                    assert 0 <= kp.all() < self.res, 'kp not in image'
                    list_keypoints.append(kp)
                    if kp_mask is None:  # if all kp visible all time
                        list_masks = None
                    else:
                        kp_mask = kp_mask.astype(np.float32)  # todo mask out kp to zero
                        list_masks.append(kp_mask)
                    kp_to_txt(self.path, self.n_kp, kp, kp_mask, self.make_trainset, image_idx, image_path)
                else:
                    list_keypoints = None
                    list_masks = None

                if action is not None:
                    tfname = action + '_'  # todo make same for to_dir: in order to save action name in

                # save image
                image_path = image_path.replace(from_dir, to_dir)
                save_img(image, image_path, self.shape)
                list_imgpaths.append(image_path)

            to_tfrecords(self.path, tfname + str(video_idx + 1).zfill(4), video_idx,
                         list_imgpaths, list_keypoints=list_keypoints, list_masks=list_masks)

    def process_img(self, frame):
        return NotImplementedError

