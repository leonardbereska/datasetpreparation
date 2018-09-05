import glob
import cv2
import numpy as np
import os
import scipy.io as sio
import h5py
import timeit


from dataset_class import Dataset
from ops_image import pad_img, crop_img, resize_img, pad_crop_resize, double_margin, get_bb, invert_img
from ops_general import read_table, show_kp, make_dir, to_tfrecords


class DogRun(Dataset):
    def __init__(self, path_to_dataset, res, make_trainset=True):
        super(DogRun, self).__init__(path_to_dataset, res, make_trainset)
        self.n_kp = 0

    def process_img(self, image):
        h, w, _ = image.shape

        crop = [max(w, h), max(w, h)]

        image, _, _ = pad_crop_resize(image, crop, self.shape)

        return image


class Jockel(Dataset):
    def __init__(self, path_to_dataset, res, make_trainset=True):
        super(Jockel, self).__init__(path_to_dataset, res, make_trainset)
        self.n_kp = 0

    def process_img(self, image):
        h, w, _ = image.shape

        crop = [min(w, h), min(w, h)]

        image, _, _ = pad_crop_resize(image, crop, self.shape)

        return image


class Olympic(Dataset):
    def __init__(self, path_to_dataset, res, make_trainset, from_dir, selected_actions):
        super(Olympic, self).__init__(path_to_dataset, res, make_trainset, from_dir)
        self.n_kp = 0
        self.make_train_test_lists()
        self.selected_actions = selected_actions
        self.all_actions = ['basketball_layup', 'bowling', 'clean_and_jerk', 'discus_throw', 'diving_platform_10m',
                            'diving_springboard_3m', 'hammer_throw', 'high_jump', 'javelin_throw', 'long_jump',
                            'pole_vault', 'shot_put', 'snatch', 'tennis_serve', 'triple_jump', 'vault']

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

    def process_img(self, image):
        image = resize_img(image, self.shape)

        return image

    def is_testset(self, video_path):
        video_id = video_path.split('/').pop()
        if video_id in self.test_list:
            test = True
        elif video_id in self.train_list:
            test = False
        else:
            raise NotImplementedError
        return test


class PennAction(Dataset):
    def __init__(self, path_to_dataset, res, make_trainset, from_dir, selected_actions):
        super(PennAction, self).__init__(path_to_dataset, res, make_trainset, from_dir)
        self.n_kp = 13
        self.all_actions = ['baseball_pitch', 'clean_and_jerk', 'pull_ups', 'strumming_guitar', 'baseball_swing',
                            'golf_swing', 'push_ups', 'tennis_forehand', 'bench_press', 'jumping_jacks', 'sit_ups',
                            'tennis_serve', 'bowling', 'jump_rope', 'squats']
        self.selected_actions = selected_actions
        self.bbox_factor = 2.

    def is_testset(self, f):
        split = f['train'][0][0]
        if split == 1:
            is_testset = False
        elif split == -1:
            is_testset = True
        else:
            raise NotImplementedError
        return is_testset

    def exclude_video(self, video_path):
        metadata_dir = video_path.replace(self.from_dir, 'labels') + '.mat'
        f = sio.loadmat(metadata_dir)
        action = f['action'][0]
        exclude = False
        if not action in self.selected_actions:
            print('action {} not selected'.format(action))
            exclude = True
        if self.make_trainset:
            if self.is_testset(f):
                exclude = True
        else:
            if not self.is_testset(f):
                exclude = True
        return exclude

    def get_frame(self, frame_idx, frame_path, video_path, video_idx):
        # get meta data
        metadata_dir = video_path.replace(self.from_dir, 'labels') + '.mat'
        f = sio.loadmat(metadata_dir)
        action = f['action'][0]
        # pose = f['pose'][0]
        bbox = f['bbox']
        x = f['x']
        y = f['y']
        # dim = f['dimensions'][0]
        visibility = f['visibility']
        kp_x = x[frame_idx]
        kp_y = y[frame_idx]
        mask = visibility[frame_idx]
        image = cv2.imread(frame_path)
        bb = bbox[frame_idx]
        bb_w = int(bb[2] - bb[0])
        bb_h = int(bb[3] - bb[1])
        if not (bb_w > 0 and bb_h > 0):
            print('bbox_w {}, bbox_h {} in frame {}, video {}, continue..'.format(bb_w, bb_h, frame_idx, video_path))
            return None
        center = [int((bb[2] + bb[0]) / 2), int((bb[3] + bb[1]) / 2)]  # x, y
        kp = (kp_x, kp_y)
        max_bbox = np.max([bb_w, bb_h])
        frame = image, kp, mask, max_bbox, center, action
        return frame

    def process_img(self, frame):
        image, kp, mask, max_bbox, center, action = frame
        # pad
        pad = (self.res, self.res)
        image, kp, center = pad_img(image, pad, kp, center)

        # crop around center
        crop = (int(max_bbox * self.bbox_factor), int(max_bbox * self.bbox_factor))
        image, kp, center = crop_img(image, crop, kp=kp, center=center)

        # resize image
        image, kp, center = resize_img(image, to_shape=self.shape, kp=kp, center=center)

        return image, kp, mask, max_bbox, center


class Human(Dataset):
    def __init__(self, path_to_dataset, res, selected_actions, use_mask, frame_rate):
        super(Human, self).__init__(path_to_dataset, res, frame_rate=frame_rate)
        self.n_kp = 32
        self.all_actions = ['directions', 'discussion', 'posing', 'waiting', 'greeting', 'walking', 'sitting', 'photo']
        self.selected_actions = selected_actions
        self.use_mask = use_mask

    def exclude_video(self, video_path):
        action = self.get_action(video_path)
        exclude = False
        if action not in self.selected_actions:
            print('action {} not selected'.format(action))
            exclude = True
        return exclude

    def get_kp(self, activity, video_idx, frame_idx):

        Activity = (activity[:1]).upper() + activity[1:]  # first letter upper case, e.g. 'Sitting '
        keypoint_path = self.path + 'MyPoseFeatures/D2_Positions/' + Activity + '*.mat'  # select only matlab files
        video_paths = glob.glob(keypoint_path)
        pose_mat = video_paths[video_idx]  # select video  # todo is this really the same video as the index indicates?
        assert os.path.exists(pose_mat), 'matlab file for keypoints does not exist'
        f = sio.loadmat(pose_mat)
        video_frames = f['video_frames']
        original_idx = frame_idx * self.frame_rate
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

    def get_mask(self, activity, video_idx, frame_idx):
        """
        Get background mask for specific activity, video and frame
        :param activity: string of activity from self.activities
        :param video_idx: int for video index
        :param frame_idx: int for frame index
        :return:
        """
        vid_path = self.path + 'MySegmentsMat/ground_truth_bs/'

        activity_upper = (activity[:1]).upper() + activity[1:]  # first letter upper case
        vid_path += activity_upper
        video_paths = glob.glob(vid_path + '*')  # can use other format also e.g. .avi
        vid_path = video_paths[video_idx]  # choose specific video

        with h5py.File(vid_path, 'r') as f:
            # print(f.keys())  # have to know which keys to select
            all_frames = np.array(f['Masks'])
            try:
                mask = np.array(f[all_frames[frame_idx * self.frame_rate][0]])  # choose specific frame
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

        vid_path = self.path + 'MySegmentsMat/ground_truth_bb/'

        activity_upper = (activity[:1]).upper() + activity[1:]  # first letter upper case
        vid_path += activity_upper
        video_paths = glob.glob(vid_path + '*')  # can use other format also e.g. .avi
        vid_path = video_paths[video_idx]  # choose specific video

        with h5py.File(vid_path, 'r') as f:
            # print(f.keys())  # have to know which keys to select
            all_frames = np.array(f['Masks'])
            try:
                bbox = np.array(f[all_frames[frame_idx * self.frame_rate][0]])  # choose specific frame
                bbox = bbox.transpose()
            except:
                bbox = None
        return bbox

    def get_frame(self, frame_idx, frame_path, video_path, video_idx):
        action = self.get_action(video_path)
        if self.use_mask:
            bg_mask = self.get_mask(action, video_idx, frame_idx)
        else:
            bg_mask = None
        # img = self.get_img_raw(activity, video_idx, frame_idx)
        image = cv2.imread(frame_path)  # for speed
        bbox = self.get_bbox(action, video_idx, frame_idx)
        kp = self.get_kp(action, video_idx, frame_idx)

        # mask and video not exactly aligned sometimes for last frame, -> just ignore those
        if image is None:
            print('warning: image does not exist for {}, video {}, frame {}'.format(action, video_idx, frame_idx))
            return None
        if self.use_mask and bg_mask is None:
            print('warning: mask does not exist for {}, video {}, frame {}'.format(action, video_idx, frame_idx))
            return None
        if bbox is None:
            print('warning: bbox does not exist for {}, video {}, frame {}'.format(action, video_idx, frame_idx))
            return None
        if kp is None:
            print('warning: keypoints do not exist for {}, video {}, frame {}'.format(action, video_idx, frame_idx))
            return None

        # crop image
        i_big = 1000
        image = image[0:i_big, 0:i_big, :]  # make sure all images same size
        if self.use_mask:
            bg_mask = bg_mask[0:i_big, 0:i_big]
            assert (pix == 0 or pix == 1 for pix in bg_mask)  # mask has values 0 or 1
        bbox = bbox[0:i_big, 0:i_big]
        bbox_idx = np.nonzero(bbox)
        bb_ymin = min(bbox_idx[0])
        bb_xmin = min(bbox_idx[1])
        bb_ymax = max(bbox_idx[0])
        bb_xmax = max(bbox_idx[1])
        assert (pix == 0 or pix == 1 for pix in bbox)  # bbox has values 0 or 1

        kp_x = kp[0]
        kp_y = kp[1]
        kp = [kp_x, kp_y]

        center = [int((bb_xmin + bb_xmax) / 2), int((bb_ymin + bb_ymax) / 2)]  # image center
        max_bbox = np.max((bb_xmax, bb_ymax))

        frame = image, kp, bg_mask, max_bbox, center, action
        return frame

    def process_img(self, frame):
        img, kp, bg_mask, max_bbox, center, action = frame

        if self.use_mask:
            # crop = (max_bbox, max_bbox)
            # img, kp, center = crop_img(img, crop, center=center, kp=kp)
            # bg_mask, _, _ = crop_img(bg_mask, crop, center=center, kp=None)

            bckgr = np.empty(img.shape)
            bckgr.fill(255)  # 255 for white
            for i in range(3):
                img[:, :, i] = bg_mask * img[:, :, i] + (1 - bg_mask) * bckgr[:, :, i]  # mask out image background

            img, kp, center = pad_img(img, kp=kp, center=center, mode='constant')

            c = int(max_bbox)
            crop = (c, c)
            img, kp, center = crop_img(img, crop=crop, center=center, kp=kp)

            img, kp, center = resize_img(img, to_shape=self.shape, center=center, kp=kp)

        else:
            crop = [max_bbox, max_bbox]
            img, kp, center = pad_crop_resize(img, crop, to_shape=self.shape, kp=kp, center=center)

        mask = None
        return img, kp, mask, max_bbox, center


class Birds(Dataset):
    def __init__(self, path_to_dataset, res, make_trainset, from_dir, only_simple_classes=True, visible_thresh=None,
                 align_parity=True, bb_margin=0.125, exclude_big=True):
        super(Birds, self).__init__(path_to_dataset, res, make_trainset, from_dir)
        self.classes = read_table(self.path + 'classes.txt', type_float=False)
        self.train_test = read_table(self.path + 'train_test_split.txt')
        self.bboxes = read_table(self.path + 'bounding_boxes.txt')
        self.kp = read_table(self.path + 'parts/part_locs.txt')
        self.img_class = read_table(self.path + 'image_class_labels.txt')
        self.img_path = read_table(self.path + 'images.txt', type_float=False)
        self.bad_categories = read_table(self.path + 'classes_difficult.txt',
                                         type_float=False)  # sb: sea bird, bb: big bird, fb: flying bird, cm: camouflage
        self.n_kp = 15
        self.only_simple_classes = only_simple_classes
        self.visible_thresh = visible_thresh
        self.align_parity = align_parity
        self.bb_margin = bb_margin
        self.exclude_big = exclude_big

    def is_testset(self, img_idx):
        return bool(self.train_test[img_idx - 1][1])

    def exclude_video(self, video_path):
        video_name = video_path.split('/').pop()
        cat_idx = int(video_name.split('.').pop(0))
        exclude = False
        if self.only_simple_classes:  # select only good categories:
            bad_category = False
            for bad_cat in self.bad_categories:
                if int(bad_cat[0]) == cat_idx:  # is it a bad category?
                    print('skip category {}, because {}'.format(video_name, bad_cat[1]))
                    bad_category = True
            if bad_category:
                exclude = True
        return exclude

    def get_bbox(self, i):
        bbox = self.bboxes[i][1:5]
        bbox = [float(bb) for bb in bbox]
        bbox = [int(bb) for bb in bbox]
        return bbox

    def get_kp(self, i):
        # get keypoints
        kp = self.kp[i * 15:(i + 1) * 15]
        kp_xy = [k[2:4] for k in kp]  # x, y, visible; dtype in strings
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

        kp = [kp_x, kp_y]
        return kp, kp_visible

    def get_frame(self, frame_idx, frame_path, video_path, video_idx):

        image = cv2.imread(frame_path)

        # find frame_idx corresponding to frame_path
        split = frame_path.split('/')
        img = split.pop()
        vid = split.pop()
        rel_path = '/'.join([vid, img])
        k = None
        for e in self.img_path:
            if e[1] == rel_path:
                k = int(e[0])
        if k is None:
            print('img_path not found')
            raise NotImplementedError
        frame_idx = k - 1

        # train-test split
        img_in_trainset = bool(int(self.train_test[frame_idx][1]))
        if self.make_trainset:
            if not img_in_trainset:
                # print('exclude test set')
                return None
        elif not self.make_trainset:
            if img_in_trainset:
                # print('exclude train set')
                return None

        bbox = self.get_bbox(frame_idx)
        bb_x, bb_y, bb_w, bb_h = bbox
        max_bbox = np.max([bb_h, bb_w])  # parameter for size of object in image -> later used for zooming
        center = [int(bb_y + bb_h / 2), int(bb_x + bb_w / 2)]  # center of bbox
        (h, w, _) = image.shape
        min_bb_ratio = 1. + self.bb_margin * 2  # margin to both sides, need complete bird in image
        too_big, y_min, y_max, x_min, x_max = get_bb(center, min_bb_ratio, bb_w, bb_h, w, h)

        # exclude images with bbox too close to edges
        if too_big and self.exclude_big:
            print('skip image {}, because bird (bbox) too big in image'.format(frame_idx))
            return None
        real_bb_ratio = min(w / bb_w, h / bb_h)
        assert real_bb_ratio >= min_bb_ratio

        kp, kp_mask = self.get_kp(frame_idx)
        kp_x, kp_y = kp
        kp = kp_y, kp_x
        show_kp(image, kp)
        # select only images with visible kp:
        if self.visible_thresh is not None:
            visible_ratio = np.mean(kp_mask)
            if visible_ratio < self.visible_thresh:
                print('skip image {}, because kp visible ratio < {}'.format(frame_idx, visible_ratio.round(2)))
                return None

        # invert image to have same parity (birds looking in one direction)
        if self.align_parity:
            left_eye = kp_mask[6]
            right_eye = kp_mask[10]
            if left_eye and right_eye:  # exclude frontal view
                print('skip image {}, because both eyes visible'.format(frame_idx))
                return None

        bbox_wh = (bb_w, bb_h)
        frame = image, kp, kp_mask, bbox_wh, center
        return frame

    def process_img(self, frame):
        image, kp, kp_mask, bbox_wh, center = frame

        bb_w, bb_h = bbox_wh
        max_bbox = np.max(bbox_wh)
        (h, w, _) = image.shape
        real_bb_ratio = min(w / bb_w, h / bb_h)
        # padding sides: mirror edges to both sides
        img_length = max(w, h)
        image, kp, center = double_margin(image, new_margin=0.5, c=center, kp=kp, ratio=real_bb_ratio, l=img_length,
                                          bb_h=bb_h, bb_w=bb_w)
        image, kp, center = double_margin(image, new_margin=1., c=center, kp=kp, ratio=real_bb_ratio, l=img_length,
                                          bb_h=bb_h, bb_w=bb_w)

        if self.align_parity:
            left_eye = kp_mask[6]
            right_eye = kp_mask[10]
            if not left_eye and right_eye:
                image, kp = invert_img(image, kp)

        # resize image to intended final resolution
        image, kp, _ = resize_img(image, self.shape, kp=kp)
        # kp_x, kp_y = kp
        #
        # kp_x = [kp * kp_mask[k] for k, kp in enumerate(kp_x)]  # todo mask out hidden kp: set to zero
        # kp_y = [kp * kp_mask[k] for k, kp in enumerate(kp_y)]
        # kp = [kp_x, kp_y]

        return image, kp, kp_mask, max_bbox, center


class Cats(Dataset):
    def __init__(self, path_to_dataset, res, make_trainset, from_dir):
        super(Cats, self).__init__(path_to_dataset, res, make_trainset, from_dir)
        self.n_kp = 7  # originally 9

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

    def get_frame(self, frame_idx, frame_path, video_path, video_idx):
        image = cv2.imread(frame_path)

        # get kp
        if not self.make_trainset:
            kp_paths = glob.glob(video_path + '/*.jpg.cat')
            # kp_paths = glob.glob(video_path + '/*.jpg')
            kp_path = kp_paths[frame_idx]
        else:
            kp_path = frame_path.replace(self.from_dir, 'frames/') + '.cat'
            # kp_path = frame_path.replace(self.from_dir, 'frames/')
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
        kp = np.array(kp)  # x, y, x, y ...

        # get bbox (from kp)
        kp_x = kp[::2]
        kp_y = kp[1::2]
        kp_bb_x = np.delete(kp_x, [4, 7], axis=0)
        kp_bb_y = np.delete(kp_y, [4, 7], axis=0)
        bb = [min(kp_bb_x), min(kp_bb_y), max(kp_bb_x), max(kp_bb_y)]
        bb_w = int(bb[2] - bb[0]) + 1
        bb_h = int(bb[3] - bb[1]) + 1
        center = [int((bb[2] + bb[0]) / 2), int((bb[3] + bb[1]) / 2)]  # x, y
        kp_x = kp_bb_x.astype(float)
        kp_y = kp_bb_y.astype(float)

        max_bbox = np.max([bb_w, bb_h])
        kp = [kp_x, kp_y]
        frame = image, kp, None, max_bbox, center
        return frame

    def process_img(self, frame):
        image, kp, kp_mask, max_bbox, center = frame

        crop = (int(max_bbox * 2), int(max_bbox * 2))

        image, _, _ = pad_crop_resize(image, crop, self.shape, kp=kp, center=center)

        return image, kp, None, max_bbox, center


class CelebA(Dataset):
    def __init__(self, path_to_dataset, res, make_trainset, from_dir):
        super(CelebA, self).__init__(path_to_dataset, res, make_trainset, from_dir)
        self.n_kp = 5
        self.keypoints = self.get_kp()  # comment out for speed
        self.bboxes = self.get_bb()

    def process(self, from_dir=None, to_dir='processed'):
        """
        Create short sequences (videos) of the 200.000 images
        :return:
        """
        path_to_data = self.path + 'img_align_celeba/'
        length = 1000  # length of one video
        n_videos = int(self.n_images / length) + 1
        all_images = glob.glob(path_to_data + '*')

        for video_idx in range(n_videos):
            from_ = video_idx * length
            to_ = (video_idx + 1) * length
            try:
                list_imgpaths = all_images[from_:to_]  # todo exclude mafl
            except IndexError:
                to_ = -1
                list_imgpaths = all_images[from_:to_]
                length = len(list_imgpaths)

            list_keypoints = self.keypoints[from_:to_]

            to_tfrecords(root_path=self.path, video_name=str(video_idx).zfill(2), video_idx=video_idx,
                         list_imgpaths=list_imgpaths, list_keypoints=list_keypoints, list_masks=None)
            print('{}/{}'.format(video_idx, n_videos))

    def get_kp(self):
        """
        Reads txt file and returns array of all kp coordinates for all images
        :return: np array with shape: (n_images, n_kp, coordinates dimensions (x,y))
        """
        with open(self.path + "list_landmarks_align_celeba.txt") as file:
            data = file.read()
            inlist = data.split('\n')
            inlist.pop(0)  # header
            inlist.pop(0)  # metadata
            inlist.pop()  # garbage: "
            self.n_images = len(inlist)
            all_kp = []
            for image_idx in range(len(inlist)):

                kp_list = inlist[image_idx]
                kp_list = kp_list.split(' ')
                kp = []
                for k in kp_list:
                    try:
                        kp.append(int(k))
                    except ValueError:  # e.g. for '' or '000002.jpg'
                        pass
                assert len(kp) == self.n_kp * 2
                kp = np.array(kp)  # .reshape((self.n_kp, 2)) <- tf records wants 1-dim np.array
                all_kp.append(kp)
            kp = np.array(all_kp)
            return kp

    def get_bb(self):
        """
        Reads txt file and returns array of all bb coordinates for all images
        :return: np array with shape: (n_images, 4)    x_1, y_1 width, height
        """
        with open(self.path + "list_bbox_celeba.txt") as file:
            data = file.read()
            inlist = data.split('\n')
            inlist.pop(0)  # header
            inlist.pop(0)  # metadata
            inlist.pop()  # garbage: "
            assert len(inlist) == self.n_images
            all_bb = []
            for image_idx in range(len(inlist)):

                bb_list = inlist[image_idx]
                bb_list = bb_list.split(' ')
                bb = []
                for k in bb_list:
                    try:
                        bb.append(int(k))
                    except ValueError:  # e.g. for '' or '000002.jpg'
                        pass
                assert len(bb) == 4  # x_1, y_1 width, height
                bb = np.array(bb)
                all_bb.append(bb)
            bb = np.array(all_bb)
            return bb

    def make_mafl(self, mode):
        mafl_path = self.path + 'mafl/'
        filename = mafl_path + mode + '.txt'
        with open(filename) as f:
            data = f.read()
            id_list = data.split('\n')
            save_dir = mafl_path + mode + '/'
            make_dir(save_dir)

            list_imgpaths = []
            list_keypoints = []

            for i, test_id in enumerate(id_list):
                image_path = self.path + 'img_align_celeba/' + test_id
                img_id = test_id.replace('.jpg', '')
                idx = int(img_id) - 1  # 0th element is 000001.jpg
                kp = self.keypoints[idx, :]
                print(img_id)

                list_imgpaths.append(image_path)
                list_keypoints.append(kp)

            to_tfrecords(root_path=self.path, video_name=0, video_idx=0, list_imgpaths=list_imgpaths,
                         list_keypoints=list_keypoints, list_masks=None)


class BBCPose(Dataset):
    def __init__(self, path_to_dataset, res, make_trainset, from_dir):
        super(BBCPose, self).__init__(path_to_dataset, res, make_trainset, from_dir)
        self.n_kp = 7
        # self.all_actions = ['baseball_pitch', 'clean_and_jerk', 'pull_ups', 'strumming_guitar', 'baseball_swing',
        #                     'golf_swing', 'push_ups', 'tennis_forehand', 'bench_press', 'jumping_jacks', 'sit_ups',
        #                     'tennis_serve', 'bowling', 'jump_rope', 'squats']
        # self.selected_actions = selected_actions
        self.bbox_factor = 2.
        self.video = self.update_video(0)
        self.train_frames = list(self.video['train_frames'][0].astype(int).astype(str))
        self.train_joints = self.video['train_joints']
        self.test_frames = list(self.video['test_frames'][0].astype(int).astype(str))
        self.test_joints = self.video['test_joints']
        self.video_key = lambda x: int(x.split('/').pop())
        self.image_key = lambda x: int(x.split('/').pop().split('.').pop(0))

    def is_testset(self, video_path):

        video_name = video_path.split('/').pop()
        video_idx = int(video_name) - 1
        self.update_video(video_idx)
        type = self.video['type'][0]
        # select train/test-set
        if type == 'test':
            is_test = True
        elif type == 'train' or type == 'val':
            is_test = False
        else:
            raise NotImplementedError
        source = self.video['source'][0]
        assert source == 'buehler11', 'video belongs to extended bbc pose, not to normal bbc pose'

        return is_test

    def update_video(self, video_idx):
        pose = self.path + 'bbcpose_extbbcpose_code_1.0/' + 'bbcpose.mat'
        f = sio.loadmat(pose)
        videos = f['bbcpose'][0]
        video = videos[video_idx]
        self.train_frames = list(video['train_frames'][0].astype(int).astype(str))
        self.train_joints = video['train_joints']
        self.test_frames = list(video['test_frames'][0].astype(int).astype(str))
        self.test_joints = video['test_joints']
        self.video = video
        return self.video

    def get_frame(self, frame_idx, frame_path, video_path, video_idx):

        img_name = frame_path.split('/').pop().split('.').pop(0)

        try:
            if self.make_trainset:

                idx = self.train_frames.index(img_name)
                kp_x = self.train_joints[0, :, idx]
                kp_y = self.train_joints[1, :, idx]
            else:
                idx = self.test_frames.index(img_name)
                kp_x = self.test_joints[0, :, idx]  # y, x
                kp_y = self.test_joints[1, :, idx]
        except:
            # print('frame {} not in list of frames with annotations'.format(frame_idx))
            return None  #

        image = cv2.imread(frame_path)

        bb = [np.min(kp_x), np.max(kp_x), np.min(kp_y), np.max(kp_y)]
        bb_w = int(bb[1] - bb[0])
        bb_h = int(bb[3] - bb[2])
        center = [int((bb[1] + bb[0]) / 2), int((bb[3] + bb[2]) / 2)]  # x, y
        kp = (kp_x, kp_y)
        max_bbox = np.max([bb_w, bb_h])
        frame = image, kp, None, max_bbox, center
        return frame

    def process_img(self, frame):
        image, kp, mask, max_bbox, center = frame
        # pad
        pad = (self.res, self.res)
        image, kp, center = pad_img(image, pad, kp, center)

        # crop around center
        crop = (int(max_bbox * self.bbox_factor), int(max_bbox * self.bbox_factor))
        image, kp, center = crop_img(image, crop, kp=kp, center=center)

        # resize image
        image, kp, center = resize_img(image, to_shape=self.shape, kp=kp, center=center)

        return image, kp, mask, max_bbox, center
