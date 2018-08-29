import cv2
import numpy as np
import glob
import os
from scipy.misc import imresize
from ops import make_dir, to_tfrecords, read_table, kp_to_txt
from ops_img import save_img, resize_img, pad_img, crop_img, invert_img, double_margin, get_bb


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
        self.res = 720
        self.shape = (self.res, self.res, 3)

    def is_test_set(self, img_idx):
        return bool(self.train_test[img_idx - 1][1])

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

    def process(self, save_in_dir='processed/',
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
                    if int(bad_cat[0]) == cat_idx+1:  # is it a bad category?
                        print('skip category {}, because {}'.format(cat_name, bad_cat[1]))
                        bad_category = True
                if bad_category:
                    continue

            if not make_dir(cat_dir):
                print('warning: video {}, {} exists, skipping conversion..'.format(cat_idx, cat_name))
                continue

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
                # if img is None:
                #     print('warning: image does not exist for image {}'.format(i))
                #     continue

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

                bbox = self.get_bbox(i)
                # if bbox is None:
                #     print('warning: bbox does not exist for image {}'.format(i))
                #     continue
                kp, kp_visible = self.get_kp(i)

                # select only images with visible kp:
                if visible_thresh is not None:
                    visible_ratio = np.mean(kp_visible)
                    if visible_ratio < visible_thresh:
                        print('skip image {}, because kp visible ratio < {}'.format(i, visible_ratio.round(2)))
                        continue

                # exclude images with bbox too close to edges
                bb_x, bb_y, bb_w, bb_h = bbox
                max_bbox = np.max([bb_h, bb_w])  # parameter for size of object in image -> later used for zooming
                (h, w, _) = img.shape
                center = [int(bb_y + bb_h / 2), int(bb_x + bb_w / 2)]  # center of bbox
                min_bb_ratio = 1. + bb_margin * 2  # margin to both sides, need complete bird in image
                too_big, y_min, y_max, x_min, x_max = get_bb(center, min_bb_ratio, bb_w, bb_h, w, h)

                if too_big and exclude_big:
                    print('skip image {}, because bird (bbox) too big in image'.format(i))
                    continue
                real_bb_ratio = min(w/bb_w, h/bb_h)
                assert real_bb_ratio >= min_bb_ratio

                # padding sides: mirror edges to both sides
                img_length = max(w, h)

                img, kp, center = double_margin(img, new_margin=0.5, c=center, kp=kp, ratio=real_bb_ratio, l=img_length, bb_h=bb_h, bb_w=bb_w)
                img, kp, center = double_margin(img, new_margin=1., c=center, kp=kp, ratio=real_bb_ratio, l=img_length, bb_h=bb_h, bb_w=bb_w)

                # invert image to have same parity (birds looking in one direction)
                if align_parity:
                    left_eye = kp_visible[6]
                    right_eye = kp_visible[10]
                    if left_eye and right_eye:  # exclude frontal view
                        print('skip image {}, because both eyes visible'.format(i))
                        continue
                    if not left_eye and right_eye:
                        img, kp = invert_img(img, kp)
                    else:
                        print('left_eye {}, right_eye {}'.format(left_eye, right_eye))
                        continue

                # resize image to intended final resolution
                img, kp, _ = resize_img(img, self.shape, kp=kp)
                kp_x, kp_y = kp
                kp_x = [kp * kp_visible[k] for k, kp in enumerate(kp_x)]  # mask out hidden kp: set to zero
                kp_y = [kp * kp_visible[k] for k, kp in enumerate(kp_y)]
                # kp = [kp_x, kp_y]

                img = img.astype(int)
                kp = np.concatenate([kp_y, kp_x], axis=0)
                kp = kp.astype(np.float32)

                # save image (for visual debugging)
                h_out, w_out, _ = self.shape
                assert kp.dtype == np.float32, print('kp dtype: {}'.format(kp.dtype))
                assert max(kp_x) <= w_out
                if max(kp_y) >= h_out:
                    print('warning: max(kp_y): {} > h_out: {}'.format(max(kp_y), h_out))
                assert min(kp) >= 0.
                assert kp_visible.dtype == np.float32

                frame_path = image_path.replace('images/', save_in_dir)
                save_img(img, frame_path, self.shape)

                # add relevant info to lists (for tfrecords)
                list_imgpaths.append(frame_path)
                list_max_bbox.append(max_bbox)
                list_kp.append(kp)
                list_masks.append(kp_visible)

                kp_to_txt(self.path, self.n_kp, kp, kp_visible, make_trainset, i+1, frame_path)

            print('video {}'.format(cat_name))
            if make_tfrecords:
                to_tfrecords(self.path, cat_name, cat_idx, list_imgpaths, list_kp, list_masks)  # save in tfrecords


path_to_root = '/Users/leonardbereska/myroot/'
path = path_to_root + 'birds/birds/'
# path_to_harddrive = '/Volumes/Uni/human_backgr/'
assert os.path.exists(path)
# path = path_to_harddrive + ID + '/'

mybirds = Birds(path_to_dataset=path)

mybirds.process(save_in_dir='processed/',
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

