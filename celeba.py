import numpy as np
import glob
import os
from ops import to_tfrecords, make_dir


class CelebA(object):
    def __init__(self, path_to_dataset):
        self.path = path_to_dataset
        self.n_kp = 5
        self.keypoints = self.get_kp()  # comment out for speed
        self.bboxes = self.get_bb()
        # self.n_images

    def make_videos(self, path_to_data):
        """
        Create short sequences (videos) of the 200.000 images
        :return:
        """
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

            to_tfrecords(video_name=0, video_idx=0, list_imgpaths=list_imgpaths, list_keypoints=list_keypoints, list_masks=None)


path_to_dataset = '../../../../myroot/celeba/'
assert os.path.exists(path_to_dataset)
celeb = CelebA(path_to_dataset=path_to_dataset)
# celeb.make_videos(path_to_dataset + 'img_align_celeba/')
celeb.make_mafl(mode='testing')  # or 'training'
# celeb.make_videos(path_to_dataset + 'mafl/testing/')
