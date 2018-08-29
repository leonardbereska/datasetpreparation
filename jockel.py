import numpy as np
import glob
import cv2
import tensorflow as tf
import os
from scipy.misc import imresize
from ops import to_tfrecords, make_dir
from ops_img import save_img, pad_img, crop_img, resize_img, get_image_data


FRAME_RATE = 2
N_VID = 2  # maximum number of videos for each activity: 10^n_vid
N_IMG = 4  # maximum number of images for each video: 10^n_img


class OwnData(object):
    def __init__(self, path_to_dataset):
        self.path = path_to_dataset
        self.res = 360
        self.shape = (self.res, self.res, 3)

    def make_videos(self, orig_path='raw/'):
        """
        :return:
        """
        make_dir(self.path + 'tfrecords/')
        make_dir(self.path + 'processed/')

        video_paths = glob.glob(self.path + orig_path + '*')
        for video_idx, video in enumerate(video_paths):
            video_name = video.split('/').pop()  # e.g. afghan_hound.mp4
            video_name = video_name.replace('.MTS', '')  # e.g. afghan_hound
            img_paths = glob.glob(video + '/*.jpg')

            save_dir = 'processed/'
            save_path = video.replace(orig_path, save_dir)
            if os.path.exists(save_path):
                print('video {} done already, skipping..'.format(video_idx))
                continue
            make_dir(save_path)

            list_imgpaths = []

            for image_idx, image_path in enumerate(img_paths):

                image = cv2.imread(image_path)

                if not get_image_data:
                    print('skipping empty image {} in video {}'.format(image_idx, video_idx))
                    continue
                else:
                    h, w, center = get_image_data(image)

                # pad
                image, center = pad_img(image, center=center)

                # crop around center
                crop = [min(w, h), min(w, h)]
                image, _, center = crop_img(image, crop, center=center)

                # resize to bbox
                image, _, center = resize_img(image, self.shape, center=center)

                image_path = image_path.replace(orig_path, save_dir)
                save_img(image, image_path, self.shape)
                list_imgpaths.append(image_path)

            to_tfrecords(self.path, video_name=str(video_idx + 1).zfill(4), video_idx=video_idx,
                         list_imgpaths=list_imgpaths, list_keypoints=None, list_masks=None)

    # def videos_to_raw(self, from_dir='videos/', to_dir='raw/', bg0=False, start_id=0):
    #     """
    #     Takes a folder structure (subject, background, action) and extracts the frames into raw/
    #     name is s..-b..-a..
    #     """
    #     global N_IMG
    #     global N_VID
    #     global FRAME_RATE  # save only every nth frame
    #
    #     make_dir(self.path + to_dir)
    #
    #     subject_files = glob.glob(path + from_dir + '*')
    #     i_b = 1  # start at one (0 is for other dataset)
    #     for i_s, file_s in enumerate(subject_files):
    #         s = str.zfill(str(i_s + start_id), 3)
    #         backgr_files = glob.glob(file_s + '/*')
    #         for file_b in backgr_files:
    #             if file_b.split('/')[-1] == 'x':
    #                 b = 'xxx'
    #             else:
    #                 b = str.zfill(str(i_b), 3)
    #             i_b += 1
    #             videos = glob.glob(file_b + '/*')
    #             for i_v, file_v in enumerate(videos):
    #                 v = str(i_v)
    #                 if bg0:
    #                     b = '000'
    #                 video_name = 's' + s + '-b' + b + '-a' + v
    #                 save_path = self.path + to_dir + video_name + '/'
    #
    #
    #
    #                 if not os.path.exists(save_path):
    #                     os.mkdir(save_path)  # create video folder
    #                 else:
    #                     print('warning: {} folder exists, skipping conversion..'.format(i_v))
    #                     continue
    #
    #                 video_capture = cv2.VideoCapture(file_v)
    #
    #                 img_count = 0
    #                 img_i = 0
    #                 success = True
    #                 while success:
    #                     success, image = video_capture.read()
    #                     try:
    #                         if img_count % FRAME_RATE == 0:
    #                             img_name = save_path + str(img_i).zfill(N_IMG) + ".jpg"  # save images in video folder
    #
    #                             # de-interlacing: delete every second row and column
    #                             image = image[::2, ::2, :]
    #
    #                             cv2.imwrite(img_name, image)     # save frame as JPEG file
    #                             img_i += 1
    #                     except:
    #                         pass
    #                     img_count += 1
    #                 print('Finished video {}/{}: {}'.format(i_v, len(videos), video_name))

# step 1
# step 1.1: put videos to be renamed in "path/rename" (will then be stored in "path/exported") -> execute .rename_raw()
# step 1.2: put to videos in folder "path/exported" -> execute .convert_mp4_to_img()

# alternatively: if videos in folder structure:
# step 1.1: put videos in "path/videos" (folder structure: subject/background/action)

# step 2: convert videos to frames to be stored in folder "path/raw"
# step 3: make processed frames and tfrecords files, stored in "path/processed" and "path/tfrecords"


path = '../../../../myroot/bigjockel/'
assert os.path.exists(path)

own = OwnData(path_to_dataset=path)
# own.rename_raw(bg0=True, start_id=100)
# own.convert_mg4_to_img()

# own.videos_to_raw(from_dir='videos/', to_dir='raw/', bg0=False, start_id=0)  # TODO test if working as required
# own.make_videos()

# own.rename()
