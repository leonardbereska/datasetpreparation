import glob
import cv2
import os
from ops import to_tfrecords, make_dir, extract_frames
from ops_img import save_img, pad_img, crop_img, resize_img


class DogRun(object):
    def __init__(self, path_to_dataset):
        self.path = path_to_dataset
        self.res = 600
        self.shape = (self.res, self.res, 3)

    def make_videos(self, orig_path='raw/'):
        """
        :return:
        """
        make_dir(self.path + 'processed/')

        video_paths = glob.glob(self.path + orig_path + '*')
        for video_idx, video in enumerate(video_paths):

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
                try:
                    h, w, _ = image.shape
                except AttributeError:
                    print('skipping empty image {} in video {}'.format(image_idx, video_idx))
                    continue

                center = [int(w / 2), int(h / 2)]  # x, y

                # pad
                image, _, center = pad_img(image, center=center)

                # crop around center
                c = max(w, h)
                crop = (c, c)
                image, _, center = crop_img(image, crop, center)

                # resize to final shape
                image, _, center = resize_img(image, self.shape, center=center)

                image_path = image_path.replace(orig_path, save_dir)
                save_img(image, image_path, self.shape)
                list_imgpaths.append(image_path)

            to_tfrecords(self.path, str(video_idx).zfill(2), video_idx, list_imgpaths, list_keypoints=None, list_masks=None)


path = '../../../../myroot/'
assert os.path.exists(path)

dogs = DogRun(path_to_dataset=path)

# extract_frames(path, from_dir='exported', to_dir='raw', video_format='mov', img_format='jpg', frame_rate=2)

dogs.make_videos()

# for video in ['002', '003', '004', '005', '006', '007', '008', '009']:
# dogs.left_to_right('007')
