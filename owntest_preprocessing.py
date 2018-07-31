import numpy as np
import glob
import cv2
import tensorflow as tf
import os
from scipy.misc import imresize

FRAME_RATE = 2
N_VID = 2  # maximum number of videos for each activity: 10^n_vid
N_IMG = 4  # maximum number of images for each video: 10^n_img


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_float32(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))  # wtf only float


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class OwnData(object):
    def __init__(self, path_to_dataset):
        self.path = path_to_dataset
        self.img_size = [360, 360]  # todo find maximum in dataset

    def convert_mg4_to_img(self, to_dir='raw/'):
        """
        Converts video from mp4 to img frames
        """
        global N_IMG
        global N_VID
        global FRAME_RATE  # save only every nth frame

        make_dir(self.path + to_dir)

        video_paths = glob.glob(self.path + 'exported/*')  # can use other format also e.g. .avi

        for video_idx, video_path in enumerate(video_paths):
            video_name = video_path.split('/').pop()  # e.g. afghan_hound.mp4
            video_name = video_name.replace('.MOV', '')  # e.g. afghan_hound
            save_path = self.path + to_dir + video_name + '/'

            if not os.path.exists(save_path):
                os.mkdir(save_path)  # create video folder
            else:
                print('warning: {} folder for {} exists, skipping conversion..'.format(video_idx, activity))
                continue

            video_capture = cv2.VideoCapture(video_path)

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
            print('Finished video {}/{}: {}'.format(video_idx, len(video_paths), video_name))

    def make_videos(self, orig_path='raw/'):
        """
        :return:
        """
        make_dir(self.path + 'tfrecords/')
        make_dir(self.path + 'processed/')

        video_paths = glob.glob(self.path + orig_path + '*')
        for video_idx, video in enumerate(video_paths):
            video_name = video.split('/').pop()  # e.g. afghan_hound.mp4
            video_name = video_name.replace('.mov', '')  # e.g. afghan_hound
            img_paths = glob.glob(video + '/*.jpg')

            save_dir = 'processed/'
            save_path = video.replace(orig_path, save_dir)
            if os.path.exists(save_path):
                print('video {} done already, skipping..'.format(video_idx))
                continue
            make_dir(save_path)

            list_imgpaths = []
            list_keypoints = []
            list_max_bbox = []

            for image_idx, image_path in enumerate(img_paths):

                image = cv2.imread(image_path)
                try:
                    h, w, _ = image.shape
                except AttributeError:
                    print('skipping empty image {} in video {}'.format(image_idx, video_idx))
                    continue

                # resize to quadratic

                center = [int(w / 2), int(h / 2)]  # x, y

                # pad
                pad_x = 1000  # self.max_dim[0]
                pad_y = 1000  # self.max_dim[1]

                center[0] += pad_x
                center[1] += pad_y
                image = np.lib.pad(image, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), 'symmetric')

                # crop around center
                c = min(w, h)
                # c = 1080
                crop_w = c
                crop_h = c
                crop_x = int(center[0] - crop_w / 2)
                crop_y = int(center[1] - crop_h / 2)
                center[0] -= crop_x
                center[1] -= crop_y
                image = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

                # resize to bbox
                out_shape = (self.img_size[0], self.img_size[1], 3)
                image = imresize(image, out_shape)
                center[1] = center[1] / crop_h * out_shape[1]
                center[0] = center[0] / crop_w * out_shape[0]

                # # visualize
                # from matplotlib import pyplot as plt
                # i = 7
                # plt.imshow(image)
                # plt.scatter(kp_x[i], kp_y[i])
                # plt.scatter(center[0], center[1], c='r')  # 7, 4

                image_path = image_path.replace(orig_path, save_dir)
                dim_correct = (image.shape == (self.img_size[0], self.img_size[0], 3))
                assert dim_correct, '{}'.format(image.shape)  # must be rectangular and same size
                assert (image.dtype == np.uint8)
                cv2.imwrite(image_path, image)
                list_imgpaths.append(image_path)

                kp = np.array([0])
                list_keypoints.append(kp)

                max_bbox = int(min(w, h) / c * out_shape[0])  # smaller of image size scaled to out size
                list_max_bbox.append(max_bbox)

            self.to_tf_records(video_name, video_idx, list_imgpaths, list_keypoints, list_max_bbox)
        #     print('{}/{}'.format(video_idx, n_videos))
        # # except IndexError:
        # #     print('Finished conversion')

    def to_tf_records(self, video_name, video_idx, list_imgpaths, list_keypoints, list_max_bbox, save_dir='tfrecords/'):
        """
        Create tfrecords file from video
        :param video_name: int index of video
        :param list_imgpaths: all image paths in video
        :param list_keypoints: all keypoints as numpy array
        :param list_max_bbox: list of maxima of bounding box width/height
        :return:
        """
        save_path = self.path + save_dir
        make_dir(save_path)
        out_path = os.path.join(save_path + video_name + ".tfrecords")

        print("Converting: " + out_path)
        with tf.python_io.TFRecordWriter(out_path) as writer:
            # Iterate over all the image-paths and class-labels.
            for i, (img_path, keypoints, max_bbox) in enumerate(zip(list_imgpaths, list_keypoints, list_max_bbox)):
                with open(img_path, 'rb') as f:
                    img_raw = f.read()

                # Convert the image to raw bytes.
                # img_bytes = img_raw.tostring()
                # Create a dict with the data we want to save in the
                # TFRecords file. You can add more relevant data here.
                data = {'image': wrap_bytes(img_raw),
                        'video': wrap_int64(video_idx),
                        'keypoints': wrap_float32(keypoints),
                        'bbox_max': wrap_int64(max_bbox)
                        }

                # Wrap the data as TensorFlow Features.
                feature = tf.train.Features(feature=data)

                # Wrap again as a TensorFlow Example.
                example = tf.train.Example(features=feature)

                # Serialize the data.
                serialized = example.SerializeToString()

                # Write the serialized data to the TFRecords file.

                writer.write(serialized)

    def down_sample(self, video_path):
        """Select specific video with a high frame rate, to downsample by factor of 2"""
        image_paths = glob.glob(self.path + video_path + '*')
        image_paths = image_paths[::2]
        save_dir = self.path + 'down/'
        make_dir(save_dir)
        for i, image_path in enumerate(image_paths):
            cv2.imwrite(save_dir + str(i+1).zfill(4) + '.jpg', cv2.imread(image_path))

    def left_to_right(self, video_path):
        # self.path + video_path
        image_paths = glob.glob(self.path + video_path + '/*')
        save_dir = self.path + video_path + '_rev/'
        make_dir(save_dir)
        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            image_rev = image[:, ::-1, :]
            cv2.imwrite(save_dir + str(i+1).zfill(4) + '.jpg', image_rev)


# step 1: put to videos in folder "path/exported"
# step 2: convert videos to frames to be stored in folder "path/raw"
# step 3: make processed frames and tfrecords files, stored in "path/processed" and "path/tfrecords"

path = '../../../../myroot/jockel/'
assert os.path.exists(path)

own = OwnData(path_to_dataset=path)
# own.convert_mg4_to_img()
own.make_videos()
