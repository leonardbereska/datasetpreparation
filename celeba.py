import numpy as np
import glob
import cv2
import tensorflow as tf
import os


def make_dir(path):
    if not os.path.exists(path):
        os.mkdir(path)


def wrap_int64(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def wrap_float32(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))  # wtf only float


def wrap_bytes(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


class CelebA(object):
    def __init__(self, path_to_dataset):
        self.path = path_to_dataset
        self.n_kp = 5
        self.keypoints = self.get_kp()  # comment out for speed
        self.bboxes = self.get_bb()
        # self.n_images

    # todo train/test split (MAFL dataset)
    # Eval/ Test set partition:
        # from 162771.jpg 1 eval
        # from 182638.jpg 2 test set

    def make_videos(self, path_to_data):
        """
        Create short sequences (videos) of the 200.000 images
        :return:
        """
        # later: todo use metadata (annotations) for making grouping similar images in videos
        length = 1000  # length of one video
        n_videos = int(self.n_images / length) + 1  # rounding up here, todo dirty
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
            my_bboxes = self.bboxes[from_:to_]
            list_max_bbox = []
            for img_idx in range(length):
                max_ = max(my_bboxes[img_idx, 2:])  # get maximum of width and height
                list_max_bbox.append(max_)

            self.to_tf_records(video_idx, list_imgpaths, list_keypoints, list_max_bbox)
            print('{}/{}'.format(video_idx, n_videos))
        # except IndexError:
        #     print('Finished conversion')

    def to_tf_records(self, video_idx, list_imgpaths, list_keypoints, list_max_bbox, save_dir='tfrecords/'):
        """
        Create tfrecords file from video
        :param video_idx: int index of video
        :param list_imgpaths: all image paths in video
        :param list_keypoints: all keypoints as numpy array
        :param list_max_bbox: list of maxima of bounding box width/height
        :return:
        """
        save_path = self.path + save_dir
        make_dir(save_path)
        out_path = os.path.join(save_path + "_" + str(video_idx).zfill(2) + ".tfrecords")

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
            list_max_bbox = []

            for i, test_id in enumerate(id_list):
                image_path = self.path + 'img_align_celeba/' + test_id
                img_id = test_id.replace('.jpg', '')
                idx = int(img_id) - 1  # 0th element is 000001.jpg
                kp = self.keypoints[idx, :]
                max_bbox = np.max(self.bboxes[idx, :2])
                print(img_id)

                list_imgpaths.append(image_path)
                list_keypoints.append(kp)
                list_max_bbox.append(max_bbox)
            self.to_tf_records(video_idx=0, list_imgpaths=list_imgpaths, list_keypoints=list_keypoints, list_max_bbox=list_max_bbox)

                # def make_mafl(self, mode):
    # #     self.path
    #     length = 1000  # length of one video
    #     n_videos = int(self.n_images / length) + 1  # rounding up here, todo dirty
    #     all_images = glob.glob(path_to_data + '*')
    #
    #     for video_idx in range(n_videos):
    #         from_ = video_idx * length
    #         to_ = (video_idx + 1) * length
    #         try:
    #             list_imgpaths = all_images[from_:to_]  # todo exclude mafl
    #         except IndexError:
    #             to_ = -1
    #             list_imgpaths = all_images[from_:to_]
    #             length = len(list_imgpaths)
    #
    #         list_keypoints = self.keypoints[from_:to_]
    #         my_bboxes = self.bboxes[from_:to_]
    #         list_max_bbox = []
    #         for img_idx in range(length):
    #             max_ = max(my_bboxes[img_idx, 2:])  # get maximum of width and height
    #             list_max_bbox.append(max_)
    #
    #         self.to_tf_records(video_idx, list_imgpaths, list_keypoints, list_max_bbox)
    #         print('{}/{}'.format(video_idx, n_videos))
    #         # except IndexError:
    #         #     print('Finished conversion')

path_to_dataset = '../../../../myroot/celeba/'
assert os.path.exists(path_to_dataset)
celeb = CelebA(path_to_dataset=path_to_dataset)
# celeb.make_videos(path_to_dataset + 'img_align_celeba/')
celeb.make_mafl(mode='testing')  # or 'training'
# celeb.make_videos(path_to_dataset + 'mafl/testing/')
