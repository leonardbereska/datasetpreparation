import glob
import os
import cv2
from ops_general import make_dir


#######################################################################################################################
# VIDEO OPS
#######################################################################################################################

def down_sample(path_to_video, save_path):
    """Select specific video with a high frame rate, to downsample by factor of 2"""
    image_paths = glob.glob(path_to_video + '*')
    image_paths = image_paths[::2]
    make_dir(save_path)
    for i, image_path in enumerate(image_paths):
        cv2.imwrite(save_path + str(i + 1).zfill(4) + '.jpg', cv2.imread(image_path))


def left_to_right(path_to_video, save_path):
    image_paths = glob.glob(path_to_video + '/*')
    make_dir(save_path)
    for i, image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        image_rev = image[:, ::-1, :]
        cv2.imwrite(save_path + str(i + 1).zfill(4) + '.jpg', image_rev)


def extract_frames(root_path, from_dir='exported', to_dir='raw', video_format='mp4', img_format='jpg', frame_rate=2,
                   deinterlace=True):
    """
    Converts video from mp4/mov to img frames
    assumes all videos in one directory
    """

    o_img = 4  # order of magnitude: maximum number of images for each activity: 10^img
    # o_vid = 2  # save for videos
    make_dir(root_path + to_dir)
    video_paths = glob.glob(root_path + from_dir + '/*')  # can use other format also e.g. .avi

    for video_idx, video_path in enumerate(video_paths):
        video_name = video_path.split('/').pop()  # e.g. afghan_hound.mp4
        video_name = video_name.replace('.{}'.format(video_format), '')  # e.g. afghan_hound
        save_path = root_path + to_dir + '/' + video_name + '/'

        if not make_dir(save_path):
            print('warning: {} folder exists, skipping...'.format(video_name))
            continue

        video_capture = cv2.VideoCapture(video_path)

        img_count = 0
        img_i = 0
        success = True
        while success:
            success, image = video_capture.read()
            try:
                if img_count % frame_rate == 0:
                    img_name = save_path + str(img_i).zfill(o_img) + '.' + img_format  # save images in video folder

                    if deinterlace:
                        # de-interlacing: delete every second row and column
                        image = image[::2, ::2, :]

                    cv2.imwrite(img_name, image)  # save frame as JPEG file
                    img_i += 1
            except:
                print('could not extract frame, continuing...')
                pass
            img_count += 1
        print('Finished extracting video {}/{}: {}'.format(video_idx, len(video_paths), video_name))


def rename_files(root_path, bg0=False, start_id=0):
    # files = glob.glob(path + 'tfrecords/*')
    files = glob.glob(root_path + 'rename/*')

    for i, file in enumerate(files):
        # os.rename(file, file.replace('exported/', 'exported/'+str.zfill(str(i), 2)))
        # os.rename(file, file.replace('.tfrecords', '-'+'.tfrecords'))
        # n, s, a, tf = file.split('-')
        # os.rename(file, n + '-' + a + '-' + s + '-' + tf)

        a = i % 5
        s = int(i / 5)
        new_name = str.zfill(str(i), 4) + '-' + str(a) + '-' + str.zfill(str(s), 3)

        os.rename(file, root_path + new_name)

    # # files = glob.glob(path + 'tfrecords/*')
    # subject_files = glob.glob(path + 'rename/' + '*')
    # i_b = 1  # start at one (0 is for other dataset)
    # for i_s, file_s in enumerate(subject_files):
    #     # os.rename(file, file.replace('exported/', 'exported/'+str.zfill(str(i), 2)))
    #     # os.rename(file, file.replace('.tfrecords', '-'+'.tfrecords'))
    #     # n, s, a, tf = file.split('-')
    #     # os.rename(file, n + '-' + a + '-' + s + '-' + tf)
    #     # subject_name = file.split('/')[-1]
    #     # os.rename(file_s, path + 's' + str(i_s))
    #     s = str.zfill(str(i_s+start_id), 3)
    #     backgr_files = glob.glob(file_s + '/*')
    #     for file_b in backgr_files:
    #         if file_b.split('/')[-1] == 'x':
    #             b = 'xxx'
    #         else:
    #             b = str.zfill(str(i_b), 3)
    #         i_b += 1
    #         videos = glob.glob(file_b + '/*')
    #         for i_v, file_v in enumerate(videos):
    #             v = str(i_v)
    #             if bg0:
    #                 b = '000'
    #             new_name = path + 'exported/' + 's' + s + '-b' + b + '-a' + v
    #             os.rename(file_v, new_name)