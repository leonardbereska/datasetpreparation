from dataset_subclasses import PennAction, Olympic, DogRun, Human, Birds, Cats, BBCPose
import scipy.io as sio
import numpy as np
import cv2
# paths
path_to_root = '/Users/leonardbereska/myroot/'
path_to_harddrive = '/Volumes/Uni/'

# relative path
# path = path_to_harddrive + 'owny/'
# assert os.path.exists(path)

# construct dataset
# test = Jockel(path_to_dataset=path, res=600)
# extract_frames(path, from_dir='exported', to_dir='raw', video_format='mp4', img_format='jpg', frame_rate=2)
# test.process(from_dir='raw', to_dir='processed')

# PennAction
# path = path_to_harddrive + 'penn/'
# penn = PennAction(path_to_dataset=path, make_trainset=True, res=600, from_dir='frames', selected_actions=["tennis_serve", "tennis_forehand", "baseball_pitch","baseball_swing", "jumping_jacks", "golf_swing"])
# penn.process()

# Olympic Sports
# path = path_to_harddrive + 'olympic/'
# olymp = Olympic(path_to_dataset=path, make_trainset=True, res=600, from_dir='frames', selected_actions=["long_jump"])
# olymp.action_process(from_dir='frames', to_dir='processed')

# Dogs
# path = path_to_harddrive + 'dogs/'
# dogs = DogRun(path, 480)
# dogs.process()

# Human
# ID_list = ['S1', 'S5', 'S6', 'S7', 'S8', 'S9', 'S11']
# for ID in ID_list:
#     path = path_to_harddrive + 'human/human_all_copy/' + ID + '/'
#     human = Human(path, res=600, selected_actions=['walking', 'discussion'], use_mask=True, frame_rate=10)
#     human.process(to_dir=ID)
#     print('Finished {}'.format(ID))

# Birds
# path = path_to_root + 'birds/birds/'
# birds = Birds(path, res=720, make_trainset=True, from_dir='images')
# birds.process()

# Cats
# path = path_to_harddrive + 'cats/cats2/'
# cats = Cats(path_to_dataset=path, res=600, make_trainset=True, from_dir='train')
# cats.process()

# BBCPose
path = path_to_harddrive + 'bbcpose/'
bbc = BBCPose(path, res=300, make_trainset=False, from_dir='raw')
bbc.process(to_dir='test')
#
# pose = path + 'bbcpose_extbbcpose_code_1.0/' + 'bbcpose.mat'
# f = sio.loadmat(pose)
# videos = f['bbcpose'][0]
# video_idx = 1
# #
# print(len(videos))
# for video_idx in range(20):
#     video = videos[video_idx]
#     name = video['videoName']
#     type = video['type']
#     source = video['source']
#     train_frames = video['train_frames']
#     train_joints = video['train_joints']
#     test_frames = video['test_frames']
#     test_joints = video['test_joints']
#     print('{}, {}, {}, {}'.format(name, type, source, train_frames))
#
#     frame_idxs = train_frames.astype(int).astype(str)[0]
#     img_paths = [str(path + str(video_idx) + '/' + i + '.jpg') for i in frame_idxs]
#     i = 0
#
#     image = cv2.imread(img_paths[i])
#     kp_x = train_joints[0, :, i]
#     kp_y = train_joints[1, :, i]
#     bb = [np.min(kp_x), np.max(kp_x), np.min(kp_y), np.max(kp_y)]
#     bb_w = int(bb[1] - bb[0])
#     bb_h = int(bb[3] - bb[2])
#     center = [int((bb[1] + bb[0]) / 2), int((bb[3] + bb[2]) / 2)]  # x, y
#     kp = (kp_x, kp_y)
#     max_bbox = np.max([bb_w, bb_h])
#     frame = image, kp, None, max_bbox, center

