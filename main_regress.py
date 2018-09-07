"""
Ground truth keypoint (gt kp) regression and evaluation
require:
- gt kp txt file (train+test)
- pred kp txt file (train+test)
"""
import os
from regress_kp import Regressor


path_to_harddrive = '/Volumes/Uni/'
path_to_root = '/Users/leonardbereska/myroot/'


"""BBC Pose"""
dataset = 'bbcpose'
dim_y = 7
data_dir = 'bbcpose/save/'
file_dir = 'bbcpose/'

img_size = 300.


def process_kp(x):
    scal = 0.9
    x = (x + 1)/2 * scal * img_size + (1 - scal)/2 * img_size
    x = x[:, :, ::-1]
    return x

"""Birds"""
# dataset = 'birds'
# dim_y = 15
# data_dir = 'birds/birds_kp/'
# img_size = 720.
# file_dir = 'birds/'
#
# name = 'yuting_birds'
# dim_x = 10
# testpath = 'yuting_predictions_test.npy'
# trainpath = 'yuting_predictions_train.npy'
# def process_kp(x):
#     x = (x * 80 + 10) * 6 + 60
#     x = x[:, :, ::-1]
#     return x
# # train loss = 29.74293256058819
# # train pck = 0.5528333342075348
# # train pck_per_kp = [0.6503125 0.6634375 0.703125  0.7084375 0.69125   0.7075    0.3346875
# #  0.495     0.3059375 0.7296875 0.3503125 0.4915625 0.333125  0.4025
# #  0.725625 ]
# # test loss = 29.486419826864815
# # test pck = 0.556570515036583
# # test pck_per_kp = [0.65534854 0.66947114 0.7172476  0.71875    0.6974159  0.70252407
# #  0.34495193 0.47896636 0.33353364 0.7244591  0.35396636 0.5012019
# #  0.33082932 0.38581732 0.73407453]

# name = 'leo_birds_new'
# dim_x = 15
# testpath = 'leo_test_predicted_kp_new.npy'
# trainpath = 'leo_train_predicted_kp_new.npy'
# def process_kp(x):
#     x = (x + 1) * 300 + 60
#     x = x[:, :, ::-1]
#     return x
# train loss = 21.438840356253458
# train pck = 0.6666458314657211
# train pck_per_kp = [0.743125  0.854375  0.81      0.8203125 0.91      0.90875   0.45125
#  0.503125  0.3671875 0.900625  0.4475    0.495625  0.3671875 0.5071875
#  0.9134375]
# test loss = 21.021419375361425
# test pck = 0.6707331707271246
# test pck_per_kp = [0.7295673  0.87530047 0.82572114 0.82992786 0.91646636 0.9092548
#  0.44891828 0.50390625 0.36688703 0.9083534  0.45522836 0.50811297
#  0.35246393 0.51622593 0.91466343]



# BBC labels:
# 0 head; 1/2 right/left hand; 3/4 right/left elbow, 5,6 right/left shoulder

# name = 'bbc_6'
# dim_x = 20
# testpath = 'leo_test_predicted_kp.npy'
# trainpath = 'leo_train_predicted_kp.npy'
#
# name = 'bbc_8'
# dim_x = 50
# testpath = '8_test_predicted_kp.npy'
# trainpath = '8_train_predicted_kp.npy'
#
name = 'domili'
dim_x = 16
testpath = 'domili_test_predicted_kp.npy'
trainpath = 'domili_train_predicted_kp.npy'
# train loss = 7.238437057788869
# train pck = 0.9080803579092026
# train pck_per_kp = [0.99875   0.790625  0.853125  0.8290625 0.9115625 0.9775    0.9959375]
# test loss = 16.016994660369765
# test pck = 0.6160714328289032
# test pck_per_kp = [0.78125  0.703125 0.703125 0.4375   0.5      0.53125  0.65625 ]


path_to_files = path_to_root + 'test_regression/' + file_dir
path_to_data = path_to_harddrive + data_dir
assert os.path.exists(path_to_files)
assert os.path.exists(path_to_data)
regr = Regressor(dataset=dataset, name=name, dim_y=dim_y, dim_x=dim_x, b=32, lr=0.0002, n_epochs=100000, path_to_files=path_to_files,
                 path_to_data=path_to_data, testpath=testpath, trainpath=trainpath,
                 process_kp=process_kp, img_size=img_size, regularize=False)
# regr.train(from_scratch=False)
#
# regr.test(testset=False)
# regr.test(testset=True)

regr.visualize(original=True, testset=True, n_show=100, save=True, savepath=path_to_files+'images/ours_orig/')
regr.visualize(original=False, testset=True, n_show=100, save=True, savepath=path_to_files+'images/ours_regr/')
