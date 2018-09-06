"""
Ground truth keypoint (gt kp) regression and evaluation
require:
- gt kp txt file (train+test)
- pred kp txt file (train+test)
"""
import os
from regress_kp import Regressor


def set_paths(data_dir, file_dir='test_regression/'):
    path_to_harddrive = '/Volumes/Uni/'
    path_to_root = '/Users/leonardbereska/myroot/'
    path_to_files = path_to_root + file_dir
    path_to_data = path_to_harddrive + data_dir
    assert os.path.exists(path_to_files)
    assert os.path.exists(path_to_data)
    return path_to_data, path_to_files


"""Birds"""
# dim_y = 15
# data_dir = 'birds/birds_kp/'
# path_to_data, path_to_files = set_paths(data_dir)
# img_size = 720.
# name = 'yuting_birds'  # todo save this data automatically
# dim_x = 10
# testpath ='yuting_predictions_test.npy'
# trainpath = 'yuting_predictions_train.npy'
# process_kp = lambda x: (x * 80 + 10) * 6 + 60
# test loss: 32.15

# name = 'leo_birds'
# dim_x = 15
# testpath ='leo_test_predicted_kp.npy'
# trainpath = 'leo_train_predicted_kp.npy'
# process_kp = lambda x: (x + 1) * 200 + 160
# test loss: 27.58

# name = 'leo_birds_new'
# dim_x = 15
# testpath ='leo_test_predicted_kp_par.npy'
# trainpath = 'leo_train_predicted_kp_par.npy'
# process_kp = lambda x: (x + 1) * 300 + 60
# # test loss: 23.51

# name = 'leo_par'
# dim_x = 15
# testpath ='leo_test_predicted_kp_par.npy'
# trainpath = 'leo_train_predicted_kp_par.npy'
# process_kp = lambda x: (x + 1) * 200 + 160
# test loss: 30.08


"""BBC Pose"""
dim_y = 7
data_dir = 'bbcpose/'
path_to_data, path_to_files = set_paths(data_dir)
img_size = 300.


def process_kp(x):
    scal = 0.9
    x = (x + 1)/2 * scal * img_size + (1 - scal)/2 * img_size
    x = x[:, :, ::-1]
    return x


# name = 'bbc_8'
# dim_x = 50
# testpath = 'leo_test_predicted_kp.npy'
# trainpath = 'leo_train_predicted_kp.npy'

name = 'domili'
dim_x = 16
testpath = 'domili_test_predicted_kp.npy'
trainpath = 'domili_train_predicted_kp.npy'

regr = Regressor(name=name, dim_y=dim_y, dim_x=dim_x, b=12, lr=0.0002, n_epochs=10000, path_to_files=path_to_files,
                 path_to_data=path_to_data, testpath=testpath, trainpath=trainpath,
                 process_kp=process_kp, img_size=img_size)
regr.train(from_scratch=False)

regr.test(testset=False)
regr.test(testset=True)

regr.visualize(original=True)
