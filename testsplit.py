import glob
import numpy as np
import os
"""
- Extracts all image paths from folder structure, 
- make a train-test-split 
- and write these paths to text files.
"""


def get_all_img_paths(path_to_folder):
    """
    Get all paths to images in the folder and subfolders
    :return: relative_paths
    """
    all_subfolders = glob.glob(path_to_folder + '*')
    all_paths = []
    for folder in all_subfolders:
        all_paths.extend(glob.glob(folder + '/*'))
    # get relative paths
    common_prefix = path_to_folder
    relative_paths = [os.path.relpath(path, common_prefix) for path in all_paths]
    return relative_paths


def make_train_test_split(all_paths, train_test_ratio):

    # random split
    np.random.seed(1)
    train_size = int(len(all_paths)*train_test_ratio)
    index_list = np.random.choice(range(len(all_paths)), size=train_size, replace=None)
    list_train = [all_paths[i] for i in index_list]
    list_test = sorted(list(set(all_paths) - set(list_train)))
    return list_train, list_test


def list_to_txt(list, name):
    with open(name, 'w+') as file:
        for i, img_path in enumerate(list):
            file.write(img_path + '\n')


path_to_root = '/Users/leonardbereska/myroot/'
path = path_to_root + 'birds_big_processed/'

all_img_paths = get_all_img_paths(path)
list_train, list_test = make_train_test_split(all_img_paths, train_test_ratio=0.8)
list_to_txt(list_train, 'birds_train.txt')
list_to_txt(list_test, 'birds_test.txt')