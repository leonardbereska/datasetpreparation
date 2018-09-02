import numpy as np
import glob
import cv2
import tensorflow as tf
import os
from matplotlib import pyplot as plt
mode = 'raw_kp'
which = 'dogs'
# mode = 'raw_kp'

if mode == 'regr':
    things = ['cats10', 'cats20', 'celeb10', 'celeb30', 'human', 'penn']
elif mode == 'raw_kp':
    things = ['cats10', 'cats20', 'celeb10', 'celeb30', 'dogs', 'human', 'penn']

# for thing in things:  # put folder with cherries
#     which = thing


# # for raw_kp
if mode == 'raw_kp':
    # grid_w = 10
    # grid_h = 12
    grid_w = 6
    grid_h = 1

elif mode == 'regr':
    grid_w = 4
    grid_h = 10

save_path = '../../../../myroot/images_regr/cherries/cherries/dogs'   #raw_kp/'
# path = save_path + which + '/'
path = save_path

images = []

image_paths = glob.glob(path + '*')
# image_path = image_paths[0] # todo out
if mode == 'raw_kp':
    c = 8
    h = 470
    w = h  # todo these par for all?
elif mode == 'regr':
    c = 8
    h = 310
    w = 2 * h

if not mode == 'regr':
    figsize = (grid_w,grid_h)
else:
    figsize = (grid_w, grid_h)
fig, axes = plt.subplots(grid_h, grid_w, figsize=figsize, frameon=False)
for i in range(grid_h):
    for j in range(grid_w):
        idx = j + grid_w * i
        # print(idx)
        # idx = np.random.randint(len(image_paths))
        image_path = image_paths[idx]
        image = cv2.imread(image_path, )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_crop = image[c:h + c, c:w + c, :]
        if grid_h > 1:
            ax = axes[i, j]
        else:
            ax = axes[j]
        ax.imshow(image_crop)
        ax.axis('off')
        ax.set_aspect('auto')
        if mode == 'regr':
            ax.set_aspect('equal')

if not mode == 'regr':
    plt.subplots_adjust(hspace=0, wspace=0)

plt.axis('off')#fig.tight_layout()

# plt.show()
plt.savefig('{}img/{}w{}h{}.png'.format(save_path, which, grid_w, grid_h), bbox_inches='tight', pad_inches=0, format='png',
            dpi=500)
print('saved {}'.format(which))
