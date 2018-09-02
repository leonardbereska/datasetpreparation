#######################################################################################################################
# IMAGE OPS
#######################################################################################################################
import cv2
import numpy as np
from scipy.misc import imresize


def save_img(img, img_path, img_shape):
    assert (img.shape == img_shape)  # must be rectangular and same size
    assert (img.dtype == np.uint8)  # save as int to reduce disk usage
    cv2.imwrite(img_path, img)  # only if well-tested


def pad_img(img, pad=[1000, 1000], kp=None, center=None, mode='symmetric'):
    pad_x = int(pad[0])
    pad_y = int(pad[1])
    if kp is not None:
        kp_x, kp_y = kp
        kp_x += float(pad_x)
        kp_y += float(pad_y)
        kp = (kp_x, kp_y)
    if center is not None:
        center[0] += float(pad_x)
        center[1] += float(pad_y)
    if mode == 'symmetric':
        img = np.lib.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode)
    elif mode == 'constant':
        img = np.lib.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode, constant_values=((255, 255), (255, 255), (255, 255)))
    else:
        raise NotImplementedError
    return img, kp, center


def crop_img(img, crop, center, kp=None):
    crop_w, crop_h = crop
    crop_x = int(center[0] - crop_w / 2)
    crop_y = int(center[1] - crop_h / 2)
    if kp is not None:
        kp_x, kp_y = kp
        kp_x -= float(crop_x)
        kp_y -= float(crop_y)
        kp = (kp_x, kp_y)
    center[0] -= float(crop_x)
    center[1] -= float(crop_y)
    img = img[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
    return img, kp, center


def resize_img(img, to_shape, kp=None, center=None):
    (h, w, _) = img.shape
    img = imresize(img, to_shape)
    if kp is not None:
        kp_x, kp_y = kp
        kp_y = kp_y / h * to_shape[1]
        kp_x = kp_x / w * to_shape[0]
        kp = [kp_x, kp_y]
    if center is not None:
        center[1] = center[1] / h * to_shape[1]
        center[0] = center[0] / w * to_shape[0]
    return img, kp, center


def invert_img(img, kp):
    kp_x, kp_y = kp
    img = img[:, ::-1, :]  # parity transform if only right eye visible
    _, w, _ = img.shape
    kp_x = w - kp_x  # invert kp x location
    kp = [kp_x, kp_y]
    return img, kp


def get_image_data(image):
    try:
        h, w, _ = image.shape
        center = [int(w / 2), int(h / 2)]  # x, y
    except AttributeError:
        return False
    return h, w, center


def get_bb(c, bb_ratio, bb_w, bb_h, w, h):
    y_min = max(int(c[0] - bb_ratio * bb_h / 2), 0)
    x_min = max(int(c[1] - bb_ratio * bb_w / 2), 0)
    y_max = min(int(c[0] + bb_ratio * bb_h / 2), h)
    x_max = min(int(c[1] + bb_ratio * bb_w / 2), w)
    too_big = y_min == 0 or x_min == 0 or y_max == h or x_max == w
    return too_big, y_min, y_max, x_min, x_max


def double_margin(img_, new_margin, kp, c, ratio, l, bb_w, bb_h):
    # pad = 1000  # max(new_h, new_w)
    kp_x, kp_y = kp
    p = int((ratio - 1) / 2 * l)
    pad = [p, p]
    img_, kp, c = pad_img(img_, pad, kp, c)

    bb_ratio_crit = 1. + new_margin * 2
    (new_h, new_w, _) = img_.shape

    _, y_min, y_max, x_min, x_max = get_bb(c, bb_ratio_crit, bb_w, bb_h, new_w, new_h)  # get box

    img_ = img_[y_min:y_max, x_min:x_max]  # crop to box
    kp_x -= float(x_min)
    kp_y -= float(y_min)
    (new_h, new_w, _) = img_.shape
    c = [int(new_h / 2), int(new_w / 2)]  # center of image

    kp = [kp_x, kp_y]
    return img_, kp, c


def pad_crop_resize(image, crop, to_shape, kp=None, center=None):
    if center is None:
        h, w, center = get_image_data(image)

    # pad
    image, kp, center = pad_img(image, center=center, kp=kp)

    # crop around center
    image, kp, center = crop_img(image, crop, center=center, kp=kp)

    # resize to final shape
    image, kp, center = resize_img(image, to_shape, center=center, kp=kp)
    return image, kp, center