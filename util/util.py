from __future__ import print_function
import os
import cv2
import numpy as np
from PIL import Image

import torch
from torch.autograd import Variable

try:
    # Python 2
    xrange
except NameError:
    # Python 3, xrange is now named range
    xrange = range

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, img_type='regular', imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    if img_type == 'regular':
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    elif img_type == 'heatmap':
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        v_max = np.max(image_numpy)
        v_min = np.min(image_numpy)
        image_numpy = ((image_numpy - v_min) / (v_max - v_min)) * 255.0
    elif img_type == 'vutils':
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0 + 0.5
        image_numpy = np.clip(image_numpy, 0, 255)
    return image_numpy.astype(imtype)

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def th_affine2d(x, matrix, output_img_width, output_img_height, center=True):
    """
    2D Affine image transform on torch.Tensor
    
    """
    assert(matrix.dim() == 2)
    matrix = matrix[:2,:]
    transform_matrix = matrix.numpy()

    src = x.numpy().transpose((1, 2, 0)).astype(np.uint8)
    # cols, rows, channels = src.shape
    dst = cv2.warpAffine(src, transform_matrix, (output_img_width,output_img_height), cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, borderValue=(127,127,127))

    # for gray image
    if len(dst.shape) == 2:
        dst = np.expand_dims(np.asarray(dst), axis=2)

    dst = dst.transpose((2, 0, 1))
    dst = torch.from_numpy(dst).float()

    return dst

def normalize_image(input_tf, normalisation_type):
    img_normalise = np.empty((input_tf.shape[0], input_tf.shape[1], input_tf.shape[2]), dtype=np.float32)
    num_channels = input_tf.shape[0]
    img = input_tf.numpy()
    
    if normalisation_type == 'channel-wise':
        for i in xrange(num_channels):
            mean = np.mean(img[i])
            std = np.std(img[i])
            if std < 1E-6:
                std = 1.0
            img_normalise[i] = (img[i] - mean) / std
    elif normalisation_type == 'regular':
        for i in xrange(num_channels):
            mean = 0.5
            std = 0.5
            img_normalise[i] = (img[i]/255. - mean) / std
    elif normalisation_type == 'heatmap':
        assert(num_channels==1)
        for i in xrange(num_channels):
            img_normalise[i] = img[i]/255.

    dst = torch.from_numpy(img_normalise).float()

    return dst

def de_normalise(batch):
    # de normalise for regular normalisation
    batch = (batch + 1.0) / 2.0 * 255.0
    return batch

def normalise_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.data.new(batch.data.size())
    std = batch.data.new(batch.data.size())
    mean[:, 0, :, :] = 0.485
    mean[:, 1, :, :] = 0.456
    mean[:, 2, :, :] = 0.406
    std[:, 0, :, :] = 0.229
    std[:, 1, :, :] = 0.224
    std[:, 2, :, :] = 0.225
    batch = torch.div(batch, 255.0)
    batch -= Variable(mean)
    batch = batch / Variable(std)
    return batch

def create_path_list(opt, prefix):
    root = opt.root_dir[0]

    A_pre = [prefix[ii] for ii in [opt.which_target]]
    A_path = ['{}{}'.format(root, AA) for AA in A_pre]
    #A_list = ['{}/{}/list_Img_1_sampled.txt'.format(root, AA) for AA in A_pre]

    B_pre = [prefix[ii] for ii in list(range(opt.which_target)) + list(range(opt.which_target + 1, len(prefix)))]
    B_path = ['{}{}'.format(root, BB) for BB in B_pre]
    #B_list = ['{}/{}/list_Img_1_sampled.txt'.format(root, BB) for BB in B_pre]    

    return A_path, B_path

def get_list(dir, label_file):
    labels = []
    images = []
    fh = open(label_file)
    for line in fh.readlines():
        item = line.split()
        path = os.path.join(dir, item.pop(-1))
        images.append(path)
        labels.append(tuple([float(v) for v in item]))
    return labels, images
