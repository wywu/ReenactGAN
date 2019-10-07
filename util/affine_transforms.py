"""
Affine transforms implemented on torch tensors, and
requiring only one interpolation
"""

import math
import random
import torch as th
import cv2

from .util import th_affine2d, normalize_image
import numpy as np

try:
    # Python 2
    xrange
except NameError:
    # Python 3, xrange is now named range
    xrange = range

class AffineCompose(object):

    def __init__(self,
                rotation_range,
                translation_range,
                zoom_range,
                output_img_width,
                output_img_height,
                mirror=False,
                normalise=False,
                normalisation_type='regular',
                ):

        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.zoom_range = zoom_range
        self.output_img_width = output_img_width
        self.output_img_height = output_img_height
        self.mirror = mirror
        self.normalise = normalise
        self.normalisation_type = normalisation_type

    def __call__(self, *inputs):
        input_img_width = inputs[0].size(1)
        input_img_height = inputs[0].size(2)
        rotate = random.uniform(-self.rotation_range, self.rotation_range)
        trans_x = random.uniform(-self.translation_range, self.translation_range)
        trans_y = random.uniform(-self.translation_range, self.translation_range)
        if not isinstance(self.zoom_range, list) and not isinstance(self.zoom_range, tuple):
            raise ValueError('zoom_range must be tuple or list with 2 values')
        zoom = random.uniform(self.zoom_range[0], self.zoom_range[1])

        # rotate 
        transform_matrix = th.FloatTensor(3, 3).zero_()
        center = (inputs[0].size(1)/2.-0.5, inputs[0].size(2)/2-0.5)
        M = cv2.getRotationMatrix2D(center, rotate, 1)
        transform_matrix[:2,:] = th.from_numpy(M).float()
        transform_matrix[2,:] = th.FloatTensor([[0, 0, 1]])
        # translate 
        transform_matrix[0,2] += trans_x
        transform_matrix[1,2] += trans_y
        # zoom
        for i in xrange(3):
            transform_matrix[0,i] *= zoom
            transform_matrix[1,i] *= zoom
        transform_matrix[0,2] += (1.0 - zoom) * center[0]
        transform_matrix[1,2] += (1.0 - zoom) * center[1]
        # if needed, apply crop together with affine to accelerate
        transform_matrix[0,2] -= (input_img_width-self.output_img_width) / 2.0;
        transform_matrix[1,2] -= (input_img_height-self.output_img_height) / 2.0;

        # mirror about x axis in cropped image
        do_mirror = False
        if self.mirror:
            mirror_rng = random.uniform(0.,1.)
            if mirror_rng>0.5:
                do_mirror = True
        if do_mirror:
            transform_matrix[0,0] = -transform_matrix[0,0]
            transform_matrix[0,1] = -transform_matrix[0,1]
            transform_matrix[0,2] = float(self.output_img_width)-transform_matrix[0,2];


        outputs = []
        for idx, _input in enumerate(inputs):
            input_tf = th_affine2d(_input,
                                   transform_matrix,
                                   output_img_width=self.output_img_width,
                                   output_img_height=self.output_img_height)
            if self.normalise:
                # input_tf.shape: (1L/3L, 256L, 256L)
                if idx == 0:
                    input_tf = normalize_image(input_tf, self.normalisation_type)
                else:
                    # for heatmap ground truth generation
                    input_tf = normalize_image(input_tf, 'heatmap')

            outputs.append(input_tf)
        return outputs if idx >= 1 else outputs[0]
