import os.path
import cv2
import sys
from PIL import Image
import numpy as np
import torch
from data.base_dataset import BaseDataset
from util.affine_transforms import AffineCompose
from util.util import normalize_image, get_list

class AlignedFace2Boudnary2Face(BaseDataset):
    def __init__(self, opt):
        super(AlignedFace2Boudnary2Face, self).__init__()
        self.opt = opt
        self.F1_paths = []
        self.F1_labels = []
        for root_dir_item in opt.root_dir:
            #print("root_dir_item: ",root_dir_item)
            dir_F1_tmp = os.path.join(root_dir_item, 'Image')
            img_list = os.path.join(root_dir_item, opt.name_landmarks_list)
            F1_labels_tmp, F1_paths_tmp = get_list(dir_F1_tmp, img_list)

            self.F1_paths += F1_paths_tmp
            self.F1_labels += F1_labels_tmp

        self.serial_batches = opt.serial_batches
        self.fineSize_F1 = opt.fineSize_F1
        self.fineSize_Boundary = opt.fineSize_Boundary
        self.nc_Boundary = opt.nc_Boundary
        self.sigma = opt.sigma
        self.transform = AffineCompose(rotation_range=opt.rotate_range,
                                      translation_range=opt.translate_range,
                                      zoom_range=opt.zoom_range,
                                      output_img_width=opt.fineSize_F1,
                                      output_img_height=opt.fineSize_F1,
                                      mirror=opt.mirror,
                                      normalise=opt.normalise,
                                      normalisation_type=opt.normalisation_type)

    def __getitem__(self, index):
        inputs = []
        inputs_transform = []

        F1_path = self.F1_paths[index]
        #print(F1_path)
        F1_img = cv2.imread(F1_path, 1)
        # convert BGR to RGB
        b,g,r = cv2.split(F1_img)
        F1_img = cv2.merge([r,g,b])
        F1_img = np.asarray(F1_img)
        F1_img = F1_img.transpose((2, 0, 1))
        F1_img = torch.from_numpy(F1_img).float()
        inputs.append(F1_img)

        inputs_transform = self.transform(*inputs)

        F1 = inputs_transform
        Boundary = torch.FloatTensor(self.opt.nc_Boundary,self.opt.fineSize_Boundary,self.opt.fineSize_Boundary).zero_()

        # debug
        assert(F1.shape == (self.opt.nc_F1,self.opt.fineSize_F1,self.opt.fineSize_F1))
        assert(Boundary.shape == (self.opt.nc_Boundary,self.opt.fineSize_Boundary,self.opt.fineSize_Boundary))

        return {'F1': F1, 'Boundary': Boundary, 'F2': F1, 'F1_path': F1_path}


    def __len__(self):
        return len(self.F1_paths)

    def name(self):
        return 'AlignedFace2Boudnary2Face'

class TransformerDataset(BaseDataset):
    def __init__(self, opt, pic_path):
        super(TransformerDataset, self).__init__()
        self.opt = opt
        self.pic_path = pic_path[0] + '/Image'
        #self.pic_path = pic_path[0] + '/train_A'
        self.F1_paths = []

        img_list = os.path.join(pic_path[0], opt.name_landmarks_list)
        F1_labels_tmp, F1_paths_tmp = get_list(self.pic_path, img_list)
        self.F1_paths += F1_paths_tmp

        self.transform_align = AffineCompose(rotation_range=opt.rotate_range,
                                      translation_range=opt.translate_range,
                                      zoom_range=opt.zoom_range,
                                      output_img_width=opt.fineSize_F1,
                                      output_img_height=opt.fineSize_F1,
                                      mirror=opt.mirror,
                                      normalise=opt.normalise,
                                      normalisation_type=opt.normalisation_type)

    def __getitem__(self, index):
        inputs = []
        F1_path = self.F1_paths[index]
        F1_img = cv2.imread(F1_path, 1)
        # convert BGR to RGB
        b,g,r = cv2.split(F1_img)
        F1_img = cv2.merge([r,g,b])
        F1_img = np.asarray(F1_img)
        F1_img = F1_img.transpose((2, 0, 1))
        F1_img = torch.from_numpy(F1_img).float()
        inputs.append(F1_img)

        inputs_transform = self.transform_align(*inputs)
        F1 = inputs_transform
        return F1

    def __len__(self):
        return len(self.F1_paths)

    def name(self):
        return 'TransferDataset'

class SingleDataset(BaseDataset):
    def __init__(self, opt):
        super(SingleDataset, self).__init__()
        self.opt = opt

        self.F1_paths = []
        self.F1_labels = []
        for root_dir_item in opt.root_dir:
            dir_F1_tmp = root_dir_item
            img_list = opt.name_list
            with open(img_list, 'r') as f:
                tmp = f.readlines()
            F1_paths_tmp = [dir_F1_tmp + ii.strip('\n') for ii in tmp]
            self.F1_paths += F1_paths_tmp

        self.fineSize_F1 = opt.fineSize_F1
        self.fineSize_Boundary = opt.fineSize_Boundary
        self.nc_Boundary = opt.nc_Boundary

    def __getitem__(self, index):

        F1_path = self.F1_paths[index]
        F1_img = cv2.imread(F1_path, 1)
        # convert BGR to RGB
        b,g,r = cv2.split(F1_img)
        F1_img = cv2.merge([r,g,b])
        F1_img = np.asarray(F1_img)
        #crop_center
        #F1_img = F1_img[64:64 + 256, 64:64 + 256,:]
        F1_img = F1_img.transpose((2, 0, 1))
        F1_img = torch.from_numpy(F1_img).float()
        F1_img = normalize_image(F1_img, 'regular')

        # check
        assert(F1_img.shape == (self.opt.nc_F1,self.opt.fineSize_F1,self.opt.fineSize_F1))

        return {'F1': F1_img, 'F1_path': F1_path}

    def __len__(self):
        return len(self.F1_paths)

    def name(self):
        return 'SingleDataset'
