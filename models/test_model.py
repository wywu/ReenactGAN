from torch.autograd import Variable
from collections import OrderedDict
import util.util as util
from .base_model import BaseModel
from . import networks
import torch
import cv2
import numpy as np

class TestModel(BaseModel):
    def name(self):
        return 'TestModel'
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor

        self.fineSize_F1 = opt.fineSize_F1
        self.batchSize = opt.batchSize
        self.nc_Boundary = opt.nc_Boundary
        # define tensors
        self.real_F1 = self.Tensor(opt.batchSize, opt.nc_F1,
                                   opt.fineSize_F1, opt.fineSize_F1)
        self.boundary_map_resized = self.Tensor(opt.batchSize, opt.nc_Boundary,
                                    opt.fineSize_F1, opt.fineSize_F1)
        self.boundary_map_transformed_resized = self.Tensor(opt.batchSize, opt.nc_Boundary,
                                                opt.fineSize_F1, opt.fineSize_F1)

        # load/define networks
        self.netG = networks.define_G(opt.nc_Boundary, opt.nc_F2, opt.ngf, 'deconv_unet', opt.norm, not opt.no_dropout, opt.init_type, opt.gpu_ids)
        self.load_network(self.netG, 'decoder', opt.which_decoder, opt.model_dir)

        self.netBoundary = networks.define_G(opt.nc_F1, opt.nc_Boundary, opt.ngf, 'stack_hourglass_net', opt.norm, not opt.no_dropout, opt.init_type, opt.gpu_ids, opt.num_stacks, opt.num_blocks)
        self.load_network(self.netBoundary, 'boundary_detection', opt.which_boundary_detection, opt.model_dir)

        self.netT = networks.define_G(opt.nc_Boundary, opt.nc_Boundary, opt.ngf, 'resnet_9blocks', opt.norm, False , opt.init_type, opt.gpu_ids)
        self.load_network(self.netT, 'transformer', opt.which_transformer, opt.model_dir)

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        networks.print_network(self.netBoundary)
        networks.print_network(self.netT)
        print('-----------------------------------------------')

    def set_input(self, input):
        #if len(self.opt.gpu_ids) > 0:
        #    real_F1 = input['F1'].cuda()
        #else:
        real_F1 = input['F1']
        self.real_F1.resize_(real_F1.size()).copy_(real_F1)
        self.image_path = input['F1_path']

    def forward(self):
        self.fake_Boundary = self.netBoundary(self.real_F1)
        self.boundary_map = self.fake_Boundary[-1]
        self.boundary_map_transformed = self.netT(self.boundary_map)
        self.fake_F2 = self.netG(self.boundary_map_transformed)

        # for visualisation of boundary heatmap before transformation
        boundary_map_resized_cpu = torch.FloatTensor(self.boundary_map.shape[0], self.boundary_map.shape[1], self.fineSize_F1, self.fineSize_F1)
        for i in range(self.boundary_map.shape[0]):
            for j in range(self.boundary_map.shape[1]):
                # boundary_tmp.shape = (64,64,1)
                boundary_tmp = self.boundary_map.data[i,j,:,:].cpu().numpy()
                boundary_tmp = np.expand_dims(np.asarray(boundary_tmp), axis=2)
                boundary_tmp = boundary_tmp.astype(np.float32)
                # boundary_tmp_resized.shape = (256,256)
                boundary_tmp_resized = cv2.resize(boundary_tmp, (self.fineSize_F1,self.fineSize_F1), 0, 0, cv2.INTER_CUBIC)
                boundary_tmp_resized = np.expand_dims(np.asarray(boundary_tmp_resized), axis=2)
                boundary_tmp_resized = boundary_tmp_resized.transpose((2, 0, 1))
                boundary_tmp_resized = torch.from_numpy(boundary_tmp_resized).float()
                if j == 0:
                    boundary_map_resized_stack = boundary_tmp_resized
                else:
                    boundary_map_resized_stack = torch.cat((boundary_map_resized_stack, boundary_tmp_resized), 0)
            boundary_map_resized_cpu[i,:,:,:] = boundary_map_resized_stack
        #self.boundary_map_resized = boundary_map_resized_cpu.to(self.device)
        self.boundary_map_resized.resize_(boundary_map_resized_cpu.size()).copy_(boundary_map_resized_cpu)
        self.fake_Boundary_resized = Variable(self.boundary_map_resized)

        # for visualisation of boundary heatmap after transformation
        boundary_map_transformed_resized_cpu = torch.FloatTensor(self.boundary_map_transformed.shape[0], self.boundary_map_transformed.shape[1], self.fineSize_F1, self.fineSize_F1)
        for i in range(self.boundary_map_transformed.shape[0]):
            for j in range(self.boundary_map_transformed.shape[1]):
                # boundary_tmp.shape = (64,64,1)
                boundary_tmp = self.boundary_map_transformed.data[i,j,:,:].cpu().numpy()
                boundary_tmp = np.expand_dims(np.asarray(boundary_tmp), axis=2)
                boundary_tmp = boundary_tmp.astype(np.float32)
                # boundary_tmp_resized.shape = (256,256)
                boundary_tmp_resized = cv2.resize(boundary_tmp, (self.fineSize_F1,self.fineSize_F1), 0, 0, cv2.INTER_CUBIC)
                boundary_tmp_resized = np.expand_dims(np.asarray(boundary_tmp_resized), axis=2)
                boundary_tmp_resized = boundary_tmp_resized.transpose((2, 0, 1))
                boundary_tmp_resized = torch.from_numpy(boundary_tmp_resized).float()
                if j == 0:
                    boundary_map_resized_stack = boundary_tmp_resized
                else:
                    boundary_map_resized_stack = torch.cat((boundary_map_resized_stack, boundary_tmp_resized), 0)
            boundary_map_transformed_resized_cpu[i,:,:,:] = boundary_map_resized_stack
        #self.boundary_map_transformed_resized = boundary_map_transformed_resized_cpu.to(self.device)
        self.boundary_map_transformed_resized.resize_(boundary_map_transformed_resized_cpu.size()).copy_(boundary_map_transformed_resized_cpu)
        self.fake_Boundary_transformed_resized = Variable(self.boundary_map_transformed_resized)

    def get_image_paths(self):
        return self.image_path

    def get_current_visuals(self):
        real_F1 = util.tensor2im(self.real_F1.data)
        fake_F2 = util.tensor2im(self.fake_F2.data)

        # visualise 15-channels boundary heatmap before transformation on 1-channel
        boundary_map = np.zeros([self.fake_Boundary_resized.data.shape[2], self.fake_Boundary_resized.data.shape[3], 1])
        assert(boundary_map.shape == (256, 256, 1))
        for i in range(self.fake_Boundary_resized.data.shape[1]):
            boundary_map_tmp = util.tensor2im(self.fake_Boundary_resized.data[:,i,:,:].unsqueeze(1), 'heatmap')
            for j in range(self.fake_Boundary_resized.data.shape[2]):
                for k in range(self.fake_Boundary_resized.data.shape[3]):
                    if boundary_map_tmp[j,k,0] > boundary_map[j,k,0]:
                        boundary_map[j,k,0] = boundary_map_tmp[j,k,0]
        boundary_map = np.tile(boundary_map, (1, 1, 3)).astype(np.uint8)

        # visualise 15-channels boundary heatmap after transformation on 1-channel
        boundary_map_transformed = np.zeros([self.fake_Boundary_transformed_resized.data.shape[2], self.fake_Boundary_transformed_resized.data.shape[3], 1])
        assert(boundary_map_transformed.shape == (256, 256, 1))
        for i in range(self.fake_Boundary_transformed_resized.data.shape[1]):
            boundary_map_transformed_tmp = util.tensor2im(self.fake_Boundary_transformed_resized.data[:,i,:,:].unsqueeze(1), 'heatmap')
            for j in range(self.fake_Boundary_transformed_resized.data.shape[2]):
                for k in range(self.fake_Boundary_transformed_resized.data.shape[3]):
                    if boundary_map_transformed_tmp[j,k,0] > boundary_map_transformed[j,k,0]:
                        boundary_map_transformed[j,k,0] = boundary_map_transformed_tmp[j,k,0]
        boundary_map_transformed = np.tile(boundary_map_transformed, (1, 1, 3)).astype(np.uint8)

        return OrderedDict([('real_F1', real_F1), ('boundary_map', boundary_map), ('boundary_map_transformed', boundary_map_transformed), ('fake_F2', fake_F2)])

    # get image paths
    def get_image_paths(self):
        return self.image_path
