import numpy as np
import torch
import os
import cv2
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

try:
    # Python 2
    xrange
except NameError:
    # Python 3, xrange is now named range
    xrange = range

class Face2Boundary2FaceModel(BaseModel):
    def name(self):
        return 'Face2Boundary2FaceModel'
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

        self.input_nc = opt.input_nc
        self.output_nc = opt.output_nc
        self.output_nc = opt.output_nc

        self.size = opt.fineSize_F1
        self.fineSize_F1 = opt.fineSize_F1
        self.fineSize_Boundary = opt.fineSize_Boundary
        self.batchSize = opt.batchSize
        self.nc_Boundary = opt.nc_Boundary
        self.feature_loss = opt.feature_loss
        self.input_type_D = opt.input_type_D
        self.feature_loss_type = opt.feature_loss_type
        # define tensors
        self.input_F1 = self.Tensor(opt.batchSize, opt.nc_F1,
                                   opt.fineSize_F1, opt.fineSize_F1)
        self.input_Boundary = self.Tensor(opt.batchSize, opt.nc_Boundary,
                                   opt.fineSize_Boundary, opt.fineSize_Boundary)
        self.input_F2 = self.Tensor(opt.batchSize, opt.nc_F2,
                                   opt.fineSize_F2, opt.fineSize_F2)
        self.boundary_map_resized = self.Tensor(opt.batchSize, opt.nc_Boundary,
                                   opt.fineSize_F1, opt.fineSize_F1)

        # load/define networks
        self.netG = networks.define_G(opt.nc_Boundary, opt.nc_F2, opt.ngf,
                                    opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        self.netBoundary = networks.define_G(opt.nc_F1, opt.nc_Boundary, opt.ngf, 'stack_hourglass_net', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids, opt.num_stacks, opt.num_blocks)
        self.load_network(self.netBoundary, 'boundary_detection', opt.which_boundary_detection, model_dir=self.opt.pretrain_root)

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.nc_Boundary + opt.nc_F2, self.size, opt.ndf,
                                        opt.which_model_netD, opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if self.feature_loss:
            self.vgg = networks.define_Feature_Net(False, 'vgg16', self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch, self.opt.load_path)
            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, self.opt.load_path)

        # also need to use in testing
        self.criterionL2 = torch.nn.MSELoss()
        if self.isTrain:
            self.fake_F1F2_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor, reduce=True)
            self.criterionL1 = torch.nn.L1Loss()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        networks.print_network(self.netBoundary)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        input_F1 = input['F1']
        input_Boundary = input['Boundary']
        input_F2 = input['F2']
        self.input_F1.resize_(input_F1.size()).copy_(input_F1)
        self.input_Boundary.resize_(input_Boundary.size()).copy_(input_Boundary)
        self.input_F2.resize_(input_F2.size()).copy_(input_F2)
        self.image_path = input['F1_path']

    def forward(self):
        self.real_Boundary = Variable(self.input_Boundary)
        self.real_F1 = Variable(self.input_F1)
        self.netBoundaryFeature = networks.get_features_Boundary(self.netBoundary, requires_grad=False)
        self.fake_Boundary = self.netBoundaryFeature(self.real_F1)
        self.boundary_map = self.fake_Boundary[-1]
        self.fake_F2 = self.netG(self.boundary_map)
        self.real_F2 = Variable(self.input_F2)
        
        boundary_map_resized_cpu = torch.FloatTensor(self.boundary_map.shape[0], self.boundary_map.shape[1], self.fineSize_F1, self.fineSize_F1)
        for i in xrange(self.boundary_map.shape[0]):
            for j in xrange(self.boundary_map.shape[1]):
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
        self.boundary_map_resized.resize_(boundary_map_resized_cpu.size()).copy_(boundary_map_resized_cpu)
        self.fake_Boundary_resized = Variable(self.boundary_map_resized)

        # feature loss
        if self.feature_loss:
            fake_F2_de_norm = util.de_normalise(self.fake_F2)
            real_F2_de_norm = util.de_normalise(self.real_F2)
            fake_F2_norm = util.normalise_batch(fake_F2_de_norm)
            real_F2_norm = util.normalise_batch(real_F2_de_norm)
            self.features_fake_F2 = self.vgg(fake_F2_norm)
            self.features_real_F2 = self.vgg(real_F2_norm)

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_F1F2 = self.fake_F1F2_pool.query(torch.cat((self.fake_Boundary_resized, self.fake_F2), 1).data)

        pred_fake = self.netD(fake_F1F2.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_F1F2 = torch.cat((self.fake_Boundary_resized, self.real_F2), 1)

        pred_real = self.netD(real_F1F2)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake_F1F2 = torch.cat((self.fake_Boundary_resized, self.fake_F2), 1)

        pred_fake = self.netD(fake_F1F2)

        # Second, G(A) = B
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        self.loss_G_Pixel = self.criterionL1(self.fake_F2, self.real_F2) * self.opt.lambda_pix_loss

        self.loss_G = self.loss_G_GAN + self.loss_G_Pixel

        # Third, feature loss
        if self.feature_loss:
            if self.feature_loss_type == 'relu1_2':
                self.loss_G_Feature = self.criterionL2(self.features_fake_F2.relu1_2, self.features_real_F2.relu1_2) * self.opt.lambda_feature_loss
            elif self.feature_loss_type == 'relu2_2':
                self.loss_G_Feature = self.criterionL2(self.features_fake_F2.relu2_2, self.features_real_F2.relu2_2) * self.opt.lambda_feature_loss
            elif self.feature_loss_type == 'relu3_3':
                self.loss_G_Feature = self.criterionL2(self.features_fake_F2.relu3_3, self.features_real_F2.relu3_3) * self.opt.lambda_feature_loss
            elif self.feature_loss_type == 'relu4_3':
                self.loss_G_Feature = self.criterionL2(self.features_fake_F2.relu4_3, self.features_real_F2.relu4_3) * self.opt.lambda_feature_loss
            elif self.feature_loss_type == 'relu1_2_and_relu2_2':
                self.loss_G_Feature = (self.criterionL2(self.features_fake_F2.relu1_2, self.features_real_F2.relu1_2) +
                                       self.criterionL2(self.features_fake_F2.relu2_2, self.features_real_F2.relu2_2)) * self.opt.lambda_feature_loss
            elif self.feature_loss_type == 'relu2_2_and_relu3_3':
                self.loss_G_Feature = (self.criterionL2(self.features_fake_F2.relu2_2, self.features_real_F2.relu2_2) +
                                       self.criterionL2(self.features_fake_F2.relu3_3, self.features_real_F2.relu3_3)) * self.opt.lambda_feature_loss
            elif self.feature_loss_type == 'relu3_3_and_relu4_3':
                self.loss_G_Feature = (self.criterionL2(self.features_fake_F2.relu3_3, self.features_real_F2.relu3_3) +
                                       self.criterionL2(self.features_fake_F2.relu4_3, self.features_real_F2.relu4_3)) * self.opt.lambda_feature_loss
            self.loss_G = self.loss_G + self.loss_G_Feature

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        if self.feature_loss:
            return OrderedDict([('G_GAN', self.loss_G_GAN.data.item()),
                                ('G_Pixel', self.loss_G_Pixel.data.item()),
                                ('G_Feature', self.loss_G_Feature.data.item()),
                                ('D_real', self.loss_D_real.data.item()),
                                ('D_fake', self.loss_D_fake.data.item())
                                ])
        else:
            return OrderedDict([('G_GAN', self.loss_G_GAN.data.item()),
                                ('G_Pixel', self.loss_G_Pixel.data.item()),
                                ('D_real', self.loss_D_real.data.item()),
                                ('D_fake', self.loss_D_fake.data.item())
                                ])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netBoundary, 'B', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

    def get_current_visuals(self):
        real_F1 = util.tensor2im(self.real_F1.data)

        boundary_map = np.zeros([self.fake_Boundary_resized.data.shape[2], self.fake_Boundary_resized.data.shape[3], 1])
        assert(boundary_map.shape == (256, 256, 1))
        for i in xrange(self.fake_Boundary_resized.data.shape[1]):
            boundary_map_tmp = util.tensor2im(self.fake_Boundary_resized.data[:,i,:,:].unsqueeze(1), 'heatmap')
            for j in xrange(self.fake_Boundary_resized.data.shape[2]):
                for k in xrange(self.fake_Boundary_resized.data.shape[3]):
                    if boundary_map_tmp[j,k,0] > boundary_map[j,k,0]:
                        boundary_map[j,k,0] = boundary_map_tmp[j,k,0]
        boundary_map = np.tile(boundary_map, (1, 1, 3)).astype(np.uint8)

        fake_F2 = util.tensor2im(self.fake_F2.data)
        
        return OrderedDict([('real_F1', real_F1), ('boundary_map', boundary_map), ('fake_F2', fake_F2)])
