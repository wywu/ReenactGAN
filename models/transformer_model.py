import os
import sys
import time
from itertools import chain
from collections import OrderedDict

import torch
from torch import nn
import numpy as np
import torchvision.utils as vutils

from . import networks
import util.util as util
from .base_model import BaseModel

class TransformerModel(BaseModel):
    def name(self):
        return 'TransformerModel'
    def __init__(self, opt, multi_n):
        self.opt = opt
        self.gpu_ids = self.opt.gpu_ids
        self.multi_n = multi_n
        self.model_dir = self.opt.model_dir
        self.nrow = 8 if self.opt.batchSize >= 8 else self.opt.batchSize
        print(torch.__version__, 'torch version')

        self.build_model()
        self.build_loss(use_lsgan=not self.opt.no_lsgan)

        self.loss_list = {'patch':self.patchLoss, 'img':self.imgLoss}

        self.real_patch = self.real_patch.cuda()
        self.fake_patch = self.fake_patch.cuda()

        if self.opt.continue_train:
            self.load_network()

        self._init_Align() # map heatmap to 212 vectors, that is x,y 106 points
        self._init_PCA()  
        self._init_Bound() # not stack, generate boundary
        self._init_Edge()  # use laplas gradient


        optimizer = torch.optim.Adam
        self.optimizer_D = optimizer(
            chain(*[vv.parameters() for kk, vv in self.D_net.items()]),
            lr=0.5*self.opt.lr, betas=(self.opt.beta1, self.opt.beta2), weight_decay=self.opt.weight_decay)

        self.optimizer_G = optimizer(
            chain(*[vv.parameters() for kk, vv in self.G_net.items()]),
            lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))

    def build_model(self):
        self.G_net = {}
        self.G_net['G_BA'] = networks.define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.which_model_netG, self.opt.norm, not self.opt.no_dropout, self.opt.init_type, self.gpu_ids)

        for ii in range(self.multi_n):
            self.G_net['G_AB{}'.format(ii)] = networks.define_G(self.opt.input_nc, self.opt.output_nc, self.opt.ngf, self.opt.which_model_netG, self.opt.norm, not self.opt.no_dropout, self.opt.init_type, self.gpu_ids)

        use_sigmoid = self.opt.no_lsgan
        D_A = networks.define_D(self.opt.output_nc, None, self.opt.ndf, self.opt.which_model_netD, self.opt.n_layers_D, self.opt.norm, use_sigmoid, self.opt.init_type, self.gpu_ids)
        self.D_net = {'D_A': D_A}
        for ii in range(self.multi_n):
            self.D_net['D_B{}'.format(ii)] = networks.define_D(self.opt.output_nc, None, self.opt.ndf, self.opt.which_model_netD, self.opt.n_layers_D, self.opt.norm, use_sigmoid, self.opt.init_type, self.gpu_ids)

    def build_loss(self, use_lsgan=True):
        #patch
        if use_lsgan:
            self.patchLoss = nn.MSELoss()
        else:
            self.patchLoss = nn.BCELoss()

        if self.opt.n_layers_D == 2:
            map_size = 14
        elif self.opt.n_layers_D == 3:
            map_size = 6
        self.real_patch = torch.FloatTensor(self.opt.batchSize, 1, map_size, map_size).fill_(1)
        self.fake_patch = torch.FloatTensor(self.opt.batchSize, 1, map_size, map_size).fill_(0)
        #id loss
        self.imgLoss = torch.nn.L1Loss()

    def load_network(self):
        map_location = None
        for name, model in tuple(self.G_net.items()) + tuple(self.D_net.items()):
            file_name = '{}/{}_{}.pth'.format(self.opt.load_path, name, self.opt.which_iter)
            if not os.path.isfile(file_name):
                print('no CheckPoint in {}\n'.format(file_name))
                return

            pre_model = torch.load(file_name, map_location=map_location)
            model_dict = model.state_dict()
            for kk, vv in pre_model.items():
                model_dict[kk].copy_(vv)
        print("[*] Model loaded: {}".format(self.opt.load_path))

    def set_input(self, x_A, x_B):
        self.x_A = x_A
        self.x_B = x_B

    def optimize_parameters(self):
        self.Bound.zero_grad()
        self.Edge.zero_grad()
        with torch.no_grad(): 
            self.x_A = self.Bound(self._get_variable(self.x_A))[1]
            self.x_B = [self.Bound(self._get_variable(bb))[1] for bb in self.x_B]

        self.optimizer_D.zero_grad()
        self.update_D()
        self.optimizer_D.step()

        self.Align.zero_grad()
        self.PCA.zero_grad()
        self.optimizer_G.zero_grad()
        self.update_G()
        self.optimizer_G.step()

    def update_D(self):
        with torch.no_grad(): 
            # update D network
            x_AB = [self.G_net['G_AB{}'.format(ii)](self.x_A) for ii in range(self.multi_n)]
            x_BA = [self.G_net['G_BA'](self.x_B[ii]) for ii in range(self.multi_n)]

        #update A
        D_A = self.D_net['D_A']
        D_A.zero_grad()
        self.loss_d_real_A = self.patchLoss(D_A(self._gen_Edge(self.x_A)), self.real_patch)
        self.loss_d_fake_A = [self.patchLoss(D_A(self._gen_Edge(ba)), self.fake_patch) for ba in x_BA]

        loss_d_A = self.loss_d_real_A + sum(self.loss_d_fake_A)

        #update B
        self.loss_d_real_B = []
        self.loss_d_fake_B = []
        for ii in range(self.multi_n):
            D_B = self.D_net['D_B{}'.format(ii)]
            D_B.zero_grad()
            self.loss_d_real_B.append(self.patchLoss(D_B(self._gen_Edge(self.x_B[ii])), self.real_patch))
            self.loss_d_fake_B.append(self.patchLoss(D_B(self._gen_Edge(x_AB[ii])), self.fake_patch))

        loss_d_B = sum(self.loss_d_real_B) + sum(self.loss_d_fake_B)

        self.loss_d = loss_d_A + loss_d_B
        self.loss_d.backward()

    def update_G(self):
        # update G network
        self.loss_G_AB = {'gan':[], 'cycle':[], 'pca':[]}
        self.loss_G_BA = {'gan':[], 'cycle':[], 'pca':[]}

        lam = self.opt.lam_pix
        lam_pca = self.opt.lam_align

        G_BA = self.G_net['G_BA']
        G_BA.zero_grad()
        for ii in range(self.multi_n):
            G_AB = self.G_net['G_AB{}'.format(ii)]
            G_AB.zero_grad()

            x_AB = G_AB(self.x_A)
            x_BA = G_BA(self.x_B[ii])

            x_ABA = G_BA(x_AB)
            x_BAB = G_AB(x_BA)

            self.loss_G_AB['gan'].append(self.patchLoss(self.D_net['D_B{}'.format(ii)](self._gen_Edge(x_AB)), self.real_patch))
            self.loss_G_BA['gan'].append(self.patchLoss(self.D_net['D_A'](self._gen_Edge(x_BA)), self.real_patch))

            self.loss_G_AB['cycle'].append(self.imgLoss(x_ABA, self.x_A))
            self.loss_G_BA['cycle'].append(self.imgLoss(x_BAB, self.x_B[ii]))

            self.loss_G_AB['pca'].append(self._pca_align_loss(x_AB, self.x_A))
            self.loss_G_BA['pca'].append(self._pca_align_loss(x_BA, self.x_B[ii]))

        self.loss_gan = sum(self.loss_G_AB['gan']) + sum(self.loss_G_BA['gan'])
        self.loss_cycle = sum(self.loss_G_AB['cycle']) + sum(self.loss_G_BA['cycle'])
        self.loss_pca = sum(self.loss_G_AB['pca']) + sum(self.loss_G_BA['pca'])

        self.loss_g = self.loss_gan + lam * self.loss_cycle + lam_pca * self.loss_pca
        self.loss_g.backward()

    def get_current_visuals(self, inputs, idx=None):
        Boundary = []
        Channel = []
        Edge = []
        pic_name = ['ABA', 'BAB']
        G_BA = self.G_net['G_BA']
        for ii in range(self.multi_n):
            G_AB = self.G_net['G_AB{}'.format(ii)]
            net1 = [G_AB, G_BA]
            net2 = [G_BA, G_AB]

            with torch.no_grad(): 
                for nn, x, net, in_net in zip(pic_name, [inputs[0], inputs[1][ii]], net1, net2):
                    x1 = net(x )
                    x2 = in_net(x1 )

                    boundary_name = 'pp{}_{}{}'.format(ii, nn, ii)
                    img_cat = torch.cat((x1, x2), 0)
                    pt_cat = self.Align(self._get_variable(img_cat))
                    img_cat = self._gen_heat_map(img_cat)
                    tmp_r = 0 if self.opt.default_r==0 else (idx/self.opt.display_freq)%2
                    img_cat = self._plot_pt(img_cat, pt_cat.data, tmp_r)
                    img_cat = vutils.make_grid(img_cat, nrow=self.nrow).unsqueeze(0)
                    img_cat = util.tensor2im(img_cat.data, 'vutils')
                    Boundary.append((boundary_name, img_cat))
                    #img_channel
                    channel_name = 'pp{}_channel_{}{}'.format(ii, nn, ii)
                    img_channel = self._gen_channel(x1)
                    img_channel = vutils.make_grid(img_channel, nrow=15).unsqueeze(0)
                    img_channel = util.tensor2im(img_channel.data, 'vutils')
                    Channel.append((channel_name, img_channel))
                    #img_edge
                    edge_name = 'pp{}_edge_{}{}'.format(ii, nn, ii)
                    img_edge = self._gen_channel(self._gen_Edge(x1))
                    img_edge = vutils.make_grid(img_edge, nrow=15).unsqueeze(0)
                    img_edge = util.tensor2im(img_edge.data, 'vutils')
                    Edge.append((edge_name, img_edge))
        return OrderedDict(Boundary), OrderedDict(Channel), OrderedDict(Edge) 

    def _get_variable(self, inputs):
        if len(self.opt.gpu_ids) > 0:
            out = inputs.cuda()
        else:
            out = inputs
        out.requires_grad_()
        return out


    def _inver_trans(self, inputs):
        dim = inputs.size()
        mean = torch.FloatTensor(dim[0], dim[1], dim[2], dim[3]).fill_(0.5).cuda()
        var = torch.FloatTensor(dim[0], dim[1], dim[2], dim[3]).fill_(0.5).cuda()
        return inputs*var + mean


    def _gen_channel(self, heat_map):
        # N * 15 *64 * 64
        img_channel = [xx.view(15, 1, 64, 64) for xx in heat_map]
        return torch.cat(tuple(img_channel), 0)


    def _gen_heat_map(self, heat_map):
        heat_max = torch.max(heat_map, 1)[0].view(heat_map.size(0), 1, heat_map.size(2), heat_map.size(3))
        return heat_max


    def _pca_align_loss(self, inputs, target):
        tmp1 = self.PCA(self.Align(inputs))
        tmp2 = self.PCA(self.Align(target)).detach()
        return self.alignLoss(tmp1[:, :self.opt.pca_dim], tmp2[:, :self.opt.pca_dim])


    def _plot_pt(self, pic, pt, r=0):
        col = [1]
        #r = 1
        size_ = pic.size()
        #print(size_)
        xx = pt[:, ::2]
        yy = pt[:, 1::2]
        out = pic
        for kk in range(int(size_[0])):
            for ii in range(int(r)):
                xx_tmp = (xx[kk] + ii).clamp_(0, size_[2]-1)
                yy_tmp = (yy[kk] + ii).clamp_(0,  size_[3]-1)
                for xx_, yy_ in zip(xx_tmp, yy_tmp):
                    for jj, cc in enumerate(col):
                        out[kk, jj, int(yy_), int(xx_)] = cc
        return out

    
    def _save_valid_pic(self, valid_x_A, valid_x_B):
        valid_pt_A = self.inv_PCA(self.PCA(self.Align(valid_x_A)))
        vutils.save_image(self._plot_pt(self._gen_heat_map(valid_x_A.detach()), valid_pt_A.detach()), '{}/valid_x_A.png'.format(self.model_dir))
        #channel
        valid_channel = self._gen_channel(valid_x_A.detach())
        vutils.save_image(valid_channel, '{}/valid_channel_A.png'.format(self.model_dir), nrow=15)
        #edge
        vutils.save_image(self._gen_channel(self._gen_Edge(valid_x_A).detach()), '{}/valid_edge_A.png'.format(self.model_dir), nrow=15)

        for ii, bb in enumerate(valid_x_B):
            bb_pt = self.inv_PCA(self.PCA(self.Align(bb)))
            vutils.save_image(self._plot_pt(self._gen_heat_map(bb.detach()), bb_pt.detach()), '{}/valid_x_B{}.png'.format(self.model_dir, ii))    

    def get_current_errors(self):
        return OrderedDict([('D_real_A', self.loss_d_real_A.data.item()),
                            ('D_fake_A', sum(self.loss_d_fake_A).data.item()),
                            ('D_real_B', sum(self.loss_d_real_B).data.item()),
                            ('D_fake_B', sum(self.loss_d_fake_B).data.item()),
                            ('Cycle_AB', sum(self.loss_G_AB['cycle']).data.item()),
                            ('Cycle_BA', sum(self.loss_G_BA['cycle']).data.item()),
                            ('GAN_AB', sum(self.loss_G_AB['gan']).data.item()),
                            ('GAN_BA', sum(self.loss_G_BA['gan']).data.item()),
                            ('PCA_AB', sum(self.loss_G_AB['pca']).data.item()),
                            ('PCA_BA', sum(self.loss_G_BA['pca']).data.item())
                            ])

    def _numpy2cuda(self, inputs):
        out =  torch.FloatTensor(inputs).cuda()
        return out

    def _init_Align(self):
        #Align
        self.Align = networks.resnet18_face_alignment(self.gpu_ids)
        #print(self.Align.state_dict().keys())
        self.alignLoss = torch.nn.L1Loss()
        self.alignLoss.cuda()
        self.Align.cuda(self.gpu_ids[0])
        tmp = torch.load('{}/Align_40000.pth'.format(self.opt.pretrain_root))
        module_dict = {'.'.join(kk.split('.')[1:]) : vv for kk, vv in tmp.items()}
        self.Align.load_state_dict(module_dict)

    def _init_Bound(self):
        #Bound
        self.Bound = networks.netBound(self.gpu_ids)
        self.Bound.cuda(self.gpu_ids[0])
        #print(self.Bound.state_dict().keys())
        tmp = torch.load('{}/v8_net_boundary_detection.pth'.format(self.opt.pretrain_root))
        #module_dict = {'model.' + kk : vv for kk, vv in tmp.items() if 'num_batches' not in kk}
        #print(module_dict.keys())
        module_dict = {kk : vv for kk, vv in tmp.items()}
        self.Bound.load_state_dict(module_dict)

    def _init_PCA(self):
        # PCA
        pca_root = '{}/pca_init'.format(self.opt.pretrain_root)
        self.PCA = networks.PCA(212, 212, self.gpu_ids)
        map_ = {}
        for ii in ['weight', 'bias']:
            tmp = np.loadtxt('{}/{}_{}.txt'.format(pca_root, ii, 212))
            map_['pca.0.{}'.format(ii)] = self._numpy2cuda(tmp)

        self.PCA.load_state_dict(map_)
        self.PCA.cuda(self.gpu_ids[0])
        # inverse PCA
        self.inv_PCA = networks.PCA(212, 212, self.gpu_ids)
        map_ = {}
        for ii in ['weight', 'bias']:
            tmp = np.loadtxt('{}/inverse_{}_{}.txt'.format(pca_root, ii, 212))
            map_['pca.0.{}'.format(ii)] = self._numpy2cuda(tmp)

        self.inv_PCA.load_state_dict(map_)
        self.inv_PCA.cuda(self.gpu_ids[0])

    def _init_Edge(self):
        # Edge
        self.Edge = networks.Edge(self.gpu_ids)
        #print(self.Edge.state_dict().keys())
        edge_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).reshape(1, 1, 3, 3)
        edge_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).reshape(1, 1, 3, 3)

        map_ = {'edge.0.weight': self._numpy2cuda(edge_x),
                'edge.1.weight': self._numpy2cuda(edge_y)}

        self.Edge.load_state_dict(map_)
        self.Edge.cuda(self.gpu_ids[0])

    def _gen_Edge(self, inputs): # laplas or sobel gradient
        #inputs: N*15*64*64  conv: 3*3
        size_ = inputs.size()
        tmp = [self.Edge(inputs[:, ii, :, :].contiguous().view(size_[0], 1, size_[2], size_[3])) for ii in range(size_[1])]
        return torch.cat(tmp, 1)

    def save(self, step):
        for name, m in tuple(self.G_net.items()) + tuple(self.D_net.items()):
            file_name = '{}/{}_{}.pth'.format(self.opt.model_dir, name, step)
            torch.save(m.state_dict(), file_name)
