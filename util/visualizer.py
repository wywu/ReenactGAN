import numpy as np
import os
import ntpath
import time
from . import util
from . import html
from time import strftime

class Visualizer():
    def __init__(self, opt):
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.opt = opt
        self.saved = False

        if self.use_html:
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print(strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '] ' + 'create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        if opt.isTrain:
            self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
            with open(self.log_name, "a") as log_file:
                now = time.strftime("%c")
                log_file.write('================ Training Loss (%s) ================\n' % now)

        if not opt.isTrain:
            self.real_F1_dir = os.path.join(opt.save_root_path, opt.real_F1_path, 'Image')
            self.Boundary_dir = os.path.join(opt.save_root_path, opt.Boundary_path, 'Image')
            self.Boundary_transformed_dir = os.path.join(opt.save_root_path, opt.Boundary_transformed_path, 'Image')
            self.fake_F2_dir = os.path.join(opt.save_root_path, opt.fake_F2_path, 'Image')
            if not os.path.exists(self.real_F1_dir):
                os.makedirs(self.real_F1_dir)
            if not os.path.exists(self.Boundary_dir):
                os.makedirs(self.Boundary_dir)
            if not os.path.exists(self.Boundary_transformed_dir):
                os.makedirs(self.Boundary_transformed_dir)
            if not os.path.exists(self.fake_F2_dir):
                os.makedirs(self.fake_F2_dir)

    def reset(self):
        self.saved = False

    # |visuals|: dictionary of images to display or save
    def display_current_results(self, visuals, epoch, save_result, transformer=False):
        if self.use_html and (save_result or not self.saved):  # save images to a html file
            self.saved = True
            #visuals = list(visuals)
            for visual in visuals:
                for label, image_numpy in visual.items():
                    img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.png' % (epoch, label))
                    util.save_image(image_numpy, img_path)
            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, reflesh=1)
            if not transformer:
                epoch_step = -1
            else:
                epoch_step = -self.opt.update_html_freq
            for n in range(epoch, 0, epoch_step):
                webpage.add_header('epoch [%d]' % n)
                for visual in visuals:
                    ims = []
                    txts = []
                    links = []
                    for label, image_numpy in visual.items():
                        img_path = 'epoch%.3d_%s.png' % (n, label)
                        ims.append(img_path)
                        txts.append(label)
                        links.append(img_path)
                    if 'channel' in label or 'edge' in label:
                        webpage.add_images(ims, txts, links, width=992, vertical=True)
                    else:
                        webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    # errors: same format as |errors| of plotCurrentErrors
    def print_current_errors(self, epoch, i, errors, t, transformer=False):
        if not transformer:
            message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        else:
            message = '(max iters: %d, iters: %d, time: %.3f) ' % (epoch, i, t)
        for k, v in errors.items():
            message += '%s: %.6f ' % (k, v)

        print(strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + '] ' + message)
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)

    # save image to the disk
    def save_images(self, webpage, visuals, image_path, vertical=False):
        image_dir = webpage.get_image_dir()
        short_path = ntpath.basename(image_path[0])
        name = os.path.splitext(short_path)[0]

        webpage.add_header(name)
        ims = []
        txts = []
        links = []

        for label, image_numpy in visuals.items():
            image_name = '%s_%s.png' % (name, label)
            save_path = os.path.join(image_dir, image_name)
            util.save_image(image_numpy, save_path)

            ims.append(image_name)
            txts.append(label)
            links.append(image_name)
        webpage.add_images(ims, txts, links, width=self.win_size, vertical=vertical)

    # save image to the split folders
    def save_images_split(self, visuals, image_path):
        name = ntpath.basename(image_path[0])

        for label, image_numpy in visuals.items():
            if label == 'real_F1':
                save_path = os.path.join(self.real_F1_dir, name)
            elif label == 'boundary_map':
                save_path = os.path.join(self.Boundary_dir, name)
            elif label == 'boundary_map_transformed':
                save_path = os.path.join(self.Boundary_transformed_dir, name)
            elif label == 'fake_F2':
                save_path = os.path.join(self.fake_F2_dir, name)
            util.save_image(image_numpy, save_path)
