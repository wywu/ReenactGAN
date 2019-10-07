import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.test_model import TestModel
from util.visualizer import Visualizer
import torch

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
model = TestModel(opt)
visualizer = Visualizer(opt)

print('#testing images = %d' % dataset_size)
for i, data in enumerate(dataset):
    model.set_input(data)
    model.test()
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('%04d: process image... %s' % (i, img_path))
    visualizer.save_images_split(visuals, img_path)

print('Done!')
