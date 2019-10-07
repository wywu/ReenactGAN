import torch
import sys
import time
import torchvision.utils as vutils

from models.transformer_model import TransformerModel
from options.transformer_options import TransformerOptions
from data.data_loader import CreateDataLoader
from util.util import create_path_list
from util.visualizer import Visualizer

opt = TransformerOptions().parse()
torch.manual_seed(opt.random_seed)
torch.cuda.manual_seed(opt.random_seed)

prefix = ['Emmanuel_Macron', 'Kathleen', 'Jack_Ma', 'Theresa_May', 'Donald_Trump']
#prefix = ['P_Emmanuel/video_0', 'P_Kathleen/video_0', 'P_MaYun/video_0', 'P_Theresa/video_0', 'P_Trump/video_0']
A_path, B_path = create_path_list(opt, prefix)

#init data loader
A_loader = CreateDataLoader(opt, A_path)
B_loader = [CreateDataLoader(opt, [pp]) for pp in B_path]

A_loader_iter = enumerate(A_loader.load_data())
B_loader_iter = [enumerate(bb.load_data()) for bb in B_loader]

print('A: {}  B: {}\n'.format(A_loader.shape, ' '.join([str(ii.shape) for ii in B_loader])))
model = TransformerModel(opt, len(B_loader))
visualizer = Visualizer(opt)

#valid
valid_x_A = model._get_variable(next(A_loader_iter)[-1])
vutils.save_image(model._inver_trans(valid_x_A.detach()), '{}/face.jpg'.format(opt.model_dir), nrow=model.nrow)
with torch.no_grad(): 
    valid_x_A = model.Bound(valid_x_A)[1]
    valid_x_B = [model.Bound(model._get_variable(next(bb)[-1]))[1] for bb in B_loader_iter]
model._save_valid_pic(valid_x_A, valid_x_B)


#start trainning
for step in range(opt.max_step):
    try:
        x_A = next(A_loader_iter)[-1]
    except StopIteration:
        A_loader_iter = enumerate(A_loader.load_data())
        x_A = next(A_loader_iter)[-1]

    x_B = []
    for ii, bb in enumerate(B_loader_iter):
        try:
            x_B.append(next(bb)[-1])
        except StopIteration:
            B_loader_iter[ii] = enumerate(B_loader[ii].load_data())
            x_B.append(next(B_loader_iter[ii])[-1])

    if x_A.size(0) != x_B[0].size(0):
        print("[!] Sampled dataset from A and B have different # of data. Try resampling...")
        continue

    iter_start_time = time.time()
    visualizer.reset()
    model.set_input(x_A, x_B)
    model.optimize_parameters()

    if step % opt.display_freq == 0:
        save_result = step % opt.update_html_freq == 0
        Boundary, Channel, Edge = model.get_current_visuals([valid_x_A, valid_x_B], idx=step)
        visualizer.display_current_results([Boundary, Channel, Edge], step, save_result, transformer=True)

    if step % opt.print_freq == 0:
        errors = model.get_current_errors()
        t = (time.time() - iter_start_time) / opt.batchSize
        visualizer.print_current_errors(opt.max_step, step, errors, t)

    if step % opt.save_epoch_freq == 0:
        print("[*] Save models to {}...".format(opt.model_dir))
        model.save(step)


