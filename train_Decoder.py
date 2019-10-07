import time
from options.decoder_options import DecoderOptions
from data.data_loader import CreateDataLoader
from models.face2boundary2face_model import Face2Boundary2FaceModel
from util.visualizer import Visualizer

opt = DecoderOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)
print("len(dataset):{}".format(len(dataset)))

model = Face2Boundary2FaceModel(opt)
visualizer = Visualizer(opt)
total_steps = 0

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(dataset):
        # print ("!i_out:.{}".format(i))
        iter_start_time = time.time()
        visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()

        # model.visualisation_check()

        if total_steps % opt.display_freq == 0:
            save_result = total_steps % opt.update_html_freq == 0
            visualizer.display_current_results([model.get_current_visuals()], epoch, save_result)

        if total_steps % opt.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)

        if total_steps % opt.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' %
                  (epoch, total_steps))
            model.save('latest')

    if epoch % opt.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' %
              (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

    ave_time_cost = time.time() - epoch_start_time
    left_time_cost = (opt.niter + opt.niter_decay - epoch) * ave_time_cost
    print('Speed: %fs / epoch.  %d:%d (H:M) to go.' % (ave_time_cost, int(left_time_cost/3600), int((left_time_cost-3600*int(left_time_cost/3600))/60)))

    model.update_learning_rate()
