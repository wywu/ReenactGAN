import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt, pic_path=None):
    dataset = None
    if opt.dataset_mode == 'single':
        from data.datasets import SingleDataset
        dataset = SingleDataset(opt)
    elif opt.dataset_mode == 'aligned_face2boundary2face':
        from data.datasets import AlignedFace2Boudnary2Face
        dataset = AlignedFace2Boudnary2Face(opt)
    elif opt.dataset_mode == 'transformer':
        from data.datasets import TransformerDataset
        dataset = TransformerDataset(opt, pic_path)
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def __init__(self, opt, pic_path=None):
        super(CustomDatasetDataLoader, self).__init__(opt)
        self.dataset = CreateDataset(opt, pic_path)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads),
            drop_last=opt.drop_last)
        self.shape = len(self.dataset)

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            # print ("!i_in:.{}".format(i))
            if i >= self.opt.max_dataset_size:
                break
            yield data
