import torch.utils.data
from data.base_data_loader import BaseDataLoader
from torch.utils.data.distributed import DistributedSampler

def CreateDataset(opt,is_val=False,keyframes=None):
    dataset = None
    if opt.dataset_mode == 'temporal':
        from data.temporal_dataset import TemporalDataset
        dataset = TemporalDataset()   
    elif opt.dataset_mode == 'face':
        from data.face_dataset import FaceDataset
        dataset = FaceDataset() 
    elif opt.dataset_mode == 'pose':
        from data.pose_dataset import PoseDataset
        dataset = PoseDataset() 
    elif opt.dataset_mode == 'test' or opt.dataset_mode == 'val':
        from data.test_dataset import TestDataset
        dataset = TestDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    if opt.dataset_mode == 'face':
        dataset.initialize(opt,is_val=is_val,keyframes=keyframes)
    else:
        dataset.initialize(opt,keyframes=keyframes)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt, is_val=False,keyframes=None):

        BaseDataLoader.initialize(self, opt)
        
        self.dataset = CreateDataset(opt,is_val=is_val,keyframes=keyframes)
        batchSize = opt.batchSize
        self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=batchSize,
                shuffle=not opt.serial_batches,
                num_workers=int(opt.nThreads),
                drop_last=True
                )


    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
