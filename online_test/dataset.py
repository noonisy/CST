import json
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader


# for COCO or nocaps
class Online_Test(Dataset):
    def __init__(self, feat_path, ann_file, max_detections=49):
        super(Online_Test, self).__init__()
        with open(ann_file, 'r') as f:
            self.info = json.load(f)
        self.images = self.info['images']
        self.feat_path = feat_path
        self.f = h5py.File(feat_path, 'r')
        self.max_detections = max_detections

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id = self.images[idx]['id']
        precomp_data = self.f['%d_grids' % image_id][()]
        delta = self.max_detections - precomp_data.shape[0]
        if delta > 0:
            precomp_data = np.concatenate([precomp_data, np.zeros((delta, precomp_data.shape[1]))], axis=0)
        elif delta < 0:
            precomp_data = precomp_data[:self.max_detections]

        return int(image_id), precomp_data
