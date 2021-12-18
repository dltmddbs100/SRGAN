import cv2
from PIL import Image
import glob

import torch
from torch.utils.data.dataset import Dataset
from torchvision.transforms import ToTensor, Resize, Compose


class TrainDataset(Dataset):
  def __init__(self, dataset_dir):
    super(TrainDataset, self).__init__()
    self.train_data=sorted(glob.glob(dataset_dir+'TRAIN_LABEL/*'))    

    self.transform_train=Compose([ToTensor(),Resize((153,204), interpolation=Image.BICUBIC)])
    self.transform_target=Compose([ToTensor(),Resize((306,408), interpolation=Image.BICUBIC)])

  def __getitem__(self, index):
    image = cv2.imread(self.train_data[index])

    return torch.clamp(self.transform_train(image),0,1),torch.clamp(self.transform_target(image),0,1)

  def __len__(self):
     return len(self.train_data)

class TestDataset(Dataset):
  def __init__(self, dataset_dir):
    super(TestDataset, self).__init__()
    self.test_data=sorted(glob.glob(dataset_dir+'TEST_INPUT/*'))    

    self.transform_lr=Compose([ToTensor(),Resize((153,204))])
    self.transform_hr=Compose([ToTensor(),Resize((306,408))])

  def __getitem__(self, index):
    image = cv2.imread(self.test_data[index])

    return torch.clamp(self.transform_lr(image),0,1), torch.clamp(self.transform_hr(image),0,1)

  def __len__(self):
     return len(self.test_data)