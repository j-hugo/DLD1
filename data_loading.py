import numpy as np
import matplotlib.pyplot as plt
import os
import random
import PIL
import json

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
# preprocessing, data iterators
# informative description of content

# get_undersample_files() and get_oversample_files() are functions used in the dataset class that return a list of files. In this list 
# slices with and without cancer tissue are represented in the same quantity.
# get_undersample_files() will return all files of the minority class and randomly choose files from the majority class
# get_oversample_files() will return all files of the majority class and randomly choose files from the minority class
# both functions return a list of files with exactly equal number of image slices with or without cancer tissue

# undersample
def get_undersample_files(json_dir):
  with open(json_dir) as json_file:
    data_index = json.load(json_file)

  index_with_cancer = [k for k,v in data_index.items() if (v['cancer'] == True) & (v['subset']=='train')]
  index_no_cancer = [k for k,v in data_index.items() if (v['cancer'] == False) & (v['subset']=='train')]

  # randomly draw indices from set of indices of slices without cancer
  # same number of slices with and without cancer tissue
  rand_index_no_cancer = random.choices(index_no_cancer,k=len(index_with_cancer))

  image_files, label_files = [], []

  # add file names of images and labels to list
  for slice_index in rand_index_no_cancer:
    image = 'image_'+slice_index + '.npy'
    label = 'label_'+slice_index + '.npy'
    image_files.append(image)
    label_files.append(label)

  for slice_index in index_with_cancer:
    image = 'image_'+slice_index + '.npy'
    label = 'label_'+slice_index+ '.npy'
    image_files.append(image)
    label_files.append(label)
  
  return(image_files,label_files)

# oversample
def get_oversample_files(json_dir):

  with open(json_dir) as json_file:
    data_index = json.load(json_file)

  index_with_cancer = [k for k,v in data_index.items() if (v['cancer'] == True) & (v['subset']=='train')]
  index_no_cancer = [k for k,v in data_index.items() if (v['cancer'] == False) & (v['subset']=='train')]

  # randomly draw indices from set of indices of slices without cancer
  # same number of slices with and without cancer tissue
  rand_index_with_cancer = random.choices(index_with_cancer,k=len(index_no_cancer))

  image_files, label_files = [], []

  # add file names of images and labels to list

  for slice_index in rand_index_with_cancer:
    image = 'image_'+slice_index + '.npy'
    label = 'label_'+slice_index + '.npy'
    image_files.append(image)
    label_files.append(label)

  for slice_index in index_no_cancer:
    image = 'image_'+slice_index + '.npy'
    label = 'label_'+slice_index+ '.npy'
    image_files.append(image)
    label_files.append(label)
  
  return(image_files,label_files)

# only_tumor_files() returns a list of all files that contain cancer tissue.
# no files without cancer tissue will be returned
def only_tumor_files(json_dir):
  with open(json_dir) as json_file:
    data_index = json.load(json_file)

  index_with_cancer = [k for k,v in data_index.items() if (v['cancer'] == True) & (v['subset']=='train')]

  image_files, label_files = [], []

  # add file names of images and labels to list
  for slice_index in index_with_cancer:
    image = 'image_'+slice_index + '.npy'
    label = 'label_'+slice_index+ '.npy'
    image_files.append(image)
    label_files.append(label)
  
  return(image_files,label_files)

# get_original_dataset() returns a dataset without any sampling method
def get_original_dataset(json_dir, test):
    with open(json_dir) as json_file:
        data_index = json.load(json_file)

    if test is True:
        file_index = [k for k, v in data_index.items() if v['subset'] == 'test']
    else:
        file_index = [k for k,v in data_index.items() if v['subset'] == 'train']

    image_files, label_files = [], []

    # add file names of images and labels to list
    for slice_index in file_index:
        image = 'image_' + slice_index + '.npy'
        label = 'label_' + slice_index + '.npy'
        image_files.append(image)
        label_files.append(label)

    return(image_files, label_files)

# dataset class for primary colon cancer dataset
class ColonDataset(Dataset):
    """Colon Cancer dataset."""
    def __init__(self, image_dir, label_dir, json_dir, image_size, torch_transform, balance_dataset=None, test=None):
        """
        Args:
            image_dir: Path to image folder.
            label_dir: Path to label folder.
            csv_dir: Path to csv file, which gives information whether slice contains annotated cancer pixels.
            balance_dataset (optional): options to create a dataset with balanced numbers of slices
                containing cancer tissue or not containing cancer
                'oversample': uniformly draws samples from minority class to reach equal size
                'undersample': uniformly draws samples from majority class to reach equal size
                'only_tumor': only includes slices with cancer tissue
                None: no balance method is applied
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.json_dir = json_dir
        self.image_size = image_size
        self.test = test
        self.balance_dataset = balance_dataset
        self.torch_transform = torch_transform
        if self.test is None:
            if self.balance_dataset == "undersample":
                self.image_files, self.label_files = get_undersample_files(self.json_dir)
            if self.balance_dataset == "oversample":
                self.image_files, self.label_files = get_oversample_files(self.json_dir)
            if self.balance_dataset == 'only_tumor':
                self.image_files, self.label_files = only_tumor_files(self.json_dir)
        if (self.balance_dataset is None) or (self.test is True):
            self.image_files, self.label_files = get_original_dataset(self.json_dir, self.test)

    def __len__(self):
      return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_path = os.path.join(self.image_dir,
                                  self.image_files[idx])
        label_path = os.path.join(self.label_dir,
                                  self.label_files[idx])
        
        image = np.load(image_path)
        label = np.load(label_path)
        if self.torch_transform:
            x, y = self.transform(image, label)

        return [x, y]

    def transform(self, image, label):
      # to PIL
      image = PIL.Image.fromarray(image)
      label = PIL.Image.fromarray(label)

      # Resize
      if self.test == None:
        image = TF.resize(image, size=(self.image_size+44, self.image_size+44))
        label = TF.resize(label, size=(self.image_size+44, self.image_size+44))
      else:
        image = TF.resize(image, size=(self.image_size, self.image_size))
        label = TF.resize(label, size=(self.image_size, self.image_size))

      # Random crop
      if self.test == None:
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(self.image_size, self.image_size))
        image = TF.crop(image, i, j, h, w)
        label = TF.crop(label, i, j, h, w)

      # Random horizontal flipping
      if self.test == None:
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

      # Random vertical flipping
      if self.test == None:
        if random.random() > 0.5:
            image = TF.vflip(image)
            label = TF.vflip(label)

      # Transform to tensor
      image = torch.from_numpy(np.array(image)) # to_tensor: /opt/conda/conda-bld/pytorch_1587428094786/work/torch/csrc/utils/tensor_numpy.cpp:141: UserWarning: The given NumPy array is not writeable, and PyTorch does not support non-writeable tensors. 
      image = image.unsqueeze(0).type(torch.FloatTensor)
      label = torch.from_numpy(np.array(np.expand_dims(label, 0))).type(torch.FloatTensor)
      # Normalize
      image = TF.normalize(image, mean=(-531.28,), std=(499.68,))


      return image, label
      
if __name__ == "__main__":
  path = os.path.abspath(".") + "/data/"
  image_dir = path+'npy_images'
  label_dir = path+'npy_labels'
  json_dir = path+'data_index_subsets.json'

  data = ColonDataset(image_dir,label_dir,json_dir, 256,torch_transform=True, balance_dataset="only_tumor")
  single_example = data[0]
  print(single_example)
  print(f"Plotting slice of Image")
  plt.imshow(single_example[0][0], cmap='gray')
  plt.imshow(single_example[1][0],alpha=0.3)
  plt.axis('off')
  plt.show()
