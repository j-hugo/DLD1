from keras.utils import to_categorical

import numpy as np
import matplotlib.pyplot as plt
import os
import random
import PIL

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

# preprocessing, data iterators
# informative description of content

# get_undersample_files() and get_upsample_files() are functions used in the dataset class that return a list of files. In this list 
# slices with and without cancer tissue are represented in the same quantity.
# get_undersample_files() will return all files of the minority class and randomly choose files from the majority class
# get_upsample_files() will return all files of the majority class and randomly choose files from the minority class
# both functions return a list of files with exactly equal number of image slices with or without cancer tissue

# undersample
def get_undersample_files(csv_file):

  # import csv file as np array
  csv_list = np.genfromtxt(csv_file,delimiter=',')

  index_with_cancer = np.where(csv_list[:,2]==1)[0] # indeces in csv table of slices containing cancer tissue
  index_no_cancer = np.where(csv_list[:,2]==0)[0] # indeces in csv table of slices not containing cancer tissue

  # randomly draw indices from set of indices of slices without cancer
  # same number of slices with and without cancer tissue
  rand_index_no_cancer = np.random.choice(index_no_cancer,size=index_with_cancer.shape[0])

  image_files, label_files = [], []

  # add file names of images and labels to list
  for image_numb, slice_numb in csv_list[rand_index_no_cancer,0:2]:
    image = 'image_'+str(int(image_numb)).zfill(3) + '_' + str(int(slice_numb)).zfill(3) + '.npy'
    label = 'label_'+str(int(image_numb)).zfill(3) + '_' + str(int(slice_numb)).zfill(3) + '.npy'
    image_files.append(image)
    label_files.append(label)

  for image_numb, slice_numb in csv_list[index_with_cancer,0:2]:
    image = 'image_'+str(int(image_numb)).zfill(3) + '_' + str(int(slice_numb)).zfill(3) + '.npy'
    label = 'label_'+str(int(image_numb)).zfill(3) + '_' + str(int(slice_numb)).zfill(3) + '.npy'
    image_files.append(image)
    label_files.append(label)
  
  return(image_files,label_files)

# upsample
def get_upsample_files(csv_file):

  # import csv file as np array
  csv_list = np.genfromtxt(csv_file,delimiter=',')

  index_with_cancer = np.where(csv_list[:,2]==1)[0] # indeces in csv table of slices containing cancer tissue
  index_no_cancer = np.where(csv_list[:,2]==0)[0] # indeces in csv table of slices not containing cancer tissue

  # randomly draw indices from set of indices of slices without cancer
  # same number of slices with and without cancer tissue
  rand_index_with_cancer = np.random.choice(index_with_cancer,size=index_no_cancer.shape[0])

  image_files, label_files = [], []

  # add file names of images and labels to list
  for image_numb, slice_numb in csv_list[rand_index_with_cancer,0:2]:
    image = 'image_'+str(int(image_numb)).zfill(3) + '_' + str(int(slice_numb)).zfill(3) + '.npy'
    label = 'label_'+str(int(image_numb)).zfill(3) + '_' + str(int(slice_numb)).zfill(3) + '.npy'
    image_files.append(image)
    label_files.append(label)

  for image_numb, slice_numb in csv_list[index_no_cancer,0:2]:
    image = 'image_'+str(int(image_numb)).zfill(3) + '_' + str(int(slice_numb)).zfill(3) + '.npy'
    label = 'label_'+str(int(image_numb)).zfill(3) + '_' + str(int(slice_numb)).zfill(3) + '.npy'
    image_files.append(image)
    label_files.append(label)
  
  return(image_files,label_files)

# only_tumor_files() returns a list of all files that contain cancer tissue.
# no files without cancer tissue will be returned

def only_tumor_files(csv_file):
  # import csv file as np array
  csv_list = np.genfromtxt(csv_file,delimiter=',')

  index_with_cancer = np.where(csv_list[:,2]==1)[0] # indeces in csv table of slices containing cancer tissue

  image_files, label_files = [], []

  # add file names of images and labels to list
  for image_numb, slice_numb in csv_list[index_with_cancer,0:2]:
    image = 'image_'+str(int(image_numb)).zfill(3) + '_' + str(int(slice_numb)).zfill(3) + '.npy'
    label = 'label_'+str(int(image_numb)).zfill(3) + '_' + str(int(slice_numb)).zfill(3) + '.npy'
    image_files.append(image)
    label_files.append(label)
  
  return(image_files,label_files)

# dataset class for primary colon cancer dataset

class ColonDataset(Dataset):
    """Colon Cancer dataset."""

    def __init__(self, image_dir, label_dir, csv_dir, image_size, torch_transform, balance_dataset=None, test=None):
        """
        Args:
            image_dir: Path to image folder.
            label_dir: Path to label folder.
            csv_dir: Path to csv file, which gives information whether slice contains annotated cancer pixels.
            balance_dataset (optional): options to create a dataset with balanced numbers of slices
                containing cancer tissue or not containing cancer
                'upsample': uniformly draws samples from minority class to reach equal size
                'undersample': uniformly draws samples from majority class to reach equal size
                'only_tumor': only includes slices with cancer tissue
                None: no balance method is applied
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.csv_dir = csv_dir
        self.image_size = image_size
        self.test = test
        self.balance_dataset = balance_dataset
        self.torch_transform = torch_transform
        if self.balance_dataset == "undersample":
          self.image_files, self.label_files = get_undersample_files(self.csv_dir)
        if self.balance_dataset == "upsample":
          self.image_files, self.label_files = get_upsample_files(self.csv_dir)
        if self.balance_dataset == 'only_tumor':
          self.image_files, self.label_files = only_tumor_files(self.csv_dir)
        if self.balance_dataset == None:
          self.image_files = os.listdir(self.image_dir)
          self.label_files = os.listdir(self.label_dir)

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
      label = torch.from_numpy(np.moveaxis(to_categorical(label, num_classes=2), -1, 0)).type(torch.FloatTensor)

      # Normalize
      image = TF.normalize(image, mean=-531.28, std=499.68)

      return image, label
      
if __name__ == "__main__":
  path = os.path.abspath(".") + "/"
  image_dir = path+'npy_images'
  label_dir = path+'npy_labels'
  csv_dir = path+'contains_cancer_index.csv'

  data = ColonDataset(image_dir,label_dir,csv_dir, 256,torch_transform=True, balance_dataset="only_tumor")
  single_example = data[1]
  print(f"Plotting slice of Image")
  plt.imshow(single_example[0][0], cmap='gray')
  plt.imshow(single_example[1][1],alpha=0.3)
  plt.axis('off')
  plt.show()