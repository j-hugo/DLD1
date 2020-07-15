import os
import numpy as np
import nibabel
import sys
from glob import glob
import csv
import argparse

def makedirs(path):
    os.makedirs(path, exist_ok=True)

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def convert_to_npy_train(args):
    train_data_path = os.path.join(args.datapath, 'imagesTr')
    label_data_path = os.path.join(args.datapath, 'labelsTr')
    images = sorted(os.listdir(train_data_path))
    labels = sorted(os.listdir(label_data_path))
    
    split = int(len(images)*args.split)
    images = images[:-split]
    labels = labels[:-split]

    image_saved_path =args.path +'npy_train_images/'
    try:
        os.mkdir(image_saved_path)
    except OSError:
        print ("Creation of the directory %s failed" % image_saved_path)
    else:
        print ("Successfully created the directory %s " % image_saved_path)
        
    label_saved_path = args.path + 'npy_train_labels/'
    try:
        os.mkdir(label_saved_path)
    except OSError:
        print ("Creation of the directory %s failed" % label_saved_path)
    else:
        print ("Successfully created the directory %s " % label_saved_path)
    contains_cancer = []

    for img, label in zip(images,labels):
        # Load 3D training image
        image_number = str(''.join(filter(str.isdigit, img))).zfill(3)
        training_image = nibabel.load(os.path.join(train_data_path, img))
        training_label = nibabel.load(os.path.join(label_data_path, label))

        for k in range(training_label.shape[2]):
            # axial cuts are made along the z axis (slice) 
            image_2d = np.array(training_image.get_fdata()[:, :, k], dtype='int16') # I checked: all values in the nifti files were integers, ranging from -1024 to approx 3000
            label_2d = np.array(training_label.get_fdata()[:, :, k], dtype='uint8') # only contains 1s and 0s
            slice_number = str(k).zfill(3)
            if len(np.unique(label_2d))!=1:
              contains_cancer.append([image_number,slice_number,1])
            else:
              contains_cancer.append([image_number,slice_number,0])

            np.save((image_saved_path+'image_{}_{}.npy'.format(image_number,slice_number)), image_2d)
            np.save((label_saved_path +'label_{}_{}.npy'.format(image_number,slice_number)), label_2d)
            print(f'Saved image {image_number}, slice {slice_number}')
    
    with open(args.path+"contains_cancer_train_index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(contains_cancer)

def convert_to_npy_test(args):
    train_data_path = os.path.join(args.datapath, 'imagesTr')
    label_data_path = os.path.join(args.datapath, 'labelsTr')
    images = sorted(os.listdir(train_data_path))
    labels = sorted(os.listdir(label_data_path))
    
    split = int(len(images)*args.split)
    images = images[-split:]
    labels = labels[-split:]
    
    image_saved_path = args.path +'npy_test_images/'
    try:
        os.mkdir(image_saved_path)
    except OSError:
        print ("Creation of the directory %s failed" % image_saved_path)
    else:
        print ("Successfully created the directory %s " % image_saved_path)
        
    label_saved_path = args.path + 'npy_test_labels/'
    try:
        os.mkdir(label_saved_path)
    except OSError:
        print ("Creation of the directory %s failed" % label_saved_path)
    else:
        print ("Successfully created the directory %s " % label_saved_path)
    contains_cancer = []

    for img, label in zip(images,labels):
        # Load 3D training image
        image_number = str(''.join(filter(str.isdigit, img))).zfill(3)
        training_image = nibabel.load(os.path.join(train_data_path, img))
        training_label = nibabel.load(os.path.join(label_data_path, label))

        for k in range(training_label.shape[2]):
            # axial cuts are made along the z axis (slice) 
            image_2d = np.array(training_image.get_fdata()[:, :, k], dtype='int16') # I checked: all values in the nifti files were integers, ranging from -1024 to approx 3000
            label_2d = np.array(training_label.get_fdata()[:, :, k], dtype='uint8') # only contains 1s and 0s
            slice_number = str(k).zfill(3)
            if len(np.unique(label_2d))!=1:
              contains_cancer.append([image_number,slice_number,1])
            else:
              contains_cancer.append([image_number,slice_number,0])

            np.save((image_saved_path+'image_{}_{}.npy'.format(image_number,slice_number)), image_2d)
            np.save((label_saved_path +'label_{}_{}.npy'.format(image_number,slice_number)), label_2d)
            print(f'Saved image {image_number}, slice {slice_number}')
    
    with open(args.path+"contains_cancer_test_index.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(contains_cancer)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Slicing dataset for image segmentation of Colon"
    )
    parser.add_argument(
        "--path", type=dir_path, default='./data/', help="path for the parent folder of the sliced images and labels"
    )
    parser.add_argument(
        "--datapath", type=dir_path, default='./Task10_Colon', help="path to the dataset"
    )
    parser.add_argument(
        "--split", type=float, default=0.2, help="ratio for testset"
    )
    makedirs("./data/")
    args = parser.parse_args()
    convert_to_npy_train(args)
    convert_to_npy_test(args)