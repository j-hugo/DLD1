import os
import numpy as np
import nibabel
import sys
from glob import glob
import argparse
import json
import random

def makedirs(path):
    os.makedirs(path, exist_ok=True)

def dir_path(string):
    if os.path.isdir(string):
        return string
    else:
        raise NotADirectoryError(string)

def convert_to_npy(args):
    """
        Convert nitfi image files to numpy files

        Args:
            args: arguments from the parser. 

    """
    # load a path of image files and sort them
    train_data_path = os.path.join(args.datapath, 'imagesTr')
    label_data_path = os.path.join(args.datapath, 'labelsTr')
    images = sorted(os.listdir(train_data_path))
    labels = sorted(os.listdir(label_data_path))

    # set the location to converted image files
    image_saved_path =args.path +'npy_images/'
    
    # create a directory of the converted image files
    try:
        os.mkdir(image_saved_path)
    except OSError:
        print ("Creation of the directory %s failed" % image_saved_path)
    else:
        print ("Successfully created the directory %s " % image_saved_path)

    # set the location to converted label files
    label_saved_path = args.path + 'npy_labels/'
    # create a directory of the converted label files
    try:
        os.mkdir(label_saved_path)
    except OSError:
        print ("Creation of the directory %s failed" % label_saved_path)
    else:
        print ("Successfully created the directory %s " % label_saved_path)
        
    data_index = {}

    for img, label in zip(images,labels):
        # Load 3D CT images
        image_number = str(''.join(filter(str.isdigit, img))).zfill(3)
        training_image = nibabel.load(os.path.join(train_data_path, img))
        training_label = nibabel.load(os.path.join(label_data_path, label))

        for k in range(training_label.shape[2]):
            # axial cuts are made along the z axis (slice) 
            image_2d = np.array(training_image.get_fdata()[:, :, k], dtype='int16') # I checked: all values in the nifti files were integers, ranging from -1024 to approx 3000
            label_2d = np.array(training_label.get_fdata()[:, :, k], dtype='uint8') # only contains 1s and 0s
            slice_number = str(k).zfill(3)
            slice_index = image_number+'_'+slice_number
            
            if len(np.unique(label_2d))!=1:
              contains_cancer = True
            else:
              contains_cancer = False

            data_index[slice_index] = {
                'image': int(image_number),
                'slice': int(slice_number),
                'cancer': contains_cancer,
                'subset': None
            }

            np.save((image_saved_path+'image_{}_{}.npy'.format(image_number,slice_number)), image_2d)
            np.save((label_saved_path +'label_{}_{}.npy'.format(image_number,slice_number)), label_2d)
            
        print(f'Saved slices of image {image_number}')
    
    with open(args.path+"data_index.json", "w") as json_file:
        json.dump(data_index,json_file)

def create_data_subsets(args):
    """
        Split the data for test and train. 
        You can choose how to split the data by examples(patients) or by slices.
        It saved as json file and the json file is used one of arguments of ConlonDataset on dataset.py

        Args:
            args: arguments from the parser. 

    """
    data_index_file = args.path+"data_index.json"
    
    with open(data_index_file) as json_file:
        data_index = json.load(json_file)

        if args.split_on == "examples":
          image_index = [v['image'] for _, v in data_index.items()]
          unique_images = set(image_index)
          test_length = int(len(unique_images)*args.split)
          test_images = random.sample(unique_images,test_length)
          test_slices = [k for k,v in data_index.items() if v['image'] in test_images]
          for k,_ in data_index.items():
            if k in test_slices:
              data_index[k]['subset'] = 'test'
            else:
              data_index[k]['subset'] = 'train'

        if args.split_on == "slices":

            cancer_slice_index = [k for k,v  in data_index.items() if v['cancer'] is True]
            non_cancer_slice_index = [k for k, v in data_index.items() if v['cancer'] is False]
            cancer_slice_length = len(cancer_slice_index)
            non_cancer_slice_length = len(non_cancer_slice_index)
            slice_length = cancer_slice_length + non_cancer_slice_length

            proportion_cancer = cancer_slice_length/(non_cancer_slice_length+cancer_slice_length)
            test_length = int(slice_length*args.split)
            test_cancer_slices_length = int(test_length*proportion_cancer)
            test_non_cancer_slices_length = test_length-test_cancer_slices_length

            test_slices = [*random.sample(cancer_slice_index, test_cancer_slices_length),
                           *random.sample(non_cancer_slice_index,test_non_cancer_slices_length)]

            for k,_ in data_index.items():
                if k in test_slices:
                    data_index[k]['subset'] = 'test'
                else:
                    data_index[k]['subset'] = 'train'

    with open(args.path+"data_index_subsets.json", "w") as json_file:
        json.dump(data_index, json_file)


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
        "--split", type=float, default=0.1, help="ratio for testset"
    )
    parser.add_argument(
        "--split_on", type=str, default="examples", help="apply split ratio on number of slices (slices) or number of examples/patients (examples)"
    )
    parser.add_argument(
        "--method", type=str, default=None,
        help="create npy dataset (create_dataset) or assign training examples to subsets (assign_subsets)"
    )
    makedirs("./data/")
    args = parser.parse_args()

    if args.method == "create_dataset":
        convert_to_npy(args)
    elif args.method == "assign_subsets":
        create_data_subsets(args)
    else:
        print("Please select whether you want to convert nifti files to npy files (create_dataset) or assign slices to test or train data-subset (assign_subsets) via --method argument")

