# Colon Image Segmentation

## under construction :D

## 1. Data

The Dataset used for image segmentation was made publicly available on [Medical Segmentation Decathlon](http://medicaldecathlon.com/index.html). 

1) Download and extract the dataset. It is recommended to extract it in the same folder of this repository.

2) Run the code below on the command line to create 2D slices of the CT images from the original data are in 3D NifTI format.
    ```
    python dataset.py --method create_dataset 
    ```

3) Run the code below on the command line to split the data into training data and test data.
    ```
    python dataset.py --method assign_subsets
    ```

You can set a path for data, split method, split ratio, and other arguments. For more options and help `run: python dataset.py --h`

## 2. Train
1) Run the code below on the command line. 
    ```
    python train.py
    ```
2) After finishing the training, run the command below to see how the loss and learning rate changes with every epoch.
    ```
    tensorboard --logdir=runs
    ```

For more options and help run: `python train.py --h`

## 3. Test
1) Run the code below on the command line. 
    ```
    python test.py
    ```
For more options and help run: `python test.py --h`

