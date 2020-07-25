# Semantic Segmentation of Colon Cancer Primaries

## 0. Set up the environment
You have to install some packages that need to run our project. 

If you use anaconda, run the following command: 
```
conda env create -f environment.yaml
```
If you want to install packages using pip, run the following command:
```
pip install -r requirements.txt
```

## 1. Data

The Dataset used in this project for semantic image segmentation was made publicly available on [Medical Segmentation Decathlon](http://medicaldecathlon.com/index.html). 

1) Download and extract the dataset. It is recommended to extract it in the same folder as this repository.

2) Run the code below on the command line to create 2D slices of the original 3D CT images (NifTI file format).
    ```
    python dataset.py --method create_dataset 
    ```

3) Run the code below on the command line to split the data into training data and test data.
    ```
    python dataset.py --method assign_subsets
    ```

You can set a path for data, split method, split ratio, and other arguments. For more options and help `run: python dataset.py --h`

## 2. Train
1) Run the code below on the command line to start model training. 
    ```
    python train.py
    ```
2) After finishing the training, run the command below to see how the loss and learning rate changes with every epoch.
    ```
    tensorboard --logdir=runs
    ```

To select the model to be trained, the dataset to be used and more options and help run: `python train.py --h`

## 3. Test
1) Run the code below on the command line to evaluate trained models. 
    ```
    python test.py
    ```
For more options and help run: `python test.py --h`. In testing please specify the model and the dataset it was built on, that you want to test.

