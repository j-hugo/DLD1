# instantiates a model, conducts the training and saves the model
# adapted from https://github.com/mateuszbuda/brain-segmentation-pytorch
import argparse
import os
import torch 
from torch.utils.data import DataLoader
from architecture import UNet, ResNetUNet
from data_loading import ColonDataset

def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)

def load_datasets(args):
    dataset = ColonDataset(
        image_dir=args.images,
        label_dir=args.labels,
        csv_dir=args.csv,
        image_size=args.image_size,
        torch_transform=args.transform,
        balance_dataset=args.dataset_type
    )
    train_size = int(args.split_ratio*len(dataset))
    val_size = len(dataset)-train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset

def load_dataloader(args, train, valid):
    dataloader = {
       'train': DataLoader(train, shuffle=args.shuffle, batch_size=args.train_batch),
        'val': DataLoader(valid, shuffle=args.shuffle, batch_size=args.valid_batch)
    }
    return dataloader



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training the model for image segmentation of Colon"
    )
    parser.add_argument(
        "--train-batch",
        type=int,
        default=12,
        help="input batch size for train (default: 12)",
    )
    parser.add_argument(
        "--valid-batch",
        type=int,
        default=4,
        help="input batch size for valid (default: 12)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: cuda:0)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--weights", type=str, default="./weights", help="folder to save weights"
    )
    parser.add_argument(
        "--logs", type=str, default="./logs", help="folder to save logs"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--images", type=str, default="./npy_images", help="root folder with images"
    )
    parser.add_argument(
        "--labels", type=str, default="./npy_labels", help="root folder with labels"
    )
    parser.add_argument(
        "--csv", type=str, default="./contains_cancer_index.csv", help="root folder with csv"
    )
    parser.add_argument(
        "--transform", type=bool, default=True, help="activate data augmentation"
    )
    parser.add_argument(
        "--dataset-type", type=str, default='undersample', help="choose what type of dataset you need; \
        None=original dataset, \
        undersample=adjust to the number of non tumor images to the number of tumor images, \
        oversample=adjust to the number of tumor images to the number of non-tumor data, \
        only_tumor=take only images which have tumor"
    )
    parser.add_argument(
        "--split-ratio", type=float, default=0.8, help="the ratio to split the dataset into training and valid"
    )
    parser.add_argument(
        "--shuffle", type=bool, default=False, help="shuffle the datset or not"
    )
    args = parser.parse_args()
    train, valid = load_datasets(args)
    colon_dataloader = load_dataloader(args, train, valid)
    