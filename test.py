# evaluation 
from data_loading import ColonDataset

import argparse
import _osx_support
import copy 
import time 
import os
from collections import defaultdict

import torch 
from torch.utils.data import DataLoader
from torchsummary import summary
from architecture import UNet, ResNetUNet
from loss import calc_loss, print_metrics, dice_coeff
from torchvision import models
from torch.optim import lr_scheduler, Adam

def load_datasets(args):
    dataset = ColonDataset(
        image_dir=args.testimages,
        label_dir=args.testlabels,
        csv_dir=args.testcsv,
        image_size=args.image_size,
        torch_transform=args.transform,
        balance_dataset=args.dataset_type,
        test=True
    )
    return dataset

def load_dataloader(args, dataset):
    dataloader = {
       'test': DataLoader(dataset, shuffle=args.shuffle, batch_size=1)
    }
    return dataloader

def test_model(model, device, dataloaders):
    model.load_state_dict(torch.load(f"{args.weights}best_metric_model_{args.model}.pth")) 
    test_dice = list()
    print('-' * 10)
    since = time.time()
    # Test Phase
    model.eval()   # Set model to evaluate mode
    metrics = defaultdict(float)
    test_samples = 0
    i = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            dice_score = dice_coeff(torch.sigmoid(outputs), labels)
        # statistics
        print(f"The {i} image's dice score is {dice_score}.")
        test_dice.append(dice_score)
        test_samples += inputs.size(0)
        i += 1
    
    average_dice_score = sum(test_dice) / test_samples

    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return test_dice

def main(args):
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    dataset = load_datasets(args)
    colon_dataloader = load_dataloader(args, dataset)
    if args.model == 'unet':
        model = UNet(args.num_channel, args.num_class).to(device)
    elif args.model == 'resnetunet':
        base_net = models.resnet34(pretrained=True)
        base_net.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        model = ResNetUNet(base_net,args.num_class).to(device)
    result = test_model(model, device, colon_dataloader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Testing the model for image segmentation of Colon"
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
        "--weights", type=str, default="./weights/", help="folder to save weights"
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
        "--trainimages", type=str, default="./npy_train_images", help="root folder with images"
    )
    parser.add_argument(
        "--trainlabels", type=str, default="./npy_train_labels", help="root folder with labels"
    )
    parser.add_argument(
        "--testimages", type=str, default="./npy_test_images", help="root folder with images"
    )
    parser.add_argument(
        "--testlabels", type=str, default="./npy_test_labels", help="root folder with labels"
    )
    parser.add_argument(
        "--traincsv", type=str, default="./contains_cancer_train_index.csv", help="root folder with csv"
    )
    parser.add_argument(
        "--testcsv", type=str, default="./contains_cancer_test_index.csv", help="root folder with csv"
    )
    parser.add_argument(
        "--transform", type=bool, default=True, help="activate data augmentation"
    )
    parser.add_argument(
        "--dataset-type", type=str, default=None, help="choose what type of dataset you need; \
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
    parser.add_argument(
        "--num_class", type=int, default=2, help="the number of class for image segmentation"
    )
    parser.add_argument(
        "--num_channel", type=int, default=1, help="the number of channel of the image"
    )
    parser.add_argument(
        "--freeze", type=bool, default=False, help="freeze the pretrained weights of resnet"
    )
    parser.add_argument(
        "--step-size", type=int, default=50, help="step size of StepLR scheduler"
    )
    parser.add_argument(
        "--model", type=str, default='unet', help="choose the model between unet and resnet+unet; UNet-> unet, Resnet+Unet-> resnetunet"
    )
    args = parser.parse_args()
    main(args)