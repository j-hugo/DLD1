# evaluation 
from data_loading import ColonDataset

import argparse
import _osx_support
import copy 
import time 
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

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

def plot_result(img, label, pred, index, path, dice_score):
    result = torch.argmax(pred, dim=1)
    plt.figure('check', (18, 6))
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title('image')
    ax1.imshow(img[0][0][ :, :])
    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title('label')
    ax2.imshow(label[0][1][ :, :])
    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title(f'output: dice_score{dice_score}')
    ax3.imshow(result[0][:, :])
    plt.savefig(f'{path}eval_plot_{index}.png')

def test_model(model, device, dataloaders, plot_path):
    model.load_state_dict(torch.load(f"{args.weights}best_metric_model_{args.model}_{args.metric_dataset_type}_{args.epochs}.pth")) 
    test_dice = list()
    test_tumor_dice = list()
    test_non_tumor_dice = list()
    print('-' * 10)
    print('The Evaluation Starts ...')
    print('-' * 10)
    since = time.time()
    # Test Phase
    model.eval()   # Set model to evaluate mode
    metrics = defaultdict(float)
    test_samples = 0
    test_tumor_samples = 0
    test_non_tumor_samples = 0
    i = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            dice_score = dice_coeff(preds, labels)
        #if i % 100 == 0:
        #    plot_result(inputs, labels, preds, i, plot_path, dice_score)
        if len(np.unique((torch.argmax(labels, dim=1).cpu()))) != 1:
            test_tumor_dice.append(dice_score)
            test_tumor_samples += 1
        else:
            test_non_tumor_dice.append(dice_score)
            test_non_tumor_samples += 1
        #if len(np.unique(labels)) != 1:
        #    plot_result(inputs, labels, preds, i, plot_path, dice_score)
        # statistics
        #print(f"The {i} image's dice score is {dice_score}.")
        test_dice.append(dice_score)
        test_samples += 1
        i += 1
    average_dice_score = sum(test_dice) / test_samples
    average_tumor_dice_score = sum(test_tumor_dice) / test_tumor_samples
    average_non_tumor_dice_score = sum(test_non_tumor_dice) / test_non_tumor_samples
    print(f"The total samples: {test_samples}")
    print(f"The average dice score is {average_dice_score}.")
    print(f"The number of tumor samples: {test_tumor_samples}")
    print(f"The average tumor dice score is {average_tumor_dice_score}.")
    print(f"The number of non-tumor samples: {test_non_tumor_samples}")
    print(f"The average non tumor dice score is {average_non_tumor_dice_score}.")
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return test_dice

def makedirs(args):
    os.makedirs(args.eval_plot, exist_ok=True)

def main(args):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    dataset = load_datasets(args)
    colon_dataloader = load_dataloader(args, dataset)
    if args.model == 'unet':
        model = UNet(args.num_channel, args.num_class).to(device)
    elif args.model == 'resnetunet':
        base_net = models.resnet34(pretrained=True)
        base_net.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        model = ResNetUNet(base_net,args.num_class).to(device)
    print('----------------------------------------------------------------')
    print(f"The number of test set: {len(colon_dataloader['test'])*args.test_batch}")
    print('----------------------------------------------------------------')
    result = test_model(model, device, colon_dataloader, args.eval_plot)

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
        default=12,
        help="input batch size for valid (default: 12)",
    )
    parser.add_argument(
        "--test-batch",
        type=int,
        default=1,
        help="input batch size for test (default: 1)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
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
        "--eval-plot", type=str, default="./eval_plot/", help="folder to save eval plots"
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
        "--trainimages", type=str, default="./data/npy_train_images", help="root folder with images"
    )
    parser.add_argument(
        "--trainlabels", type=str, default="./data/npy_train_labels", help="root folder with labels"
    )
    parser.add_argument(
        "--testimages", type=str, default="./data/npy_test_images", help="root folder with images"
    )
    parser.add_argument(
        "--testlabels", type=str, default="./data/npy_test_labels", help="root folder with labels"
    )
    parser.add_argument(
        "--traincsv", type=str, default="./data/contains_cancer_train_index.csv", help="root folder with csv"
    )
    parser.add_argument(
        "--testcsv", type=str, default="./data/contains_cancer_test_index.csv", help="root folder with csv"
    )
    parser.add_argument(
        "--transform", type=bool, default=True, help="activate data augmentation"
    )
    parser.add_argument(
        "--dataset-type", type=str, default=None, help="choose what type of dataset you need; \
        None=original dataset, \
        undersample=adjust to the number of non tumor images to the number of tumor images, \
        upsample=adjust to the number of tumor images to the number of non-tumor data, \
        only_tumor=take only images which have tumor"
    )
    parser.add_argument(
        "--metric-dataset-type", type=str, default=None, help="choose what type of dataset you need for loading best metric; \
        None=original dataset, \
        undersample=adjust to the number of non tumor images to the number of tumor images, \
        upsample=adjust to the number of tumor images to the number of non-tumor data, \
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