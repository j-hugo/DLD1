# evaluation 
from data_loading import ColonDataset
from architecture import UNet, ResNetUNet
from loss import calc_loss, print_metrics, dice_coef

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
from torchvision import models


def load_datasets(args):
    dataset = ColonDataset(
        image_dir=args.testimages,
        label_dir=args.testlabels,
        json_dir=args.jsonfile,
        image_size=args.image_size,
        torch_transform=args.transform,
        test=True
    )
    return dataset

def load_dataloader(dataset):
    dataloader = {
       'test': DataLoader(dataset, shuffle=False, batch_size=1)
    }
    return dataloader

def plot_result(img, label, pred, index, path, dice_score):
    img = img.cpu()
    result = pred.cpu()
    label = label.cpu()
    plt.figure('check', (18, 6))
    ax1 = plt.subplot(1, 3, 1)
    ax1.set_title('image')
    ax1.imshow(img[0][0][ :, :])
    ax2 = plt.subplot(1, 3, 2)
    ax2.set_title('label')
    ax2.imshow(label[0][0][ :, :])
    ax3 = plt.subplot(1, 3, 3)
    ax3.set_title(f'prediction: dice_score ({dice_score.item():.4f})')
    ax3.imshow(result[0][0][:, :])
    plt.savefig(f'{path}eval_plot_{index}.png')

def test_model(model, device, dataloaders, plot_path, info):
    model.load_state_dict(torch.load(f"{args.model_path}best_metric_{args.model}_{args.metric_dataset_type}_{args.epochs}.pth")) 
    test_dice = list()
    test_tumor_dice = list()
    test_non_tumor_dice = list()
    print('-' * 10)
    print('The Evaluation Starts ...')
    print('-' * 10)
    since = time.time()
    # Test Phase
    model.eval()   # Set model to evaluate mode
    test_metrics = {}
    test_samples = 0
    test_tumor_samples = 0
    test_non_tumor_samples = 0
    gt_tum_pd_tum_ok = 0
    gt_tum_pd_tum_no = 0
    gt_tum_pd_no = 0
    gt_no_pd_tumor = 0
    gt_no_pd_no = 0
    i = 0
    for inputs, labels in dataloaders['test']:
        inputs = inputs.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            preds = torch.sigmoid(outputs)
            preds = torch.round(preds)
            dice_score = dice_coef(preds, labels)
            predicted_num_class = len(torch.unique(preds))
            number_label_class = len(torch.unique(labels)) 
        #if len(torch.unique(labels)) != 1:
        #    plot_result(inputs, labels, preds, i, plot_path, dice_score)
        #if i % 50 == 0:
        #    plot_result(inputs, labels, preds, i, plot_path, dice_score)
        if number_label_class != 1:
            test_tumor_dice.append(dice_score)
            test_tumor_samples += 1
            if predicted_num_class != 1 and dice_score >= 0.009:
                gt_tum_pd_tum_ok += 1
                #plot_result(inputs, labels, preds, i, plot_path, dice_score)
            elif predicted_num_class == 1:
                gt_tum_pd_no += 1
                #plot_result(inputs, labels, preds, i, plot_path, dice_score)
            else:
                gt_tum_pd_tum_no += 1
                #plot_result(inputs, labels, preds, i, plot_path, dice_score)
        else:
            test_non_tumor_dice.append(dice_score)
            test_non_tumor_samples += 1
            if predicted_num_class == 1:
                gt_no_pd_no += 1
            if predicted_num_class == 2:
                gt_no_pd_tumor += 1
        #print(f"The {i} image's dice score is {dice_score}.")
        test_dice.append(dice_score)
        test_samples += 1
        i += 1

    test_metrics = []
    average_dice_score = sum(test_dice) / test_samples
    average_tumor_dice_score = sum(test_tumor_dice) / test_tumor_samples
    average_non_tumor_dice_score = sum(test_non_tumor_dice) / test_non_tumor_samples
    print(f"The total samples: {test_samples}")
    print(f"The average dice score is {average_dice_score}.")
    print(f"The number of tumor samples: {test_tumor_samples}")
    print(f"The average dice score of the slices which have tumor is {average_tumor_dice_score}.")
    print(f"The number of correct cases when the prediction predicts some poriton of the tumor: {gt_tum_pd_tum_ok}")
    print(f"The number of incorrect cases when the prediction predicts some poriton of the tumor: {gt_tum_pd_tum_no}")
    print(f"The number of cases when the prediction predicts no tumor but it has tumor: {gt_tum_pd_no}")
    print(f"The number of non-tumor samples: {test_non_tumor_samples}")
    print(f"The average dice score of the slices which have non-tumor is {average_non_tumor_dice_score}.")
    print(f"The number of cases when the prediction predicts no tumor when it has no tumor: {gt_no_pd_no}")
    print(f"The number of cases when the prediction predicts tumor when it has no tumor: {gt_no_pd_tumor}")
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    if args.savemetrics is True:
        with open("sample_file.json", "r+") as file:
            data = json.load(file)
            data.update(test_metrics)
            file.seek(0)
            json.dump(data, file)

    return test_dice

def makedirs(args):
    os.makedirs(args.plot_path, exist_ok=True)

def main(args):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    dataset = load_datasets(args)
    colon_dataloader = load_dataloader(dataset)
    info = {'average_dice_score':0, 'average_tumor_dice_score':0, 'average_non_tumor_dice_score':0}
    if args.model == 'unet':
        model = UNet(n_channel=1,n_class=1).to(device)
    elif args.model == 'resnetunet':
        base_net = models.resnet34(pretrained=True)
        base_net.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        model = ResNetUNet(base_net,args.num_class).to(device)
    print('----------------------------------------------------------------')
    print(f"The number of test set: {len(colon_dataloader['test'])}")
    print('----------------------------------------------------------------')
    result = test_model(model, device, colon_dataloader, args.plot_path, info)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Testing the model for image segmentation of Colon"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="number of epochs to train (default: 100)",
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
        "--model-path", type=str, default="./weights/", help="folder to save weights"
    )
    parser.add_argument(
        "--plot-path", type=str, default="./eval_plot/", help="folder to save eval plots"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--testimages", type=str, default="./data/npy_images", help="root folder with images"
    )
    parser.add_argument(
        "--testlabels", type=str, default="./data/npy_labels", help="root folder with labels"
    )
    parser.add_argument(
        "--jsonfile", type=str, default="./data/data_index_subsets.json", help="json file with assigned subsets"
    )
    parser.add_argument(
        "--transform", type=bool, default=True, help="activate data augmentation"
    )
    parser.add_argument(
        "--metric-dataset-type", type=str, default=None, help="choose what type of dataset you need for loading best metric; \
        None=original dataset, \
        undersample=adjust to the number of non tumor images to the number of tumor images, \
        oversample=adjust to the number of tumor images to the number of non-tumor data, \
        only_tumor=take only images which have tumor"
    )
    parser.add_argument(
        "--model", type=str, default='unet', help="choose the model between unet and resnet+unet; UNet-> unet, Resnet+Unet-> resnetunet"
    )
    parser.add_argument(
        "--savemetrics", type=bool, default=False,
        help="save test metrics to json"
    )
    parser.add_argument(
        "--savemetricspath", type=str, default="./metrics/",
        help="metrics json dir"
    )
    args = parser.parse_args()
    main(args)
