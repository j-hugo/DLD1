# evaluation 
from data_loading import ColonDataset
from architecture import UNet, ResNetUNet
from loss import calc_loss, print_metrics, dice_coef
from utils import overlay_plot

import random
import argparse
import _osx_support
import copy 
import time 
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import json

import torch 
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import models

def makedirs(args):
    """create directories to save plot

    Args:
        args: arguments from the parser.
        
    """
    os.makedirs(args.plot_path, exist_ok=True)

def load_datasets(args):
    """load the dataset for evaluation   

    Args:
        args: arguments from the parser.
        
    """
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
    """load a dataloader using a dataset (ColonDataset)

    Args:
        dataset: a dataset to create a data loader

    Return:
        dataloader: a dataloader for evaluation 
    """
    dataloader = {
       'test': DataLoader(dataset, shuffle=False, batch_size=1)
    }
    return dataloader

def test_model(model, device, dataloaders, pred_save, info, args):
    """
        Evaluate the model

        Args:
            model: A neural netowrk model for evaluation
            device: gpu or cpu
            dataloaders: a data loader
            pred_save: save the image, label and prediction
            info: a dictionary to save metrics of evaluation
        
        Return:
            predicted: If pred_save is true, a dictionary which has image, label, and prediction of the slice returns
        """
    # load the trained model
    check = torch.load(f"{args.model_path}best_metric_{args.model}_{args.test_dataset_type}_{args.epochs}.pth")
    model.load_state_dict(check['model_state_dict']) 
    
    # initialize to save dice scores
    test_dice = list()
    test_cancer_dice = list()
    test_non_cancer_dice = list()
    if pred_save == True:
        predicted = dict()
    print('-' * 10)
    print('The Evaluation Starts ...')
    print('-' * 10)
    since = time.time()
    # Test Phase
    model.eval()   # Set model to evaluate mode
    
    # initilize dictionary to save metrics for evaluation
    test_metrics = {}
    
    # initilize variables to save metrics for evaluation
    test_samples = 0
    test_cancer_samples = 0
    test_non_cancer_samples = 0
    gt_c_pd_c_overlap = 0
    gt_c_pd_c_no_overlap = 0
    gt_c_pd_no_c = 0
    gt_n_pd_c = 0
    gt_n_pd_n = 0

    # to count images
    i = 0

    # load image and label
    for images, labels in dataloaders['test']:
        cancer = 'non-cancer'
        pd = 'non-cancer'
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(False):
            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = torch.round(preds)
            dice_score = dice_coef(preds, labels)
            if pred_save == True:
                data = {'img':images[0].cpu().numpy(), 'label':labels[0].cpu().numpy(), 'pred':preds[0].cpu().numpy(), \
                        'dice': dice_score.item(), 'cancer_gt': False, 'cancer_pd': False}
                predicted[i] = data
            predicted_num_class = len(torch.unique(preds))
            number_label_class = len(torch.unique(labels)) 
        
        # check the image has cancer
        if number_label_class != 1:
            if pred_save == True:
                predicted[i]['cancer_gt'] = True
            cancer = 'cancer'
            test_cancer_dice.append(dice_score)
            test_cancer_samples += 1
            # check the prediction has cancer and save some metrics
            if predicted_num_class != 1 and dice_score >= 0.009:
                gt_c_pd_c_overlap += 1
                pd = 'cancer'
                if pred_save == True:
                    predicted[i]['cancer_pd'] = True
            elif predicted_num_class == 1:
                gt_c_pd_no_c += 1
            else:
                gt_c_pd_c_no_overlap += 1
                pd = 'cancer'
                if pred_save == True:
                    predicted[i]['cancer_pd'] = True
        else:
            test_non_cancer_dice.append(dice_score)
            test_non_cancer_samples += 1
            if predicted_num_class == 1:
                gt_n_pd_n += 1
            if predicted_num_class == 2:
                gt_n_pd_c += 1
        #print(f"The {i} image's dice score is {dice_score}.")
        test_dice.append(dice_score)
        # save dice score, ground truth, and prediction for each slice 
        info['dice_score_each_slice'].append({i: dice_score.item(), "gt": cancer, 'pd': pd})
        test_samples += 1
        i += 1

    # calculate average dice score for the test set
    average_dice_score = sum(test_dice) / test_samples
    average_cancer_dice_score = sum(test_cancer_dice) / test_cancer_samples    
    average_non_cancer_dice_score = sum(test_non_cancer_dice) / test_non_cancer_samples

    # save metrics to the info
    info['number of cancer case'] = test_cancer_samples
    info['number of non-cancer case'] = test_non_cancer_samples

    info['average_dice_score'] = average_dice_score.item()
    info['average_cancer_dice_score'] = average_cancer_dice_score.item()
    info['average_non_cancer_dice_score'] = average_non_cancer_dice_score.item()

    info['gt_c_pd_c_overlap'] = gt_c_pd_c_overlap
    info['gt_c_pd_c_no_overlap'] = gt_c_pd_c_no_overlap
    info['gt_c_pd_no_c'] = gt_c_pd_no_c

    info['gt_n_pd_n'] = gt_n_pd_n
    info['gt_n_pd_c'] = gt_n_pd_c

    # print all the results of evaluation
    print(f"The total samples: {test_samples}")
    print(f"The average dice score is {average_dice_score}.")
    print(f"The number of cancer samples: {test_cancer_samples}")
    print(f"The average dice score of the slices which have cancer is {average_cancer_dice_score}.")
    print(f"The number of correct cases when the prediction predicts some poriton of the cancer: {gt_c_pd_c_overlap}")
    print(f"The number of incorrect cases when the prediction predicts some poriton of the cancer: {gt_c_pd_c_no_overlap}")
    print(f"The number of cases when the prediction predicts no cancer but it has cancer: {gt_c_pd_no_c}")
    print(f"The number of non-cancer samples: {test_non_cancer_samples}")
    print(f"The average dice score of the slices which have non-cancer is {average_non_cancer_dice_score}.")
    print(f"The number of cases when the prediction predicts no cancer when it has no cancer: {gt_n_pd_n}")
    print(f"The number of cases when the prediction predicts cancer when it has no cancer: {gt_n_pd_c}")
    
    time_elapsed = time.time() - since
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    
    if pred_save == True:
        return predicted

def main(args):
    if args.save_plot:
        makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0")
    dataset = load_datasets(args)
    colon_dataloader = load_dataloader(dataset)
    # initialize a dictionary to save metrics for evaluation
    info_test = {'test set size':0, 'average_dice_score':0, \
                 'number of cancer case': 0, 'average_cancer_dice_score':0, \
                 'number of non-cancer case': 0, 'average_non_cancer_dice_score':0, \
                 'gt_c_pd_c_overlap':0, 'gt_c_pd_c_no_overlap':0, 'gt_c_pd_no_c':0, \
                 'gt_n_pd_n': 0, 'gt_n_pd_c':0, 'dice_score_each_slice':[]}
    info_test['test set size'] = len(colon_dataloader['test'])
    
    # initialize the model for evaluation
    if args.model == 'unet':
        model = UNet(n_channel=1,n_class=1).to(device)
    elif args.model == 'resnetunet':
        base_net = models.resnet34(pretrained=True)
        base_net.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        model = ResNetUNet(base_net, n_class=1).to(device)
    print('----------------------------------------------------------------')
    print(f"The number of test set: {len(colon_dataloader['test'])}")
    print('----------------------------------------------------------------')
    
    # starts evaluation of the model
    result = test_model(model, device, colon_dataloader, args.save_plot, info_test, args)
    
    # save the result from the evalutaion
    with open(f"{args.metric_path}best_metric_{args.model}_{args.test_dataset_type}_{args.epochs}.json", 'ab+') as f:
        f.seek(0,2)                                #Go to the end of file    
        if f.tell() == 0 :                         #Check if file is empty
            f.write(json.dumps(info_test, indent=4).encode())  #If empty, write an array
        else:  
            f.seek(-1,2)           
            f.truncate() 
            f.write(', "test": '.encode()) 
            f.write(json.dumps(info_test, indent=4).encode())    #Dump the dictionary
            f.write('}'.encode())  

    # If save_plot is true, it saves randomly chosen overlay image; 
    # six images from cancer case, six images from non-cancer case 
    # red: ground truth label, green: predicted label
    if args.save_plot:
        c_index = list()
        n_index = list()
        for i, j in result.items():
            if j['cancer_gt'] and j['cancer_pd']:
                c_index.append(i)
            else:
                n_index.append(i)
        cancer = random.sample(c_index, 6)
        no_cancer = random.sample(n_index, 6)
        rand_index = np.concatenate((cancer, no_cancer))
        for pos, i in enumerate(rand_index):
            image = overlay_plot(result[i]['img'], result[i]['label'], result[i]['pred'], i, args, args.save_plot)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Testing the model for image segmentation of Colon cancer"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="number of epochs to train (default: 100)",
    )
    parser.add_argument(
        "--model-path", type=str, default="./save/models/", help="the path to load model"
    )
    parser.add_argument(
        "--metric-path", type=str, default="./save/metrics/",
        help="the path to save metrics result"
    )
    parser.add_argument(
        "--plot-path", type=str, default="./save/plots/",
        help="the path to save plots"
    )
    parser.add_argument(
        "--save-plot", type=bool, default=False,
        help="choose to save plots"
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
        "--test-dataset-type", type=str, default=None, help="choose what type of dataset you need for loading best metric; \
        None=original dataset, \
        undersample=adjust to the number of non cancer images to the number of cancer images, \
        oversample=adjust to the number of cancer images to the number of non-cancer data, \
        only_tumor=take only images which have cancer"
    )
    parser.add_argument(
        "--model", type=str, default='unet', help="choose the model between unet and resnet+unet; UNet-> unet, Resnet+Unet-> resnetunet"
    )
    args = parser.parse_args()
    main(args)
