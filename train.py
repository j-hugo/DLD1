from data_loading import ColonDataset

import argparse
import _osx_support
import copy 
import time 
import os
import numpy as np
from collections import defaultdict
import torch 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from architecture import UNet, ResNetUNet
from loss import calc_loss, print_metrics
from torchvision import models
from torch.optim import lr_scheduler, SGD
import json

def makedirs(args):
    """create directories to save model and metric  

    Args:
        args: arguments from the parser.
        
    """
    os.makedirs(args.model_path, exist_ok=True)
    os.makedirs(args.metric_path, exist_ok=True)

def load_datasets(args):
    """load the dataset (ColonDataset) and split the dataset into train and validation set   

    Args:
        args: arguments from the parser.
        
    Return:
        train_dataset: a dataset for training
        val_dataset: a dataset for validation
    """
    dataset = ColonDataset(
        image_dir=args.trainimages,
        label_dir=args.trainlabels,
        json_dir=args.jsonfile,
        image_size=args.image_size,
        torch_transform=args.transform,
        balance_dataset=args.dataset_type
    )
    # determine train and validation set size and split randomly
    train_size = int(args.split_ratio*len(dataset))
    val_size = len(dataset)-train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    return train_dataset, val_dataset

def load_dataloader(args, train, valid):
    """load a dataloader using train dataset and validation dataset (ColonDataset)

    Args:
        args: the object which store arguments from the parser
        train: train dataset (ColonDataset)
        valid: validation dataset (ColonDataset)
        
    Return:
        dataloader: a dataloader for training 
    """
    dataloader = {
       'train': DataLoader(train, shuffle=True, batch_size=args.train_batch, num_workers=4),
        'val': DataLoader(valid, shuffle=True, batch_size=args.valid_batch, num_workers=4)
    }
    return dataloader

# Adapted from https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, optimizer):
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer)
        elif val_loss > self.best_val_loss:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_val_loss = val_loss
            self.save_checkpoint(val_loss, model, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving best model ...')
        torch.save({'model_state_dict':model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()}, 
                    f"{args.model_path}best_metric_{args.model}_{args.dataset_type}_{args.epochs}.pth")
        self.val_loss_min = val_loss

# adapted from https://github.com/mateuszbuda/brain-segmentation-pytorch
def train_model(model, optimizer, scheduler, device, num_epochs, dataloaders, info, fine_tune=False):
    """
        Train the model

        Args:
            model: A neural netowrk model for training
            optimizer: A optimizer to calculate gradients
            scheduler: A scheduler to change a learning rate
            device: gpu or cpu
            num_epochs: a number of epochs
            dataloaders: a data loader
            info: a dictionary to save metrics
            fine_tune: If True, it saved metrics of the fine tuning phase

        Return:
            model: A trained model
            metric_train: Metrics from training phase
            metric_valid: Metrics from validation phase
    """
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10

    # Initialize list to save loss from train and validation phase
    epoch_train_loss = list()
    epoch_valid_loss = list()
    epoch_train_dice_loss = list()
    epoch_valid_dice_loss = list()
    epoch_train_bce = list()
    epoch_valid_bce = list()
    
    # Initialize SummaryWriter to visualize losses on Tensorboard
    writer = SummaryWriter()

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=args.earlystop, verbose=True)
    
    # Training starts
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        since = time.time()
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            # initialize metric dict to save losses for each epoch
            metrics = defaultdict(float)
            epoch_samples = 0

            # Load a batch of images and labels 
            for images, labels in dataloaders[phase]:
                images = images.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(images)
                    loss = calc_loss(outputs, labels, metrics)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                epoch_samples += images.size(0)

            # print metrics 
            print_metrics(metrics, epoch_samples, phase)

            # save the loss of the current epoch
            epoch_loss = metrics['loss'] / epoch_samples

            if phase == 'train':
                # save training metrics for tensorboard
                writer.add_scalar('Loss(BCE+Dice)/train', epoch_loss, epoch)
                writer.add_scalar('Dice Loss/train', metrics['dice_loss']/ epoch_samples, epoch)
                writer.add_scalar('BCE/train', metrics['bce']/ epoch_samples, epoch)

                # save training metrics for later use ;)
                epoch_train_loss.append(metrics['loss']/ epoch_samples)
                epoch_train_bce.append(metrics['bce']/ epoch_samples)
                epoch_train_dice_loss.append(metrics['dice_loss']/ epoch_samples)

            elif phase == 'val':
                # save validation metrics for tensorboard
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch) # to plot LR reduction
                writer.add_scalar('Loss(BCE+Dice)/valid', metrics['loss'] / epoch_samples, epoch)
                writer.add_scalar('Dice Loss/valid', metrics['dice_loss'] / epoch_samples, epoch)
                writer.add_scalar('BCE/valid', metrics['bce'] / epoch_samples, epoch)

                # save validation metrics for later use
                epoch_valid_loss.append(metrics['loss']/ epoch_samples)
                epoch_valid_bce.append(metrics['bce']/ epoch_samples)
                epoch_valid_dice_loss.append(metrics['dice_loss']/ epoch_samples)

                scheduler.step(epoch_loss) # pass loss to ReduceLROnPlateau scheduler

                early_stopping(epoch_loss, model, optimizer) #  evaluate early stopping criterion

                # compare loss and deep copy the model
                if epoch_loss < best_loss:
                    best_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())

        time_elapsed = time.time() - since # compute time of epoch
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        # check early stop is True or not
        if early_stopping.early_stop:
            print(f"Early stopping after epoch {epoch}")
            if fine_tune == False:
                info['early stop'] = 'True'
                info['stopping LR'] = optimizer.param_groups[0]['lr']
                info['stopping epoch'] = epoch+1
                info['best loss'] = best_loss
            else:
                info['fine_tune_early stop'] = 'True'
                info['fine_tune_stopping LR'] = optimizer.param_groups[0]['lr']
                info['fine_tune_stopping epoch'] = epoch+1
                info['fine_tune_best loss'] = best_loss
            break   

    # check early stop is True or not
    if early_stopping.early_stop != True:
        if fine_tune == False:
            info['early stop'] = 'False'
            info['stopping LR'] = optimizer.param_groups[0]['lr']
            info['stopping epoch'] = num_epochs
            info['best loss'] = best_loss
        else:
            info['fine_tune_early stop'] = 'False'
            info['fine_tune_stopping LR'] = optimizer.param_groups[0]['lr']
            info['fine_tune_stopping epoch'] = num_epochs
            info['fine_tune_best loss'] = best_loss
    
    print('Best val loss: {:4f}'.format(best_loss))

    # collect all metrics
    metric_train = (epoch_train_loss, epoch_train_bce, epoch_train_dice_loss)
    metric_valid = (epoch_valid_loss, epoch_valid_bce, epoch_valid_dice_loss)

    # load best model weights (necessary for fine tuning of ResNet-UNet)
    model.load_state_dict(best_model_wts)

    writer.close() # end tensorboard writing

    return model, metric_train, metric_valid

def main(args):
    makedirs(args) # create necessary directories
    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda:0") # set device to GPU if available
    
    train, valid = load_datasets(args) # get train and val dataset
    colon_dataloader = load_dataloader(args, train, valid) # load dataloader 
    
    # initialize dictionary to save informations about the model, dataset, and metrics at the end.
    info = {'train': {}}
    info_train = info['train']
    info_train['model'] = args.model
    info_train['dataset'] = args.dataset_type
    info_train['image_size'] = args.image_size
    info_train['train set size'] = len(train)
    info_train['val set size'] = len(valid)
    
    # initialze a model for training
    if args.model == 'unet':
        model = UNet(n_channel=1, n_class=1).to(device)
    elif args.model == 'resnetunet':
        base_net = models.resnet34(pretrained=True)
        base_net.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False) # adjust first layer of ResNet to allow input of 1 channel
        model = ResNetUNet(base_net,n_class=1).to(device)

    # describe the model
    if device == 'cpu':
        print(model)
    else:
        summary(model, input_size=(1, args.image_size, args.image_size))
    
    print('----------------------------------------------------------------')
    print(f"The number of train set: {len(train)}")
    print(f"The number of valid set: {len(valid)}")
    print('----------------------------------------------------------------')

    # specify optimizer function
    optimizer = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.9)

    # initialise learning rate scheduler
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', threshold_mode='abs', min_lr=1e-8, factor=0.1, patience=args.sched_patience)

    # continue to train the trained model
    if args.load:
        checkpoint = torch.load(f"{args.model_path}best_metric_{args.model}_{args.dataset_type}_{args.epochs}.pth")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_ft.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # freeze pretrained laeyrs
    if args.model == 'resnetunet':    
        for l in model.base_layers:
            for param in l.parameters():
                param.requires_grad = False

    # train starts
    if args.model == 'resnetunet':
        num_epochs = args.epochs - 50
    else:
        num_epochs = args.epochs

    model, metric_t, metric_v = train_model(model, optimizer, scheduler, device, num_epochs, colon_dataloader, info_train)
    
    # for fine tuning restnetunet model
    if args.model == 'resnetunet':
        print('----------------------------------------------------------------')
        print("Fine Tuning of ResNetUnet starts ...")
        print('----------------------------------------------------------------')
        for l in model.base_layers:
            for param in l.parameters():
                param.requires_grad = True
        # specify optimizer function
        optimizer_ft = SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr*0.01, momentum=0.9)
        # initialise learning rate scheduler
        scheduler_ft = lr_scheduler.ReduceLROnPlateau(optimizer_ft, 'min', threshold_mode='abs', min_lr=1e-8, factor=0.1, patience=args.sched_patience)
        model, metric_ft, metric_fv = train_model(model, optimizer_ft, scheduler_ft, device, int(num_epochs/3), colon_dataloader, info_train, True)
    
    # create json file from save information about the model, dataset, and metrics.
    with open(f"{args.metric_path}best_metric_{args.model}_{args.dataset_type}_{args.epochs}.json", "w") as json_file:
        json.dump(info, json_file, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training the model for image segmentation of Colon cancer"
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
        "--epochs",
        type=int,
        default=200,
        help="number of epochs to train (default: 200)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="initial learning rate (default: 0.001)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: 4)",
    )
    parser.add_argument(
        "--model-path", type=str, default="./save/models/", help="folder to save model"
    )
    parser.add_argument(
        "--metric-path", type=str, default="./save/metrics/",
        help="to save metrics result"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: 256)",
    )
    parser.add_argument(
        "--trainimages", type=str, default="./data/npy_images", help="root folder with images"
    )
    parser.add_argument(
        "--trainlabels", type=str, default="./data/npy_labels", help="root folder with labels"
    )
    parser.add_argument(
        "--jsonfile", type=str, default="./data/data_index_subsets.json", help="root folder with json with assigned subsets"
    )
    parser.add_argument(
        "--transform", type=bool, default=True, help="activate data augmentation"
    )
    parser.add_argument(
        "--dataset-type", type=str, default=None, help="choose what type of dataset you need; \
        None=original dataset, \
        undersample=adjust to the number of non cancer images to the number of tumor images, \
        oversample=adjust to the number of cancer images to the number of non-cancer data, \
        only_tumor=take only images which have cancer"
    )
    parser.add_argument(
        "--split-ratio", type=float, default=0.9, help="the ratio to split the dataset into training and valid"
    )
    parser.add_argument(
        "--load", type=bool, default=False, help="continute training from the best model"
    )
    parser.add_argument(
        "--model", type=str, default='unet', help="choose the model between unet and resnet+unet; UNet-> unet, Resnet+Unet-> resnetunet"
    )
    parser.add_argument(
        "--earlystop", type=int, default=30, help="the number of patience for early stopping"
    )
    parser.add_argument(
        "--sched-patience", type=int, default=10, help="the number of patience for scheduler"
    )
    args = parser.parse_args()
    main(args)
