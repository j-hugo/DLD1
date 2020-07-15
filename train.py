# instantiates a model, conducts the training and saves the model
# adapted from https://github.com/mateuszbuda/brain-segmentation-pytorch
from data_loading import ColonDataset

import argparse
import _osx_support
import copy 
import time 
import os
from collections import defaultdict

import torch 
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
from architecture import UNet, ResNetUNet
from loss import calc_loss, print_metrics
from torchvision import models
from torch.optim import lr_scheduler, Adam


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)

def load_datasets(args):
    dataset = ColonDataset(
        image_dir=args.trainimages,
        label_dir=args.trainlabels,
        csv_dir=args.traincsv,
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

def train_model(model, optimizer, scheduler, device, num_epochs, dataloaders):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    epoch_train_loss = list()
    epoch_valid_loss = list()
    epoch_train_dice = list()
    epoch_valid_dice = list()
    epoch_train_bce = list()
    epoch_valid_bce = list()
    writer = SummaryWriter()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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

            metrics = defaultdict(float)
            epoch_samples = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = calc_loss(outputs, labels, metrics)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    
                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
            epoch_loss = metrics['loss'] / epoch_samples
            writer.add_scalar('Loss/train', epoch_loss, epoch)
            writer.add_scalar('Dice/train', metrics['dice']/ epoch_samples, epoch)

            if phase == 'train':
                scheduler.step()
                epoch_train_loss.append(metrics['loss']/ epoch_samples)
                epoch_train_bce.append(metrics['bce']/ epoch_samples)
                epoch_train_dice.append(metrics['dice']/ epoch_samples)
            elif phase == 'val':
                epoch_valid_loss.append(metrics['loss']/ epoch_samples)
                epoch_valid_bce.append(metrics['bce']/ epoch_samples)
                epoch_valid_dice.append(metrics['dice']/ epoch_samples)
                for tag, value in model.named_parameters():
                    tag = tag.replace('.', '/')
                    writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch)
                    writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch)
                writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                writer.add_scalar('Loss/test', metrics['loss']/ epoch_samples, epoch)
                writer.add_scalar('Dice/test', metrics['dice']/ epoch_samples, epoch)
            # deep copy the model
            if phase == 'val' and epoch_loss < best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), f"{args.weights}best_metric_model_{args.model}.pth")        

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    print('Best val loss: {:4f}'.format(best_loss))
    metric_train = (epoch_train_loss, epoch_train_bce, epoch_train_dice)
    metric_valid = (epoch_valid_loss, epoch_valid_bce, epoch_valid_dice)
    # load best model weights
    model.load_state_dict(best_model_wts)
    writer.close()
    return model, metric_train, metric_valid

def main(args):
    makedirs(args)
    device = torch.device("cpu" if not torch.cuda.is_available() else args.device)
    train, valid = load_datasets(args)
    colon_dataloader = load_dataloader(args, train, valid)
    if args.model == 'unet':
        model = UNet(args.num_channel, args.num_class).to(device)
    elif args.model == 'resnetunet':
        base_net = models.resnet34(pretrained=True)
        base_net.conv1 = torch.nn.Conv2d(1, 64, (7, 7), (2, 2), (3, 3), bias=False)
        model = ResNetUNet(base_net,args.num_class).to(device)
    if args.device == 'cpu':
        print(model)
    else:
        summary(model, input_size=(args.num_channel, args.image_size, args.image_size))    
    # to freeze weights of pretrained resnet layers
    if args.freeze and args.model == 'resnetunet':
        for l in model.base_layers:
            for param in l.parameters():
                param.requires_grad = False

    optimizer_ft = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=args.step_size)
    if args.load:
        model.load_state_dict(torch.load(f"{args.weights}best_metric_model_{args.model}.pth")) 

    model, metric_t, metric_v = train_model(model, optimizer_ft, exp_lr_scheduler, device, args.epochs, colon_dataloader)

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
        default=12,
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
        "--freeze", type=bool, default=True, help="freeze the pretrained weights of resnet"
    )
    parser.add_argument(
        "--load", type=bool, default=False, help="continute training from the best model"
    )
    parser.add_argument(
        "--step-size", type=int, default=50, help="step size of StepLR scheduler"
    )
    parser.add_argument(
        "--model", type=str, default='unet', help="choose the model between unet and resnet+unet; UNet-> unet, Resnet+Unet-> resnetunet"
    )
    args = parser.parse_args()
    main(args)