from __future__ import print_function, division
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import matplotlib.pylab as plt

from data_loader import load_data
from train import train_model
from utils import visualize_model


def cnn_finetune(device, dataloaders, dataset_sizes, class_names):
    """
        Load a pretrained model and reset final fully conected layer
    """
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features

    model_ft.fc = nn.Linear(num_ftrs, 2)

    model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

    # decay lr by factor of 0.1 after each 7 epochas
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, device, dataloaders, dataset_sizes, num_epochs=25)

    visualize_model(model_ft, device, dataloaders, class_names)


def cnn_feature_extractor(device, dataloaders, dataset_sizes, class_names):
    """
        Freeze all nn except last layer.
        Requires requires_grade=False to freeze all the parameters so that the gradients are not computed in backward().
    """
    model_conv = models.resnet18(pretrained=True)
    # keep parameters in all layers except fully connected layer
    for param in model_conv.parameters():
        requires_grade = False

    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model_conv.fc.in_features
    model_conv.fc = nn.Linear(num_ftrs, 2)

    model_conv.to(device)
    criterion = nn.CrossEntropyLoss()

    # only final layer params
    optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)

    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    model_conv = train_model(model_conv, criterion, optimizer_conv, exp_lr_scheduler, device, dataloaders,
                             dataset_sizes, num_epochs=25)

    visualize_model(model_conv, device, dataloaders, class_names)


def main():
    parser = argparse.ArgumentParser(description='PyTorch Transfer learning tutorial')
    parser.add_argument(
        'mode',
        choices=['train', 'eval'],
        help="Execution mode train | eval"
    )
    # train
    parser.add_argument('tl_type', choices=['finetune', 'feature_extractor'],
                        help='CNN Transfered learning scenarious')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    args = parser.parse_args()

    plt.ion()  # interactive mode

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dataloaders, dataset_sizes, class_names = load_data()

    if args.mode == 'train':
        if args.tl_type == 'finetune':
            cnn_finetune(device, dataloaders, dataset_sizes, class_names)
        elif args.tl_type == 'feature_extractor':
            cnn_feature_extractor(device, dataloaders, dataset_sizes, class_names)


if __name__ == '__main__':
    main()
