from __future__ import print_function
import os
import argparse
import torch
import time
import torchvision
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR

from network import ConvNetV1
from train import train
from test import test, evaluate
from data_loader import mnist_data_loader
# from .utils import timer


def main():
    parser = argparse.ArgumentParser(description='PyTorch MNIST CNN Example')
    parser.add_argument(
        'mode',
        choices=['train', 'eval'],
        help="Execution mode train | eval | infer"
    )
    # train
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    print(f"Using device: {device}", end='\n')
    print('-' * 80, end='\n')

    # Load MNIST data
    train_loader, test_loader = mnist_data_loader(args, kwargs)

    # Model
    model = ConvNetV1().to(device)
    print(model.model_summary())
    print('-' * 80, end='\n')

    if "train" in args.mode:
        # Optimizer
        optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
        # LR Scheduler
        scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
        writer = SummaryWriter('tensorboard/mnist_experiment_1')

        # Train
        st = time.time()
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch, writer)
            test(args, model, device, test_loader, writer)
            scheduler.step()
        et = time.time() - st
        print('Training time: {:.2f}s | {:.2f}min'.format(et, et / 60))

        # Torchvision
        images, labels = next(iter(train_loader))
        grid = torchvision.utils.make_grid(images)
        writer.add_image('images', grid, 0)
        writer.add_graph(model, images)
        writer.close()

        if args.save_model:
            os.makedirs('model', exist_ok=True)
            torch.save(model.state_dict(), "./model/mnist_cnn.pt")
    elif "eval" in args.mode:
        model.load_state_dict(torch.load("./model/mnist_cnn.pt"))
        st = time.time()
        evaluate(args, model, device, test_loader)
        et = time.time() - st
        print('Evaluation time: {:.2f}s | {:.2f}min'.format(et, et / 60))


if __name__ == '__main__':
    main()
