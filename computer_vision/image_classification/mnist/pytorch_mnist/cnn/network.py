import torch
import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class ConvNetV1(nn.Module):
    def __init__(self):
        super(ConvNetV1, self).__init__()
        self.conv1 = nn.Conv2d(
            1,  # number of channels in the input image
            32,  # number of channels produced by the convolution
            3,  # size of a filter that runs over a images
            1  # how far the filter is moved after each computation
         )
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

    def model_summary(self):
        print(summary(self, (1, 28, 28)))


class ConvNetV2(nn.Module):
    def __init__(self):
        super(ConvNetV2, self).__init__()
        # convolutional layers
        # Sequential: a sequential container
        self.layer1 = nn.Sequential(
            nn.Conv2d(
                1,  # number of channels in the input image
                32,  # number of channels produced by the convolution
                kernel_size=5,  # size of a filter that runs over a images
                stride=1,  # how far the filter is moved after each computation
                padding=2  # implicit zero-paddings on both sides, kernel_size - 1 / 2
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        # Dropout:
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear(7 * 7 * 64, 1000)
        self.fc2 = nn.Linear(1000, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)  # .reshape()
        out = self.fc1(out)
        out = self.fc2(out)

        return out