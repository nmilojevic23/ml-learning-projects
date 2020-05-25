import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # input layer
        self.layer1 = nn.Linear(784, 128)
        # hidden layer
        self.layer2 = nn.Linear(128, 16)
        self.layer3 = nn.Linear(16, 16)
        # output layer
        self.layer4 = nn.Linear(16, 10)

        # activation funcs
        self.relu = nn.ReLU()
        self.LogSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        # HL1 relu activation
        out = self.relu(self.layer1(x))
        # HL2 relu activation
        out = self.relu(self.layer2(out))
        # HL 3
        out = self.relu(self.layer3(out))
        # HL3 relu activation
        out = self.LogSoftmax(self.layer4(out))
        return out


def train(model, train_loader, optimizer, lossFunction, num_epochs=10):
    for epoch in range(num_epochs):
        loss_ = 0
        for images, labels in train_loader:
            # flatten the input images [28, 28] to [1, 784]
            images = images.reshape(-1, 784)

            # forward pass
            output = model(images)
            # Loss at each iteration by comparing to target(label)
            loss = lossFunction(output, labels)

            # Backpropagation gradient of loss
            optimizer.zero_grad()
            loss.backward()

            # updating parameters (weight and bias)
            optimizer.step()

            loss_ += loss.item()

        print('Epoch{}, Training loss:{}'.format(epoch, loss_ / len(train_loader)))


def test(model, test_loader):
    with torch.no_grad():  #  localy disable gradient computation
        correct = 0
        total = 0
        for images, labels in test_loader:
            images = images.reshape(-1, 784)
            out = model(images)
            # print(torch.max(out, 1))
            _, predicted = torch.max(out, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Testing accuracy: {}%'.format(100 * correct / total))


def main():
    # train and test set
    train_set = datasets.MNIST('', download=True, train=True, transform=transforms.ToTensor())
    test_set = datasets.MNIST('', download=True, train=False, transform=transforms.ToTensor())

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=64, shuffle=True)

    model = Net()

    # negative-log-likelihood loss
    lossFunction = nn.NLLLoss()
    # stohastic gradient descend
    optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)

    train(model, train_loader, optimizer, lossFunction, num_epochs=20)
    test(model, test_loader)
    torch.save(model, 'mnist_model.pt')


if __name__ == '__main__':
    main()

