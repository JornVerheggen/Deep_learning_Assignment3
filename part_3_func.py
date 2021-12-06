import numpy as np
import torchvision
import torch
import torchvision.transforms
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib as plt
from torch.autograd import Variable


def loadData(batch_size, aug):
    train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transforms.compose(aug))
    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=2)
    test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=True, num_workers=2)

    print(aug)

    return (trainloader, testloader)


def initNetwork(batch_size):
    class Net(nn.Module):
        def __init__(self):
            self.batch_size = batch_size
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64*3*3, 10)

        def forward(self, x):
            assert x.shape == (
                self.batch_size, 1, 28, 28), f'expected {(self.batch_size, 1, 28, 28)} but got: {x.shape}'
            x = self.pool(F.relu(self.conv1(x)))
            assert x.shape == (
                self.batch_size, 16, 14, 14), f'OUR Assert expected {(self.batch_size, 16, 14, 14)} but got: {x.shape}'
            x = self.pool(F.relu(self.conv2(x)))
            assert x.shape == (
                self.batch_size, 32, 7, 7), f'OUR Assert expected {(self.batch_size, 32, 7, 7)} but got: {x.shape}'
            x = self.pool(F.relu(self.conv3(x)))

            assert x.shape == (
                self.batch_size, 64, 3, 3), f'OUR Assert expected {(self.batch_size, 64, 3, 3)} but got: {x.shape}'

            x = torch.reshape(x, (self.batch_size, 64 * 3 * 3))
            x = self.fc1(x)

            return x
    return Net()


def train(num_epochs, cnn, trainloader):
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.0005)

    num_epochs = 10

    cnn.train()

    # Train the model
    total_step = len(trainloader)

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(trainloader):

            optimizer.zero_grad()

            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)
            loss = loss_func(output, b_y)

            # clear gradients for this training step

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            if (i+1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1, num_epochs, i + 1, total_step, loss.item()))
                pass

        pass

    def test2():
        cnn.eval()
        test_loss = []
        test_accuracy = []
        with torch.no_grad():
            for i, (data, labels) in enumerate(testloader):
                # pass data through network
                outputs = cnn(data)
                _, predicted = torch.max(outputs.data, 1)
                loss = loss_func(outputs, labels)
                test_loss.append(loss.item())
                test_accuracy.append(
                    (predicted == labels).sum().item() / predicted.size(0))
            print('test loss: {}, test accuracy: {}'.format(
                np.mean(test_loss), np.mean(test_accuracy)))
        return np.mean(test_accuracy)
