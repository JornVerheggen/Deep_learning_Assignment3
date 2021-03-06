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
import matplotlib.pyplot as plt


def loadData(batch_size, aug):
    train = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transforms.ToTensor())
    train, validation = torch.utils.data.random_split(train, [50000, 10000])

    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=2)

    validationloader = torch.utils.data.DataLoader(
        validation, batch_size=batch_size, shuffle=True, num_workers=2)

    test = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=True, num_workers=2)

    print(aug)

    return (trainloader, validationloader, testloader)


def Validate(model, loss_func, dataloader):
    lossTotal = 0.0
    accuracyTotal = 0.0
    with torch.no_grad():
        for data, labels in dataloader:
            target = model(data)
            # calc loss
            loss = loss_func(target, labels)
            lossTotal = loss.item() * data.size(0)

            # calc accuracy
            _, predicted = torch.max(target.data, 1)
            accuracyTotal += (predicted == labels).sum().item() / \
                predicted.size(0)

    accuracy = accuracyTotal / len(dataloader)
    finalLoss = lossTotal / len(dataloader)
    return accuracy, finalLoss


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


def train(num_epochs, cnn, trainloader, validationloader, loss_func, optimizer):

    cnn.train()
    data = dict(train=[], validate=[])

    # Train the model
    total_step = len(trainloader)

    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(trainloader):

            # clear gradients for this training step
            optimizer.zero_grad()

            # gives batch data, normalize x when iterate train_loader
            b_x = Variable(images)   # batch x
            b_y = Variable(labels)   # batch y
            output = cnn(b_x)
            loss = loss_func(output, b_y)
            _, predicted = torch.max(output.data, 1)

            # backpropagation, compute gradients
            loss.backward()
            # apply gradients
            optimizer.step()

            # print loss every 100
            if (i+1) % 100 == 0:
                print(
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i+1}/{total_step+1}], Loss: {loss.item():.4f} Acc: {(predicted == b_y).sum().item() / predicted.size(0)}')

        # at each epoch: validate progress with both datasets
        val_accuracy, val_loss = Validate(cnn, loss_func, validationloader)
        print(
            f"Epoch:{epoch}, Validation accuracy:{val_accuracy}, Validation loss: {val_loss}")
        data["validate"].append([epoch, val_accuracy, val_loss])

        train_accuracy, train_loss = Validate(cnn, loss_func, trainloader)
        print(
            f"Epoch:{epoch}, Train accuracy:{train_accuracy}, Train loss: {train_loss}")
        data["train"].append([epoch, train_accuracy, train_loss])
    return data


def test(cnn, testloader, loss_func):
    cnn.eval()
    test_loss = []
    test_accuracy = []
    with torch.no_grad():
        for _, (data, labels) in enumerate(testloader):
            # pass data through network
            outputs = cnn(data)
            _, predicted = torch.max(outputs.data, 1)
            loss = loss_func(outputs, labels)
            test_loss.append(loss.item())
            test_accuracy.append(
                (predicted == labels).sum().item() / predicted.size(0))
        print('test loss: {}, test accuracy: {}'.format(
            np.mean(test_loss), np.mean(test_accuracy)))
    return np.mean(test_accuracy)  # ,test_accuracy, test_loss
