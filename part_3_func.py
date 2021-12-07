import numpy as np
import torchvision
import torch
import torchvision.transforms
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.autograd import Variable
import matplotlib.pyplot as plt


def loadDataMultires(batch_size, size):
    train = torchvision.datasets.ImageFolder(
        root='./repo/mnist-varres_copy/'+str(size)+'/train/',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(28),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()
        ]))

    splitLen = int(len(train)*0.8)
    train, validation = torch.utils.data.random_split(
        train, [splitLen, len(train)-splitLen])

    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=4)

    validationloader = torch.utils.data.DataLoader(
        validation, batch_size=batch_size, shuffle=True, num_workers=4)

    test = torchvision.datasets.ImageFolder(
        root='./repo/mnist-varres_copy/'+str(size)+'/test/',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(28),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()
        ]))

    testloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=True, num_workers=4)

    return (trainloader, validationloader, testloader)


def loadDataMultiresNoResize(batch_size, size):
    train = torchvision.datasets.ImageFolder(
        root='./repo/mnist-varres_copy/'+str(size)+'/train/',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()
        ]))

    splitLen = int(len(train)*0.8)
    train, validation = torch.utils.data.random_split(
        train, [splitLen, len(train)-splitLen])

    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=4)

    validationloader = torch.utils.data.DataLoader(
        validation, batch_size=batch_size, shuffle=True, num_workers=4)

    test = torchvision.datasets.ImageFolder(
        root='./repo/mnist-varres_copy/'+str(size)+'/test/',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()
        ]))

    testloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=True, num_workers=4)

    return (trainloader, validationloader, testloader)


def loadData(batch_size):
    train = torchvision.datasets.ImageFolder(
        root='./repo/mnist-varres/train/',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(28),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()
        ]))

    trainloader = torch.utils.data.DataLoader(
        train, batch_size=batch_size, shuffle=True, num_workers=4)

    test = torchvision.datasets.ImageFolder(
        root='./repo/mnist-varres/test/',
        transform=torchvision.transforms.Compose([
            torchvision.transforms.Resize(28),
            torchvision.transforms.Grayscale(),
            torchvision.transforms.ToTensor()
        ]))

    testloader = torch.utils.data.DataLoader(
        test, batch_size=batch_size, shuffle=True, num_workers=4)

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
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = torch.reshape(x, (x.shape[0], 64 * 3 * 3))
            x = self.fc1(x)

            return x
    return Net()


def initNetwork2(batch_size):
    class Net(nn.Module):
        def __init__(self):
            self.batch_size = batch_size
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64, 10)

        def forward(self, x):
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            x = self.pool(F.relu(self.conv3(x)))
            x = torch.max(torch.max(x, -1).values, -1)[0]
            x = self.fc1(x)

            return x
    return Net()


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


def train(num_epochs, cnn, trainloader, validationloader, loss_func, optimizer):

    cnn.train()
    data = dict(train=[], validate=[])

    # Train the model
    total_step = len(trainloader)

    for epoch in range(num_epochs):

        for i, (images, labels) in enumerate(trainloader):
            if i >= total_step-46:
                break
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
                    f'Epoch [{epoch + 1}/{num_epochs}], Step [{i+1}/{total_step+1}], Loss: {loss.item():.4f}, Acc: {(predicted == b_y).sum().item() / predicted.size(0)}')

        # collect data at end of epoch for analytics
        # at each epoch: validate progress with both datasets
        val_accuracy, val_loss = Validate(cnn, loss_func, validationloader)
        print(
            f"Epoch:{epoch+1}, Validation accuracy:{val_accuracy}, Validation loss: {val_loss}")
        data["validate"].append([epoch, val_accuracy, val_loss])

        train_accuracy, train_loss = Validate(cnn, loss_func, trainloader)
        print(
            f"Epoch:{epoch+1}, Train accuracy:{train_accuracy}, Train loss: {train_loss}")
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
    return np.mean(test_accuracy)


def PlotLossAcc(result, titlePrefix=""):
    epochs = [x[0]+1 for x in result["train"]]
    plt.plot(epochs, [x[1]
                      for x in result["train"]], marker='o', label='train')
    plt.plot(epochs, [x[1] for x in result["validate"]],
             marker='*', label='validate')
    plt.title(f"Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(titlePrefix+"_Accuracy.pdf")
    plt.show()

    plt.plot(epochs, [x[2]
                      for x in result["train"]], marker='o', label='train')
    plt.plot(epochs, [x[2] for x in result["validate"]],
             marker='*', label='validate')
    plt.title(f"Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(titlePrefix+"_Loss.pdf")
    plt.show()
