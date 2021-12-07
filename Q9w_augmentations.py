import pandas
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
import part_2_func


if __name__ == '__main__':
    batch_size = 16  # 16 for final
    epochs = 5

    aug = [transforms.RandomAdjustSharpness(0.9, p=0.001),
           transforms.ColorJitter(brightness=0.2, contrast=0.2),
           transforms.GaussianBlur(7), transforms.ToTensor()],

    # set up network and optimizers
    cnn = part_2_func.initNetwork(batch_size)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.0005)

    # train network on augmented data
    trainloader, validationloader, testloader = part_2_func.loadData(
        batch_size, aug)

    trainResults = part_2_func.train(
        epochs, cnn, trainloader, validationloader, loss_func, optimizer)

    # use test data to eval model
    modelAccuracy = part_2_func.test(
        cnn, testloader, loss_func)
    print(f"{modelAccuracy=}")
    test_accuracy, test_loss = part_2_func.Validate(cnn, loss_func, testloader)
    print(f"{test_accuracy=}")
    print(f"{test_loss=}")

    def PlotLossAcc(result):
        epochs = [x[0]+1 for x in result["train"]]
        plt.plot(epochs, [x[1]
                          for x in result["train"]], marker='o', label='train')
        plt.plot(epochs, [x[1] for x in result["validate"]],
                 marker='*', label='validate')
        plt.title(f"Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.savefig("Q9_Accuracy.pdf")
        plt.show()

        plt.plot(epochs, [x[2]
                          for x in result["train"]], marker='o', label='train')
        plt.plot(epochs, [x[2] for x in result["validate"]],
                 marker='*', label='validate')
        plt.title(f"Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("Q9_Loss.pdf")
        plt.show()

    PlotLossAcc(trainResults)
