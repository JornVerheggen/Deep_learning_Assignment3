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

dataAgumentations = [
    [transforms.ToTensor()],  # baseline: no agumentations
    [transforms.RandomRotation(
        degrees=(-90, 90), fill=(0,)), transforms.ToTensor()],
    [transforms.RandomAdjustSharpness(0.9, p=0.001), transforms.ToTensor()],
    [transforms.RandomPerspective(), transforms.ToTensor()],
    [transforms.GaussianBlur(7), transforms.ToTensor()],
    [transforms.RandomAffine(degrees=20, translate=(
        0.1, 0.1), scale=(0.9, 1.1)), transforms.ToTensor()],
    [transforms.ColorJitter(brightness=0.2, contrast=0.2),
     transforms.ToTensor()]
]

if __name__ == '__main__':
    batch_size = 16  # 16 for final
    epochs = 1

    inSampleLoss = []
    outSampleAccuracy = []

    for aug in dataAgumentations:
        cnn = part_2_func.initNetwork(batch_size)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn.parameters(), lr=0.0005)
        trainloader, testloader = part_2_func.loadData(batch_size, aug)
        inSampleLoss.append(part_2_func.train(
            epochs, cnn, trainloader, loss_func, optimizer))
        result = part_2_func.test(cnn, testloader, loss_func)
        outSampleAccuracy.append(result)
        print(result)
