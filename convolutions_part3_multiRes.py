
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
import part_3_func

if __name__ == '__main__':
    batch_size = 16  # 16 for final
    epochs = 5

    for imageSize in [32, 48, 64]:
        print(f"Running with image size {imageSize}")

        inSampleLoss = []
        inSampleAccuracy = []
        outSampleAccuracy = []

        # set up network and optimizers
        cnn = part_3_func.initNetwork(batch_size)
        loss_func = nn.CrossEntropyLoss()
        optimizer = optim.Adam(cnn.parameters(), lr=0.0005)

        # train network on augmented data
        trainloader, testloader = part_3_func.loadDataMultires(
            batch_size, imageSize)

        loss, acc = part_3_func.train(
            epochs, cnn, trainloader, loss_func, optimizer)
        inSampleLoss.append(loss)
        inSampleAccuracy.append(acc)

        # use test data to eval model
        result = part_3_func.test(cnn, testloader, loss_func)
        outSampleAccuracy.append(result)
