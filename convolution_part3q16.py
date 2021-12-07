
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

    # set up network and optimizers
    cnn = part_3_func.initNetwork2(batch_size)
    loss_func = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=0.0005)

    # train network on augmented data
    trainloader32, validationloader32, testloader32 = part_3_func.loadDataMultiresNoResize(
        batch_size, 32)
    trainloader48, validationloader48, testloader48 = part_3_func.loadDataMultiresNoResize(
        batch_size, 48)
    trainloader64, validationloader64, testloader64 = part_3_func.loadDataMultiresNoResize(
        batch_size, 64)

    trainResult32 = part_3_func.train(
        epochs, cnn, trainloader32, validationloader32, loss_func, optimizer)
    trainResult48 = part_3_func.train(
        epochs, cnn, trainloader48, validationloader48, loss_func, optimizer)
    trainResult64 = part_3_func.train(
        epochs, cnn, trainloader64, validationloader64, loss_func, optimizer)

    # use test data to eval model
    result32 = part_3_func.test(cnn, testloader32, loss_func)
    result48 = part_3_func.test(cnn, testloader48, loss_func)
    result64 = part_3_func.test(cnn, testloader64, loss_func)

    # plot
    #part_3_func.PlotLossAcc(trainResult64, "Q15_trainResult64_")
