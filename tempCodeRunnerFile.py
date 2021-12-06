    # train_loader = torch.utils.data.DataLoader(torchvision.datasets.MNIST('./data',
    #                                                                       download=True,
    #                                                                       train=True,
    #                                                                       transform=transforms.Compose([
    #                                                                           transforms.ToTensor(),  # first, convert image to PyTorch tensor
    #                                                                           transforms.RandomRotation(
    #                                                                               degrees=(45, -45), fill=(0,))  # normalize inputs
    #                                                                       ])),
    #                                            batch_size=10,
    #                                            shuffle=True)
