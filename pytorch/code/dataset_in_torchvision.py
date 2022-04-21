import torchvision

train_set = torchvision.datasets.CIFAR10("./dataset_data", train=True, download=True)
test_set = torchvision.datasets.CIFAR10("./dataset_data", train=False, download=True)