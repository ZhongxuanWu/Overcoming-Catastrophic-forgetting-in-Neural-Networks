import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from model import LinearLayer, Flatten, BaseModel
from baselines import EwcOn
from utils import Args, accu

# data
mnist_train = datasets.MNIST(root="./", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root="./", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

f_mnist_train = datasets.FashionMNIST("./", train=True, download=True, transform=transforms.ToTensor())
f_mnist_test = datasets.FashionMNIST("./", train=False, download=True, transform=transforms.ToTensor())
f_train_loader = DataLoader(f_mnist_train, batch_size = 100, shuffle=True)
f_test_loader = DataLoader(f_mnist_test, batch_size = 100, shuffle=False)

log = {
    'e_lambda': 0,
    'gamma': 0,
    'lr':0,
    'acc': 0,
}

param_grid = {
    'e_lambda': [0.7],
    'gamma': [1.0],
    'lr':[1e-4],
}
param_grid = ParameterGrid(param_grid)

for params in param_grid:
    # retrieve parameters
    e_lambda = params['e_lambda']
    gamma = params['gamma']
    lr = params['lr']
    print(params)
    
    # model
    net = BaseModel(28 * 28, 100, 10)
    crit = nn.CrossEntropyLoss()
    args = Args(lr=lr, e_lambda=e_lambda, gamma=gamma, batch_size=100, n_epochs=5)
    opt = torch.optim.Adam(net.parameters(), args.lr)
    ewcon = EwcOn(backbone = net, loss = crit, args = args, transform = transforms.ToTensor(), opt = opt)

    # training & testing
    ewcon.net.train()
    for _ in range(args.n_epochs):
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(ewcon.device), targets.to(ewcon.device)
            loss = ewcon.observe(inputs, targets, not_aug_inputs=None)
    ewcon.end_task(dataset=train_loader)
    ewcon.net.eval()
    print('taskA: MNIST', accu(ewcon.net, test_loader, ewcon.device))
    print('taskB: fashion-MNIST', accu(ewcon.net, f_test_loader, ewcon.device))

    ewcon.net.train()
    for _ in range(args.n_epochs):
        for inputs, targets in tqdm(f_train_loader):
            inputs, targets = inputs.to(ewcon.device), targets.to(ewcon.device)
            loss = ewcon.observe(inputs, targets, not_aug_inputs=None)
    ewcon.end_task(dataset=f_train_loader)
    ewcon.net.eval()
    print('taskA: MNIST', accu(ewcon.net, test_loader, ewcon.device))
    print('taskB: fashion-MNIST', accu(ewcon.net, f_test_loader, ewcon.device))

    # record best parameters
    continual_perf = accu(ewcon.net, test_loader, ewcon.device)
    if log['acc'] < continual_perf:
        log['c'] = c
        log['xi'] = xi
        log['lr'] = lr
        log['acc'] = continual_perf

print(log)