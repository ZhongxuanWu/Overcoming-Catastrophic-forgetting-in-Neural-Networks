import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from model import LinearLayer, Flatten, BaseModel
from synaptic_intelligence import SI
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
    'c': 0,
    'xi': 0,
    'lr':0,
    'acc': 0,
}

param_grid = {
    'c':[0.6],
    'xi':[1.0],
    'lr':[1e-4],
}
param_grid = ParameterGrid(param_grid)

for params in param_grid:
    # retrieve parameters
    c = params['c']
    xi = params['xi']
    lr = params['lr']
    print(c, xi, lr)
    
    # model
    net = BaseModel(28 * 28, 100, 10)
    crit = nn.CrossEntropyLoss()
    args = Args(lr=lr, c=c, xi=xi, batch_size=100, n_epochs=5)
    opt = torch.optim.Adam(net.parameters(), args.lr)
    si = SI(backbone = net, loss = crit, args = args, transform = transforms.ToTensor(), opt = opt)

    # training & testing
    si.net.train()
    for _ in range(args.n_epochs):
        for inputs, targets in tqdm(train_loader):
            inputs, targets = inputs.to(si.device), targets.to(si.device)
            loss = si.observe(inputs, targets, not_aug_inputs=None, SI_reg=True)
    si.end_task(dataset=train_loader)
    si.net.eval()
    print('taskA: MNIST', accu(si.net, test_loader, si.device))
    print('taskB: fashion-MNIST', accu(si.net, f_test_loader, si.device))

    si.net.train()
    for _ in range(args.n_epochs):
        for inputs, targets in tqdm(f_train_loader):
            inputs, targets = inputs.to(si.device), targets.to(si.device)
            loss = si.observe(inputs, targets, not_aug_inputs=None, SI_reg=True)
    si.end_task(dataset=f_train_loader)
    si.net.eval()
    print('taskA: MNIST', accu(si.net, test_loader, si.device))
    print('taskB: fashion-MNIST', accu(si.net, f_test_loader, si.device))

    # record best parameters
    continual_perf = accu(si.net, test_loader, si.device)
    if log['acc'] < continual_perf:
        log['c'] = c
        log['xi'] = xi
        log['lr'] = lr
        log['acc'] = continual_perf

print(log)