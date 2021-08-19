import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
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

# model
net = BaseModel(28 * 28, 100, 10)
crit = nn.CrossEntropyLoss()
args = Args(lr=1e-4, c=0.5, xi=1.0, batch_size=128, n_epochs=8)
opt = torch.optim.Adam(net.parameters(), args.lr)
si = SI(backbone = net, loss = crit, args = args, transform = transforms.ToTensor(), opt = opt)

# training
si.net.train()
for _ in range(args.n_epochs):
    for inputs, targets in tqdm(train_loader):
        inputs, targets = inputs.to(si.device), targets.to(si.device)
        loss = si.observe(inputs, targets, not_aug_inputs=None, SI_reg=False)
si.end_task(dataset=train_loader)
for _ in range(args.n_epochs):
    for inputs, targets in tqdm(f_train_loader):
        inputs, targets = inputs.to(si.device), targets.to(si.device)
        loss = si.observe(inputs, targets, not_aug_inputs=None, SI_reg=False)
si.end_task(dataset=f_train_loader)

si.net.eval()
print('taskA: MNIST', accu(si.net, test_loader, si.device))
print('taskB: fashion-MNIST', accu(si.net, f_test_loader, si.device))

