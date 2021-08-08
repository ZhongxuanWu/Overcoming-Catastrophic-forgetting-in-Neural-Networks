import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from model import LinearLayer, Flatten, BaseModel
from utils import accu
from elastic_weight_consolidation import ElasticWeightConsolidation

# data
mnist_train = datasets.MNIST(root="./", train=True, download=True, transform=transforms.ToTensor())
mnist_test = datasets.MNIST(root="./", train=False, download=True, transform=transforms.ToTensor())
train_loader = DataLoader(mnist_train, batch_size = 100, shuffle=True)
test_loader = DataLoader(mnist_test, batch_size = 100, shuffle=False)

f_mnist_train = datasets.FashionMNIST("./", train=True, download=True, transform=transforms.ToTensor())
f_mnist_test = datasets.FashionMNIST("./", train=False, download=True, transform=transforms.ToTensor())
f_train_loader = DataLoader(f_mnist_train, batch_size = 100, shuffle=True)
f_test_loader = DataLoader(f_mnist_test, batch_size = 100, shuffle=False)

# training
crit = nn.CrossEntropyLoss()
ewc = ElasticWeightConsolidation(BaseModel(28 * 28, 100, 10), crit=crit, lr=1e-4)

for _ in range(8):
    for input, target in tqdm(train_loader):
        ewc.forward_backward_update(input, target, EWC_reg=False)
ewc.register_ewc_params(mnist_train, 100, 300)
for _ in range(8):
    for input, target in tqdm(f_train_loader):
        ewc.forward_backward_update(input, target, EWC_reg=False)
ewc.register_ewc_params(f_mnist_train, 100, 300)

print('taskA: MNIST', accu(ewc.model, test_loader))
print('taskB: fashion-MNIST', accu(ewc.model, f_test_loader))
