import torch

class Args:
    def __init__(self, 
                 lr=None, c=None, xi=None,
                 e_lambda=None, gamma=None,
                 batch_size=None, n_epochs=None) -> None:
        
        self.lr = lr
        self.c = c
        self.xi = xi
        self.e_lambda = e_lambda
        self.gamma = gamma
        self.batch_size = batch_size
        self.n_epochs = n_epochs

def accu(model, dataloader, device='cpu'):
    model = model.eval()
    acc = 0
    for input, target in dataloader:
        input, target = input.to(device), target.to(device)
        o = model(input)
        acc += (o.argmax(dim=1).long() == target).float().mean()
    return acc / len(dataloader)

def get_device() -> torch.device:
    """
    Returns the GPU device if available else CPU.
    """
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")