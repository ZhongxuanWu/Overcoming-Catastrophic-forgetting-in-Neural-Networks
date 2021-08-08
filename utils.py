def accu(model, dataloader):
    model = model.eval()
    acc = 0
    for input, target in dataloader:
        o = model(input)
        acc += (o.argmax(dim=1).long() == target).float().mean()
    return acc / len(dataloader)