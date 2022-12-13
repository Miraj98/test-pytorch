import torch
import torchvision
from datetime import datetime

w1 = torch.randn((30, 784), requires_grad=True)
b1 = torch.randn((30, 1), requires_grad=True)
w2 = torch.randn((10, 30), requires_grad=True)
b2 = torch.randn((10, 1), requires_grad=True)

train_loader = torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ]))


batch_size = 10
epochs = 30
total = train_loader.__len__()
batches = total // batch_size

for e in range(epochs):
    start = datetime.now()
    for i in range(batches):
        for j in range(batch_size):
            (x, y) = train_loader[i*batch_size + j]
            x = x.reshape(784, 1)
            x = w1.matmul(x).add(b1).sigmoid()
            x = w2.matmul(x).add(b2).sigmoid()
            criterion = torch.nn.MSELoss();
            _y = torch.zeros(10, 1)
            _y[y] = 1.
            loss = criterion(x, _y)
            loss.backward()
    elapsed = datetime.now() - start
    print("time taken for epoch", e, "is", elapsed, "ms")


