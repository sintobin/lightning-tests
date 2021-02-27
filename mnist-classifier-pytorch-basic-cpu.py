# https://www.youtube.com/watch?v=OMDn66kM9Qc
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms

# define the model
model = nn.Sequential(
    nn.Linear(28 * 28, 64),
    nn.ReLU(),
    nn.Linear(64, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)

# define the optimizer
params = model.parameters() # under-the-hood
optimizer = optim.SGD(params, lr=1e-2)

# define the loss
loss = nn.CrossEntropyLoss()

# data, train and val split
train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size=32)
val_loader = DataLoader(val, batch_size=32)

# training loop
nb_epochs = 5
for epoch in range(nb_epochs):

    losses = [] # for logging

    for batch in train_loader:
        x, y = batch

        # x: b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1)

        ### 5 steps for supervised learning ###
        # under-the-hood: gives the underlying idea, code will not work as is
        
        # 1 forward
        l = model(x) # logit
        
        # 2 compute the objective function
        J = loss(l, y)
        
        # 3 cleaning the gradient
        model.zero_grad()
        # under-the-hood:
        # params.grad._zero()
        
        # 4 accumulate the partial derivatives of J wrt params
        J.backward()
        # under-the-hood:
        # params.grad.add_(dJ/dparams)
        
        # 5 step in opposite direction of the gradient
        optimizer.step()
        # under-the-hood
        # with torch.no_grad(): params = params - eta * params.grad
        
        losses.append(J.item())

    print(f'Epoch {epoch+1}, training loss: {torch.tensor(losses).mean():.2f}')
    

    losses = [] # for logging

    for batch in val_loader:
        x, y = batch

        # x: b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1)

        ### 5 steps for supervised learning ###
        # under-the-hood: gives the underlying idea, code will not work as is
        
        # 1 forward
        with torch.no_grad():
            l = model(x) # logit
        
        # 2 compute the objective function
        J = loss(l, y)
         
        losses.append(J.item())

    print(f'Epoch {epoch+1}, validation loss: {torch.tensor(losses).mean():.2f}')