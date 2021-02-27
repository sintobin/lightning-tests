# https://www.youtube.com/watch?v=OMDn66kM9Qc
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms

# define the model
# model = nn.Sequential(
#     nn.Linear(28 * 28, 64),
#     nn.ReLU(),
#     nn.Linear(64, 64),
#     nn.ReLU(),
#     nn.Dropout(0.1) # if we are overfitting
#     nn.Linear(64, 10)
# ).cuda()

# more flexible model: residual connection
# bypass middle layer with residual connection sometimes, 
# network/optimzer decides whether to use all hidden layers
# results in higher learning speed: loss goes down faster
class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.do = nn.Dropout(0.1) # if we are overfitting

    def forward(self, x):
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(self.l2(h1))
        do = self.do(h2 + h1)
        logits = self.l3(do)
        return logits
    
model = ResNet().cuda()

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
    accuracies = [] # for logging

    model.train() # has an impact on some functionality such as Dropout

    for batch in train_loader:
        x, y = batch

        # x: b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1).cuda()

        ### 5 steps for supervised learning ###
        
        # 1 forward
        l = model(x) # logits
#        import pdb; pdb.set_trace() # interactive debugging
        
        # 2 compute the objective function
        J = loss(l, y.cuda())
        
        # 3 cleaning the gradient
        model.zero_grad()
        
        # 4 accumulate the partial derivatives of J wrt params
        J.backward()
        
        # 5 step in opposite direction of the gradient
        optimizer.step()
        
        losses.append(J.item())
        accuracies.append(y.cuda().eq(l.detach().argmax(dim=1)).float().mean())
  
    print(f'Epoch {epoch+1}', end =', ')
    print(f'training loss:\t\t {torch.tensor(losses).mean():.2f}', end =', ')
    print(f'training accuracy:\t {torch.tensor(accuracies).cuda().mean():.2f}', end ='\n')


    losses = [] # for logging
    accuracies = [] # for logging
    
    model.eval()

    for batch in val_loader:
        x, y = batch

        # x: b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1).cuda()

        ### 5 steps for supervised learning ###
        # under-the-hood: gives the underlying idea, code will not work as is
        
        # 1 forward
        with torch.no_grad():
            l = model(x) # logit
        
        # 2 compute the objective function
        J = loss(l, y.cuda())
         
        losses.append(J.item())
        accuracies.append(y.cuda().eq(l.detach().argmax(dim=1)).float().mean())
  
    print(f'Epoch {epoch+1}', end =', ')
    print(f'validation loss:\t {torch.tensor(losses).mean():.2f}', end =', ')
    print(f'validation accuracy:\t {torch.tensor(accuracies).cuda().mean():.2f}', end ='\n\n')