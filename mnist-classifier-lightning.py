# https://www.youtube.com/watch?v=DbESHcCoWbM
import torch
from torch import nn
from torch import optim
from torch.utils import data
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from torchvision import datasets, transforms

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import accuracy

class ResNet(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(28 * 28, 64)
        self.l2 = nn.Linear(64, 64)
        self.l3 = nn.Linear(64, 10)
        self.do = nn.Dropout(0.1)
        
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        h1 = nn.functional.relu(self.l1(x))
        h2 = nn.functional.relu(self.l2(h1))
        do = self.do(h2 + h1)
        logits = self.l3(do)
        return logits
    
    def configure_optimizers(self): # :todo:
        # can have multiple optimizers, each has its own training loop
        optimizer = optim.SGD(self.parameters(), lr=1e-2)
        return optimizer
    
    def training_step(self, batch, batch_idx):
        x, y = batch

        # x: b x 1 x 28 x 28
        b = x.size(0)
        x = x.view(b, -1)
        
        # 1 forward
        logits = self(x) # logits
        
        # 2 compute the objective function
        J = self.loss(logits, y)

        acc = accuracy(logits, y)
        pbar = {'train_acc': acc}
        return {'loss': J, 'progress_bar': pbar}
    
    def validation_step(self, batch, batch_idx):
        results = self.training_step(batch, batch_idx)
        return results
    
    def validation_epoch_end(self, val_step_outputs):
        # [results, results, results, results, results]
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['progress_bar']['train_acc'] for x in val_step_outputs]).mean()
        
        pbar = {'avg_val_acc': avg_val_acc}
        return {'val_loss': avg_val_loss, 'progess_bar': pbar}
    
    def prepare_data(self):
        datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())

    def setup(self):
        dataset = datasets.MNIST('data', train=True, download=False, transform=transforms.ToTensor())
        # assignment should work within setup
        self.train, self.val = random_split(dataset, [55000, 5000])
        
    def train_dataloader(self):
        train_loader = DataLoader(self.train, batch_size=32)
        return train_loader
        
    def val_dataloader(self):
        val_loader = DataLoader(self.val, batch_size=32)
        return val_loader


model = ResNet()

trainer = pl.Trainer()
trainer.fit(model)