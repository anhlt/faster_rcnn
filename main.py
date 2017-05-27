from torchvision.datasets import CIFAR100
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.trainer import Trainer
from torch.optim import Adam
from torch import nn
import torch.nn.init as init
from torch.nn import Module
import torch.nn.functional as F
import torch
from torch.utils.trainer.plugins import ProgressMonitor, LossMonitor, Logger
import numpy as np


data = CIFAR100('./data', download=True, transform=transforms.Compose([
    transforms.ToTensor()
]))
data_loader = DataLoader(data, batch_size=32, shuffle=True, num_workers=4)


class Net(Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, kernel_size=5)
        init.xavier_uniform(self.conv1.weight, gain=np.sqrt(2))
        init.constant(self.conv1.bias, 0.1)
        self.conv2 = nn.Conv2d(20, 20, kernel_size=5)
        init.xavier_uniform(self.conv2.weight, gain=np.sqrt(2))
        init.constant(self.conv2.bias, 0.1)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(500, 100)
        self.fc2 = nn.Linear(100, 100)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


model = Net()
print(model)

optimizer = Adam(model.parameters())
trainer = Trainer(model=model, criterion=torch.nn.CrossEntropyLoss(),
                  optimizer=optimizer, dataset=data_loader)

trainer.register_plugin(ProgressMonitor())
trainer.register_plugin(LossMonitor())
trainer.register_plugin(Logger(fields=["loss.last"], interval=[
                        (100, 'iteration'), (1, 'epoch')]))
trainer.run(epochs=100)

print(trainer.stats)
