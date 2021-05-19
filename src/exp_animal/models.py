import torch
import torch.nn as nn
from torchvision import models
torch.set_default_dtype(torch.float32)

resnet18 = models.resnet18(pretrained=True)

def get_net(name):
    if name == 'WILDCAM':
        return resnet18_extractor


class resnet18_extractor(nn.Module):
    def __init__(self, n_classes=2):
        super(resnet18_extractor, self).__init__()
        image_modules = list(resnet18.children())[:-1]
        self.model = nn.Sequential(*image_modules)
        for param in self.model.parameters():
            param.requires_grad = False
        num_ftrs = resnet18.fc.in_features #512
        self.fc = nn.Sequential(
            # nn.Linear(num_ftrs, 200),
            # nn.ReLU(),
            # nn.Linear(200, 200),
            # nn.ReLU(),
            # nn.Linear(200, 1)
            nn.Linear(num_ftrs, 10),
            nn.Sigmoid(),
            nn.Linear(10, 1)
            # nn.Linear(num_ftrs, 80),
            # nn.Sigmoid(),
            # nn.Linear(80, 1)
            )  

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 512)
        x = self.fc(x)
        return x

    def get_embedding_dim(self):
        return 512