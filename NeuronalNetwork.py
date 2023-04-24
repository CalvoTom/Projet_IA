import torch
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class systeme_neuronal(nn.Module):
    def __init__(self):
        super(systeme_neuronal, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = systeme_neuronal().to(device)
print(model)