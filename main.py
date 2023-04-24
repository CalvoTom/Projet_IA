import torch
from torch import nn

#Modèle 
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class systeme_neuronal(nn.Module):
    def __init__(self):
        super(systeme_neuronal, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 28 * 28, 10)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = systeme_neuronal().to(device)
print(model)

#fonction de pert et optimiseur
perte = torch.nn.CrossEntropyLoss()
optimiseur = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)