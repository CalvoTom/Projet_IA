import torch
from torch import nn
from transform import transform_tenseur
import torchvision

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
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        return x

model = systeme_neuronal().to(device)
print(model)

#fonction de pert et optimiseur
perte = torch.nn.CrossEntropyLoss()
optimiseur = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#bases de donnée d'entrainement et de test
trainset = torchvision.datasets.ImageFolder(root='Data Train/Train', transform = transform_tenseur())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.ImageFolder(root='Data Test/Test', transform = transform_tenseur())
testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)