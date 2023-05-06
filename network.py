import torch.nn as nn
import torch.nn.functional as F

# Définir un réseau de neurones convolutionnel
def reseaux_neuronal():
    class Reseau(nn.Module):
        def __init__(self):
            super(Reseau, self).__init__()
            
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(12)
            self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(12)
            self.pool = nn.MaxPool2d(2,2)
            self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
            self.bn4 = nn.BatchNorm2d(24)
            self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
            self.bn5 = nn.BatchNorm2d(24)
            self.fc1 = nn.Linear(24*10*10, 10)

        def forward(self, entree):
            sortie = F.relu(self.bn1(self.conv1(entree)))      
            sortie = F.relu(self.bn2(self.conv2(sortie)))     
            sortie = self.pool(sortie)                        
            sortie = F.relu(self.bn4(self.conv4(sortie)))     
            sortie = F.relu(self.bn5(self.conv5(sortie)))     
            sortie = sortie.view(-1, 24*10*10)
            sortie = self.fc1(sortie)

            return sortie

    # Instancier un modèle de réseau de neurones
    modele = Reseau()
    return modele
