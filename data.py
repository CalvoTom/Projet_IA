from torchvision.datasets import CIFAR10
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

def data():
    #Transformation image en tenseur (sans chemin d'accès)
    transformations = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    batch_size = 10
    number_of_labels = 10 
    classes = ('Avion', 'Voiture', 'Oiseau', 'Chat', 'Cerf', 'Chien', 'Grenouille', 'Chevalle', 'Bateau', 'Camion')

    #Donnée de test
    train_set =CIFAR10(root="./data",train=True,transform=transformations,download=True)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)

    #Donnée d'entrainement
    test_set = CIFAR10(root="./data", train=False, transform=transformations, download=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)

    return(number_of_labels,train_loader,test_loader,classes)
