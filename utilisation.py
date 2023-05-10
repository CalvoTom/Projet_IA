import torch
import torchvision
import PIL.Image
from simple_colors import *
from Ressource import data
from Ressource import network

classe = data.data()[3]

def use():
    
    #ouvrir l'image souhaitée
    print(red("Veuillez glisser votre image dans le dossier ressource", "bold"))
    chm_image = "Ressource/" + input(black("Indiquer le nom de votre image:"))
    img = PIL.Image.open(chm_image)

    # Transformation de l'image en tenseur
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    tensor_img = transform(img)

    # Ajouter une dimension pour le batch
    tensor_img = tensor_img.unsqueeze(0)
                      
    # Charger le modèle sauvegardé
    modele = network.reseaux_neuronal()
    modele.load_state_dict(torch.load("./Model.pth"))

    # Faire des prédictions avec le modèle
    outputs = modele(tensor_img)
    _, predicted = torch.max(outputs, 1)
    
    # Retourner la prédiction
    print('Votre image semble être : ', ' '.join('%5s' % classe[predicted[j]] 
                              for j in range(1)))

use()
