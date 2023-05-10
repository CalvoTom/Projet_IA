import torch
import torchvision
import PIL.Image
from simple_colors import *
from Ressource import data
from Ressource import transform
from Ressource import network

classe = data.data()[3]

def use():
    """
    Cette fonction sert d'interface d'utilisation afin de pouvoir prédire la classe d'une image choisis.

    Input:
    - None

    Output:
    - None
    """
    #ouvrir l'image souhaitée
    print(red("Veuillez glisser votre image dans le dossier ressource", "bold"))
    chm_image = "Ressource/" + input(black("Indiquer le nom de votre image:"))
  
    #Convertit l'image en tensor utilisable par le model
    tensor_img = transform.transform_tenseur(chm_image)
                      
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
