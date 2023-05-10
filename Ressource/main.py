import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from precision import precision_test
from data import data
from network import reseaux_neuronal
from enregistrement import sauvegarder_modele
from train import train
from network import reseaux_neuronal

#initailisation des variables
train_loader = data()[1]
test_loader = data()[2]
classes = data()[3]
batch_size = data()[4]

model = reseaux_neuronal()

#Fonction pour montrer les images
def imageshow(img):
    """
    Cette fonction affiche une image à partir de son tenseur
    
    Input:
    - img (torch.Tensor) : le tenseur représentant l'image à afficher
    
    Output:
    - None
    """
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


#Fonction pour tester le model avec des lots d'image
def test():
    """
    Cette fonction permet de tester le modèle entraîné sur un batch d'images de test. Elle affiche les vraies classes
    et les classes prédites par le modèle pour chaque image.
    
    Input:
    - None
    
    Output:
    - None
    """
    images, labels = next(iter(test_loader))
    imageshow(torchvision.utils.make_grid(images))
   
    #Montre les classes que le model devrait trouver
    print('Vrai classes: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size)))
    
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    #Montre les classe trouver par le model
    print('Classes trouver: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size)))
    
if __name__ == "__main__":
    
    #Construction du model en fonction d'un nombre d'entrainement donner
    train(3)
    print('Entrainement terminer')
    
    #Test du nouveau model
    sauvegarder_modele(model)
    test()