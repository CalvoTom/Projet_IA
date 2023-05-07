import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from precision import precision_test
from data import data
from network import reseaux_neuronal
from enregistrement import sauvegarderModele
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
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


#Fonction pour tester le model avec des lots d'image
def test():
    images, labels = next(iter(test_loader))
    imageshow(torchvision.utils.make_grid(images))
   
    #Montre les classes que le model devrait trouver
    print('Vrai classes: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size)))
    
    outputs = model(images)
    _, predicted = torch.max(outputs, 1)
    
    ##Montre les classe trouver par le model
    print('Classes trouver: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size)))
    
if __name__ == "__main__":
    
    #Construction du model en fonction d'un nombre d'entrainement donner
    train(5)
    print('Entrainement terminer')
    
    #Test du nouveau model
    path = "Model.pth"
    model.load_state_dict(torch.load(path))
    test()