import torch
from torch.autograd import Variable

#Fonction pour sauvegarder le modèle
def sauvegarderModele(model):
    chemin = "./Model.pth"
    torch.save(model.state_dict(), chemin)