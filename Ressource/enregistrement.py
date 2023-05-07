import torch
from torch.autograd import Variable

#Fonction pour sauvegarder le modèle
def sauvegarder_modele(model):
    """
    Sauvegarde un modèle.

    Input:
    - model (torch.nn.Module): Le modèle à sauvegarder.

    Output:
    - None
    """
    chemin = "./Model.pth"
    torch.save(model.state_dict(), chemin)