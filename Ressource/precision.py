import torch
from torch.autograd import Variable

# Fonction pour tester le modèle avec l'ensemble de test et afficher la précision pour les images de test
def precision_test(model, test_loader):
    """
    Fonction qui calcule la précision du modèle sur les données de test.

    Input:
    - model (torch.nn.Module): Le modèle à évaluer.
    - test_loader (torch.utils.data.DataLoader): Les données de test.

    Output:
    - float : La précision du modèle sur les données de test.
    """
    model.eval()
    precision = 0.0
    total = 0.0
    with torch.no_grad():
        for donnees in test_loader:
            images, etiquettes = donnees
            # exécute le modèle sur l'ensemble de test pour prédire les étiquettes
            sorties = model(images)
            # l'étiquette avec la plus haute énergie sera notre prédiction
            _, pred = torch.max(sorties.data, 1)
            total += etiquettes.size(0)
            precision += (pred == etiquettes).sum().item()
    # calcule la précision sur toutes les images de test
    precision = (100 * precision / total)
    return precision
