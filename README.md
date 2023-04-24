# Projet_IA
# V.0.1
 - Création d'une fonction qui permet de transformer des images (input) en tenseur (output).
 - Ajout d'une base de donnée de d'entrainemant avec des images trier par catégorie:
 
        Animaux, Humain, Maison, Object, Vêtement, Voiture
- Ajout d'une base de donnée de test avec des nouvelles images trier par catégorie:

        Animaux, Humain, Maison, Object, Vêtement, Voiture
# V.0.2
- Création d'un premier modèle avec:

        Une condition permettant de séléctionner soit de GPU soit le CPU (device) et une class systeme_neuronal.
- Création d'une fonction de pert et d'un optimiseur.

# V.0.3
- Ajout de deux nouvelle couche au réseaux neuronal:

        nn.MaxPool2d et nn.Conv2d
- Importation des bases de donnée d'entrainement et de test (transformer en tenseur avec la fonction transform).
- Modification de la fonction transform pour pouvoir l'utiliser avec le module torchvision.

# V.0.4
- Modification de la méthode "forward" de ma classe reseaux_neuronal pour l'adapter au deux nouvelle couche du réseaux neuronal.

# V.0.5
- Création d'une fonction pour entrainer mon premier modèle a l'aide de la base de donnée de train.