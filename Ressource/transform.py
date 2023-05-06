from PIL import Image
import torch
from torchvision import transforms

def transform_tenseur(chm_image):
    """
    Cette fonction transforme une image en tenseur normalisé. Elle prend en paramètre le chemin vers cette image.
    """
    # Ouvrir l'image et vérifier la taille
    image = Image.open(chm_image)
    if image.size != (224, 224):
        image = image.resize((224, 224))
    
    # Convertir l'image en mode RVB et appliquer les transformations
    image = image.convert('RGB')
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    tensor = transform(image)
    return tensor
