from PIL import Image
import torch
import torchvision
import PIL.Image

def transform_tenseur(chm_image):
    """
    Cette fonction transforme une image en tenseur normalisé.
    
    Input:
    - chm_image (str): Le chemin vers l'image à transformer.
    
    Output:
    - torch.Tensor: Le tenseur normalisé de l'image.
    """
    img = PIL.Image.open(chm_image)

    #redimension et transformation
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    tensor_img = transform(img)

    # Ajouter une dimension pour le batch
    tensor_img = tensor_img.unsqueeze(0)
    
    return tensor_img
