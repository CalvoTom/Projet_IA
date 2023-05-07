import torch
import torchvision
import PIL.Image
from simple_colors import *
from Ressource import transform

def predict_image_class():

    print(red("Veuillez glisser votre image dans le dossier ressource", "bold"))
    chm_image = "Ressource/" + input(black("Indiquer le nom de votre image:"))
    image = PIL.Image.open(chm_image)
                      
    # Charger le modèle sauvegardé
    model_state_dict = torch.load("Model.pth", map_location=torch.device('cpu'))
    model = torchvision.models.resnet18()
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model.load_state_dict(model_state_dict)
    model.eval()

    
    # Pré-traitement de l'image
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.Resize((224, 224)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    image_tensor = preprocess(image)
    image_batch = image_tensor.unsqueeze(0)
    
    # Faire des prédictions avec le modèle
    with torch.no_grad():
        outputs = model.forward(image_batch)
    _, predicted = torch.max(outputs, 1)
    predicted_class = predicted.item()
    
    # Retourner la prédiction
    print(predicted_class)