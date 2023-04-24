import torch
import torchvision.transforms as transforms

def transform(img):
    transform = transforms.Compose(
        [transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor