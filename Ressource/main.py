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

train_loader = data()[1]
test_loader = data()[2]
classes = data()[3]
batch_size = data()[4]

model = reseaux_neuronal()

# Function to show the images
def imageshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Function to test the model with a batch of images and show the labels predictions
def testBatch():
    # get batch of images from the test DataLoader  
    images, labels = next(iter(test_loader))

    # show all images as one image grid
    imageshow(torchvision.utils.make_grid(images))
   
    # Show the real labels on the screen 
    print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
                               for j in range(batch_size)))
  
    # Let's see what if the model identifiers the  labels of those example
    outputs = model(images)
    
    # We got the probability for every 10 labels. The highest (max) probability should be correct label
    _, predicted = torch.max(outputs, 1)
    
    # Let's show the predicted labels on the screen to compare with the real ones
    print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
                              for j in range(batch_size)))
    
if __name__ == "__main__":
    
    # Let's build our model
    train(5)
    print('Finished Training')
    
    # Let's load the model we just created and test the accuracy per label 
    path = "Model.pth"
    model.load_state_dict(torch.load(path))

    # Test with batch of images
    testBatch()