import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam
from precision import precision_test
from data import data
from network import reseaux_neuronal

train_loader = data()[1]
test_loader = data()[2]

model = reseaux_neuronal()

fonction_perte = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

#Fonction d'entraînement
def train(num_epochs):
    
    best_accuracy = 0.0

    #Définir le périphérique d'exécution
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.to(device)

    for epoch in range(num_epochs):
        perte_epoch = 0.0
        precision_epoch = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            optimizer.zero_grad()
            outputs = model(images)
            loss = fonction_perte(outputs, labels)
            loss.backward()

            #ajuster les paramètres en fonction des gradients calculés
            optimizer.step()

            #Affiche les statistiques pour chaque 1 000 images
            perte_epoch += loss.item()
            if i % 1000 == 999:    
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, perte_epoch / 1000))
                perte_epoch = 0.0

        precision = precision_test(model, test_loader)
        
        #Enregistre le modele si la precision est meilleur
        if precision > best_accuracy:
            enregistre()
            best_accuracy = precision