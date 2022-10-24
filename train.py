import torch
from torch import optim, nn
from torch import nn, optim
from collections import OrderedDict
import argparse
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json

parser = argparse.ArgumentParser() 

# Adding arguments
parser.add_argument('--data_dir', type=str, default="./flowers/")
parser.add_argument('--arch', type=str, default='vgg16')
parser.add_argument('--hidden_units', type=int, default=2208)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--epochs', type=int, default=6)
parser.add_argument('--save_dir', type=str, default='./check_point.pth')
parser.add_argument('--gpu', type=str, default='gpu')

args = parser.parse_args()

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

val_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_data = datasets.ImageFolder(valid_dir, transform=val_transforms)


trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_data, batch_size=64)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
def define_model(arch,hidden_units):
    if args.arch == "vgg16":
        model = models.vgg16(pretrained=True)
    else:
        model = models.densenet161(pretrained=True)
        
    for param in model.parameters():
        param.requires_grad = False 
        
    if args.arch =='vgg16':
        classifier = nn.Sequential(OrderedDict([
                                ('dropout1', nn.Dropout(0.1)),
                                ('fc1', nn.Linear(25088,args.hidden_units)),
                                ('relu1', nn.ReLU()),
                                ('dropout2', nn.Linear(args.hidden_units, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
    else:
     classifier = nn.Sequential(OrderedDict([
                                ('dropout1', nn.Dropout(0.1)),
                                ('fc1', nn.Linear(2208,args.hidden_units)),
                                ('relu1', nn.ReLU()),
                                ('dropout2', nn.Linear(args.hidden_units, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
    model.classifier = classifier
    print(f"Model built from {arch} and {hidden_units} hidden units.")
    return model       
    
model = define_model(args.arch, args.hidden_units)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), args.lr)

if args.gpu == 'gpu':
    device = 'cuda'
else:
    device = 'cpu'
    
model.to(device)
epochs = 6

def train(model, trainloader, validloader, criterion, optimizer, epochs, device):
    steps = 0
    running_loss = 0
    print_every = 30
    for epoch in range(epochs):
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
            if steps % print_every == 0:
                val_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for img, labels in validloader:
                        img, labels = img.to(device), labels.to(device)
                        logps = model.forward(img)
                        val_loss = criterion(logps, labels) # Calculating validation loss

                                    
                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print("Epoch {}/{}  ".format(epoch+1, epochs), 
                "Training loss: {:.3f}".format(running_loss/print_every),
                "Validation loss: {:.3f}".format(val_loss/len(validloader)),
                "Validation accuracy: {:.3f}".format(accuracy/len(validloader)))
                running_loss = 0
                model.train()
        
    

train(model, trainloader, validloader, criterion, optimizer, epochs, device)

def testing(model, testloader, device): 
    test_loss = 0
    accuracy = 0
    model.eval()
    with torch.no_grad():
        for img, labels in testloader:
            img, labels = img.to(device), labels.to(device)
            logps = model.forward(img)
            batch_loss = criterion(logps, labels)
                    
            test_loss += batch_loss.item()
                    
            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
    print("Test loss:{:.3f}...".format(test_loss/len(testloader)),"Test accuracy:{:.3f}...".format(accuracy/len(testloader)))
testing(model, testloader, device)

model.to('cpu')
model.class_to_idx = train_data.class_to_idx
checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'class_to_idx': model.class_to_idx,
              'classifier': model.classifier,
              'arch': args.arch,
              'optimizer_state_dict': optimizer.state_dict
              }

torch.save(checkpoint, args.save_dir)
    
    




    
 


