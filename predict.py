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

parser.add_argument('--img', type=str, default='flowers/test/100/image_07896.jpg')
parser.add_argument('--topk', type=int, default=5)
parser.add_argument('--json', type=str, default='cat_to_name.json')
parser.add_argument('--gpu',  type=str, default='gpu')
parser.add_argument('--check_point', type=str, default='check_point.pth')

args = parser.parse_args()

with open(args.json, 'r') as f:
    cat_to_name = json.load(f)
    
def load(path):
    check_point = torch.load(path)

    if check_point['arch'] == 'vgg16':
        model = models.vgg16(pretrained=True)
               
    elif check_point['arch'] == 'densenet161':
        model = models.densenet(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False
        
    model.classifier = check_point['classifier']
    model.class_to_idx = check_point['class_to_idx']
    model.load_state_dict(check_point['state_dict'])
    
    return model

model = load(args.check_point)


def process_image(img):
   
    # TODO: Process a PIL image for use in a PyTorch model
    pil_img = Image.open(img)
    
    preprocess = transforms.Compose([transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], 
                                                          [0.229, 0.224, 0.225])])
    img_tensor = preprocess(pil_img)
    return img_tensor


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    dvc = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
   
    
    model.to(dvc)
    
    model.eval()
    img = process_image(image_path).numpy()
    img = torch.tensor(img)
    img = img.float()
    
    # Removing RunTimeError for missing batch size - add batch size of 1 
    img = img.unsqueeze(0)
    img = img.to('cuda')

    
    with torch.no_grad():
        logps = model.forward(img.cuda())
        ps = torch.exp(logps)
        probs, indices = ps.topk(topk)
        probs = [float(p) for p in probs[0]] # Define probabilities and convert them to float from tensor
        
        # Convert from indices to class labels
        mapping_dict = {val:key for key, val in model.class_to_idx.items()} # Get a mapping using dictionary 
        classes = [mapping_dict[int(i)] for i in indices[0]]
        names = [cat_to_name[i] for i in classes]
            
    return probs, names

probs, names = predict(args.img, model, args.topk)

print('Flower types: {}'.format(names))
print('Probabilities: {}'.format(probs))

    


