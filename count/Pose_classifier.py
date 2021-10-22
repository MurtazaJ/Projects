# All imports 
import numpy as np  # numpy
import matplotlib.pyplot as plt  # plots for visualization
import pathlib
import glob
import os 
import PoseModule as pm

import cv2   # computer vision or OpenCv
import torch # PyTorch DeepLearning
from torchvision import transforms, datasets # laod data and transform to tensor
from torch.utils.data import DataLoader # for batch_size data

import torch.nn as nn  # Model
from torch import optim  # optimizer
from torchsummary import summary 

from torch.autograd import Variable
import torch.nn.functional as F  
from PIL import Image 


# path 
# data_path = './data_3/'  # repalce your path 

# categories
#classes = sorted([j.name.split('/')[-1] for j in pathlib.Path(data_path+'Train').iterdir()])
# print(classes)
classes = ['pushup', 'situp']
# Convolutional Neural Network (CNN) Module
class ConvNet(nn.Module):
  def __init__(self,num_classes=2):
    super(ConvNet, self).__init__()
    
    # First convolutional Layer

    # Output size after convolutional filter
    #(w-f+2p)/s +1 = 224-3+2*1 =224
    # input shape = 32,3,224,224 'batchsize,rgb colour, height , width of an image 

    self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1)
    # New Shape =(32,12,224,224)
    # Batch_Normalization no of features = no of out_channels
    self.bn1 =nn.BatchNorm2d(num_features=16)
    self.relu1 = nn.ReLU()
    #reduce the dimension of conv output by a factor of kernel size
    self.pool = nn.MaxPool2d(kernel_size=2)
    #new shape = (32,12,112,112)
    self.dropout = nn.Dropout(0.25)
    # second convoluitonal layer 
    self.conv2 = nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1)
    # shape = (32,20,112,112)
    self.bn2 =nn.BatchNorm2d(num_features=32)
    self.relu2 = nn.ReLU()

    # Third convoluitonal layer 
    self.conv3 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,stride=1,padding=1)
    # shape = (32,32,112,112)
    self.bn3 =nn.BatchNorm2d(num_features=64)
    self.relu3 = nn.ReLU()
    
    # fully connected layer
    self.fc = nn.Linear(in_features=112 * 112 * 64,out_features=num_classes)
    
    #Forward Function
  def forward(self,input):
    output=self.conv1(input)
    output =self.bn1(output)
    output=self.relu1(output)
    output = self.dropout(output)
    output=self.pool(output)

    output=self.conv2(output)
    output =self.bn2(output)
    output=self.relu2(output)
    output = self.dropout(output)

    output=self.conv3(output)
    output =self.bn3(output)
    output=self.relu3(output)
    output = self.dropout(output)
      #Above output will be im matrix or tensor form with shape(batch_size,32,112,112)
      # reshape to feed in fc
    output=output.view(-1,64*112*112)  # flattern the data
    output =self.fc(output)

    return output

# Load the Model
checkpoint=torch.load('Best_Pose_Model_7.pth')  # replace your path with the (.pth) file
model = ConvNet(num_classes=2)
model.load_state_dict(checkpoint)
model.eval()

#transformer to tensor data
transformer = transforms.Compose([ transforms.Resize((256,256)),
                                      transforms.RandomCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5,0.5,0.5],
                                                            [0.5,0.5,0.5])
                                       ])

# have a pil format image and use this function to predict
def prediction(img_path, transformer):
  # image = Image.open(img_path)
  image_tensor = transformer(img_path).float()
  image_tensor =image_tensor.unsqueeze(0)
  input = Variable(image_tensor)

  output = model.forward(input)
  index =output.data.numpy().argmax()
  pred =classes[index]
  _, prediction = torch.max(output.data,1)
  return pred                                      


'''
pred_dict ={}
src='C:/Users/User/Documents/Build_Week_3/images_pred/'  # change the path folder here 
for i in os.listdir(src):
  img =cv2.imread(src+i) 
  img = pm.ManualFindPose(img)
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  img_new =Image.fromarray(img)
  pred_dict[i[i.rfind('/')+1:]] = prediction(img_new,transformer)

print(pred_dict)

# if we want to save it in a csv file 
import csv
with open('test.csv', 'w') as f:
    for key in pred_dict.keys():
        f.write("%s,%s\n"%(key,pred_dict[key]))

# src='C:/Users/User/Documents/Build_Week_3/images_pred/'  # change the path folder here 
'''
# img =cv2.imread(src+'download_13.jpg') 
# img = pm.ManualFindPose(img)
# img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
# img_new =Image.fromarray(img)
# result = prediction(img_new,transformer)
# print(result)