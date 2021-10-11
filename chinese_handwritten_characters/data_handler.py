# Importing the required libraries
import pandas as pd
import numpy as np
import torch
from   torch import nn as nn
from   torchvision import transforms
from   torch.utils.data import Dataset
import os
from   PIL import Image
import matplotlib.pyplot as plt
from   sklearn.model_selection import train_test_split
from   tqdm import tqdm as etqdm
from   model import ConvNet
from   torch.optim import Adam
from   sklearn.metrics import confusion_matrix

# print(os.getcwd())
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print(device)


#Reading the csv and performing EDA on the data
df = pd.read_csv('chinese_mnist.csv')
# print(df.head())
# print(df.value.nunique())
# print(df.value.unique())
# print(df.value.value_counts())
# print(df.value.value_counts().sum())
# print(df.character.unique())
df_train, df_test1 = train_test_split(df, train_size=0.7, random_state=0)
df_test, df_val    = train_test_split(df_test1, train_size=0.5, random_state=0)
# Converting the CSV into 3 splits of Train, Test and Val
df_train.to_csv('df_train.csv')
df_test.to_csv('df_test.csv')
df_val.to_csv('df_val.csv')




# Checking how to load an image
# img_path = df.iloc[0,:]
# image    = Image.open("data/input_{suite}_{id}_{code}.jpg".format(suite =img_path['suite_id'], id=img_path['sample_id'],code=img_path['code']))
# labels   = img_path['character']
# img      = np.array(image)
# print(image.getdata)
# print(labels)
# print(type(img))
# print(img.shape)
# plt.imshow(img, cmap='gray')



# Instantiating a class
class Dataset(Dataset):
    def __init__(self, csv_file, transform = None):
        self.annotations = pd.read_csv(csv_file)
        self.transform   = transform
        self.y           = df['value']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_locate = self.annotations.iloc[index,:]
        image      = Image.open("data/input_{suite}_{id}_{code}.jpg".format(suite =img_locate['suite_id'], id=img_locate['sample_id'],code=img_locate['code']))
        image      = np.array(image)
        # image      = np.expand_dims(1,image)
        label      = img_locate['value']
        if self.transform:
            image = self.transform(image)
        return image, label
    
# Loading the Training data
train_ds = Dataset('df_train.csv') 
train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)

#Loading the test data
test_ds  = Dataset('df_test.csv') 
test_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64)

# Loading the validation data
val_ds   = Dataset('df_val.csv')
val_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=64)

images ,labels = next(iter(train_dataloader))
plt.imshow(images[0], cmap='gray')
print(images.shape)
print(labels[0])
                                                       
#------------------------------------------------------------------------------

model = ConvNet()
model = model.to(device)
print(model)
optimizer  = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion  = nn.CrossEntropyLoss()
num_epochs = 6
train_size = len(train_ds)
test_size  = len(test_ds)
print(train_size, test_size)

for epoch in etqdm (range(num_epochs)):

    labels = torch.tensor([]).to(device).detach()
    preds  = torch.tensor([]).to(device).detach()
    
    total_preds = 0
    correct_preds = 0
    
    train_running_loss = 0.0
    
    for index, data in enumerate (train_dataloader):
        model.train()
        
        batch_inputs, batch_labels = data[0][:].to(device).type(torch.float), data[1][:].to(device)
        
        outputs = model(batch_inputs)
        
        loss = criterion(outputs, batch_labels) # expects distribution from model softmax as pred and target_index as target
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_running_loss += loss.mean()
        
        labels = torch.cat((labels, batch_labels))
        #total_preds += 1
        
        for index, item in enumerate(outputs):
            #if labels[index] == torch.argmax(item):
            #    correct_preds += 1
                
            preds  = torch.cat((preds, torch.argmax(item).unsqueeze(-1)))
        
        if index+1 == int(train_size / 64):
            print(f'Training Epoch: {epoch+1}, step: {index+1}, mean training loss: {train_running_loss / int(train_size / batch_size)}')
            train_running_loss = 0.0
    
    print('Calculating conf_matrix')
    conf_mat = confusion_matrix(labels.cpu().numpy(), preds.cpu().numpy())
    
    total = np.sum(conf_mat)
    
    correct_count = 0
    
    for i, data in enumerate(conf_mat[0]):
        correct_count += conf_mat[i][i]
    
    
    print(f'Training Epoch {epoch+1}:\n Accuracy: {correct_count/total}\n{conf_mat}\n')
    print()
    
    labels = torch.tensor([]).to(device).detach()
    preds  = torch.tensor([]).to(device).detach()
    
    total_preds = 0
    correct_preds = 0
    
    valid_running_loss = 0.0


