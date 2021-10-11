import torch
from torch.optim import optimizer
from   model import ConvNet
from   torch.optim import Adam
from   tqdm import tqdm as etqdm
from   torch import nn as nn
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model = ConvNet()
model = model.to(device)
print(model)
optimizer  = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
criterion  = nn.CrossEntropyLoss()
num_epochs = 1




def train_model(train_dataloader,test_dataloader, train_ds, test_ds):
    train_size = len(train_ds)
    test_size  = len(test_ds)
    best_accuracy = 0.0

    for epoch in etqdm(range(num_epochs)):
        model.train()

        train_accuracy = 0
        train_loss     = 0

        for i ,(images,labels) in enumerate(train_dataloader):
            labels = torch.tensor([]).to(device).detach()
            images  = torch.tensor([]).to(device).detach()

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs,labels)
            loss.backward()
            optimizer.step()

            train_loss+= loss.cpu().data*images(0)
            _,prediction = torch.max(outputs.data,1)

            train_accuracy += int(torch.sum(prediction == labels.data))
        train_accuracy = train_accuracy/train_size
        train_loss     = train_loss/train_size
    


    model.eval()
    
    test_accuracy = 0
    for i, (images, labels) in enumerate(test_dataloader):
        labels = torch.tensor([]).to(device).detach()
        images = torch.tensor([]).to(device).detach()
        outputs = model(images)
        _,prediction = torch.max(outputs.data,1)
        test_accuracy+=int(torch.sum(prediction==labels.data))

    test_accuracy = test_accuracy/test_size 
    
    if test_accuracy>best_accuracy:
        torch.save(model.state_dict(),'Best_epoch.model')
        best_accuracy = test_accuracy  
    return print('Epoch: '+str(epoch)+ ' Train Loss:' +str(int(train_loss))+ ' Train Accuracy: '+str(test_accuracy))
