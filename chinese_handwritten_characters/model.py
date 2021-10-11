from   torch import nn as nn




class ConvNet(nn.Module):
    def __init__(self,num_classes=15):
        super(ConvNet, self).__init__()

        # Output size after convulation filter = (w-f+2p)/s + 1
        
        # Input shape = (64,1,64,64)
        self.conv1    = nn.Conv2d(in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1)
        # New shape   = (64,12,64,64)
        self.bn1      = nn.BatchNorm2d(12)
        self.relu1    = nn.ReLU()
        self.pool     = nn.MaxPool2d(kernel_size=2) # Reduce the shape by factor 2
        
        # New shape   = (64,12,32,32)
        self.conv2    = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
        # New shape   = (64,24,32,32)
        self.bn2      = nn.BatchNorm2d(24)
        self.relu2    = nn.ReLU()
        
        # New shape   = (64,24,32,32)
        self.conv3    = nn.Conv2d(in_channels=24, out_channels=30, kernel_size=3, stride=1, padding=1)
        # New shape   = (64,30,32,32)
        self.bn3      = nn.BatchNorm2d(30)
        self.relu3    = nn.ReLU()

        self.fc       = nn.Linear(in_features=30*32*32, out_features=16)

    def forward(self,input):
        input  = input.unsqueeze(1)
        output = self.conv1(input)
        output = self.bn1(output)
        output = self.relu1(output)
        output = self.pool(output)
        output = self.conv2(output)
        output = self.bn2(output)
        output = self.relu2(output)
        output = self.conv3(output)
        output = self.bn3(output)
        output = self.relu3(output)

        output = output.view(-1, 30*32*32)
        output = self.fc(output)
        return output
