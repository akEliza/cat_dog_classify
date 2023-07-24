from torch import nn
import torch

class Net(nn.Module):

    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3,out_channels=64,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )#batch*64*112*112

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64,out_channels=128,kernel_size=3,stride=1,padding=1,groups=8),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )#batch*128*56*56

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=256,kernel_size=3,stride=1,padding=1,groups=8),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )#batch*256*28*28

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=512,kernel_size=3,stride=1,padding=1,groups=8),
            nn.BatchNorm2d(num_features=512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )#batch*512*14*14

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=512,out_channels=256,kernel_size=3,stride=1,padding=1,groups=8),
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2,stride=2)
        )#batch*256*7*7

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=256,out_channels=128,kernel_size=3,stride=1,groups=8),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(True)
        )#batch*128*5*5

        self.conv7 = nn.Sequential(
            nn.Conv2d(in_channels=128,out_channels=64,kernel_size=3,stride=1,groups=8),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(True)
        )#batch*64*3*3

        self.conv8 = nn.Conv2d(in_channels=64,out_channels=5,kernel_size=3,stride=1)
        #batch*5*1*1

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(y1)
        y3 = self.conv3(y2)
        y4 = self.conv4(y3)
        y5 = self.conv5(y4)
        y6 = self.conv6(y5)
        y7 = self.conv7(y6)
        y8 = self.conv8(y7)
        output = y8.reshape(y8.size(0),-1)
        output1 = torch.tanh(output[:,:1])
        output2 = output[:,1:]

        return output1,output2

if __name__ == '__main__':
    data = torch.randn(10,3,224,224)
    net = Net()
    out = net(data)
    # print(out[0],out[1])
    print(out[0].shape,out[1].shape)