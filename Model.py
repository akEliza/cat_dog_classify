from torch import nn
import torch

"""
input 大小 --image [128, 3, 100, 100]
output大小 --net(input) [128, 2]
label大小 --[128]
"""
class Net(nn.Module):

    def __init__(self, in_dim, n_hidden_1, n_hidden_2,n_hidden_3,n_hidden_4,n_hidden_5,out_dim):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, n_hidden_1),
            nn.BatchNorm1d(n_hidden_1),
            nn.ReLU(True))

        self.layer2 = nn.Sequential(
            nn.Linear(n_hidden_1, n_hidden_2),
            nn.BatchNorm1d(n_hidden_2),
            nn.ReLU(True))

        self.layer3 = nn.Sequential(
            nn.Linear(n_hidden_2, n_hidden_3),
            nn.BatchNorm1d(n_hidden_3),
            nn.ReLU(True))

        self.layer4 = nn.Sequential(
            nn.Linear(n_hidden_3, n_hidden_4),
            nn.BatchNorm1d(n_hidden_4))

        self.layer5 = nn.Sequential(
            nn.Linear(n_hidden_4, n_hidden_5),
            nn.BatchNorm1d(n_hidden_5))

        self.layer6 = nn.Sequential(
            nn.Linear(n_hidden_5, out_dim))

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        y1 = self.layer1(x)
        y2 = self.layer2(y1)
        y3 = self.layer3(y2)
        y4 = self.layer4(y3)
        y5 = self.layer5(y4)
        self.y6 = self.layer6(y5)
        output = torch.softmax(self.y6,1)
        return output


if __name__ == '__main__':
    net=Net(100 * 100 * 3,512,256,512,128,64,2)
    x=torch.randn([128,3,100,100])
    output=net(x)
    print(output.shape)


