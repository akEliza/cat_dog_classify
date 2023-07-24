import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset_sampling import MyDataset
from Model import Net
from ResNet18 import BasicBlock, ResNet18
from matplotlib import pyplot as plt
import os
import numpy as np
import random

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
    
if __name__ == '__main__':

    # 设置随机数种子
    setup_seed(20)
    data_path = r"E:\cat_dog_classify"
    sample_train = MyDataset(data_path,'train')
    # sample_test = MyDataset(data_path,False)
    sample_valid = MyDataset(data_path, 'valid')
    # net = Net(100 * 100 * 3,512,256,512,128,64,2)
    net = ResNet18(BasicBlock)

    if os.path.exists("./params.pth"):
        net.load_state_dict(torch.load("./params.pth"))
        print("Load Success!")
    else:
        print("No Params!")

    batch_size = 100
    num_epoches = 30

    train_loader = DataLoader(sample_train, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(sample_test, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(sample_valid, batch_size=batch_size, shuffle=True)


    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    net = net.to(device)
    # 定义损失函数和优化器
    loss_fn = nn.CrossEntropyLoss()
    # nn.MSELoss()
    optimizer = optim.Adam(net.parameters(),lr=1e-3)

    Train_Loss = []
    Train_Acc = []
    Valid_Loss = []
    Valid_Acc = []
    net.train()
    best_acc = 0
    for epoch in range(num_epoches):
        train_loss = 0
        train_acc = 0
        for img,label in train_loader:
            img = img.to(device)
            label = label.to(device)
            output = net(img)
            loss = loss_fn(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #每批次的损失累加
            train_loss += loss.item()*label.size(0)
            argmax = torch.argmax(output, 1)
            # 每批次的精度累加
            num_acc = (argmax == label).sum()
            train_acc += num_acc.item()
        train_loss /= len(sample_train)
        train_acc /= len(sample_train)
        print('epoch: {},Train Loss: {:.3f}, Train Acc: {:.3f}'.format(epoch,
        train_loss,train_acc))
        Train_Acc.append(train_acc)
        Train_Loss.append(train_loss)

        net.eval()
        valid_correct, valid_loss=(0,0)
        for batch in valid_loader:
            img, label = batch
            img = img.to(device)
            label = label.to(device)
            with torch.no_grad():
                output = net(img)
            loss = loss_fn(output, label)
            valid_loss +=loss.item()*label.size(0)
            argmax = torch.argmax(output, 1)
            valid_correct += (argmax == label).sum().item()
            del img, label
            torch.cuda.empty_cache()
        valid_acc = valid_correct / len(sample_valid)
        valid_loss = valid_loss / len(sample_valid)
        print('epoch: {},Valid Loss: {:.3f}, Valid Acc: {:.3f}'.format(epoch, valid_loss, valid_acc))
        Valid_Acc.append(valid_acc)
        Valid_Loss.append(valid_loss)

        if (valid_acc > best_acc):
            best_acc = valid_acc
            torch.save(net.state_dict(), "./params.pth")
            print('now the best acc is{}'.format(best_acc))
    
    epochs = np.arange(30)
    plt.plot(epochs, Train_Acc, 'r', label='Training Accuracy')
    plt.plot(epochs, Valid_Acc, 'b', label='Validation Accuracy')
    plt.title('Accuracy-Epoch')
    plt.legend()
    # plt.show()
    plt.savefig('./accuracy.png')
    plt.close()
    
    
    plt.plot(epochs, Train_Loss, 'r', label='Training Loss')
    plt.plot(epochs, Valid_Loss, 'b', label='Validation Loss')
    plt.title('Loss-Epoch')
    plt.legend()
    # plt.show()
    plt.savefig('./loss.png')
    plt.close()
