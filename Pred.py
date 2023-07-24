import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset_sampling import MyDataset
from Model import Net
from ResNet18 import ResNet18, BasicBlock
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
    data_path = r"E:\cat_dog_classify"
    setup_seed(20)
    sample_test = MyDataset(data_path, 'test')
    net = ResNet18(BasicBlock)
    net.load_state_dict(torch.load("./params.pth"))

    batch_size = 100
    test_loader = DataLoader(sample_test, batch_size=batch_size, shuffle=True)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    net = net.to(device)
    test_correct=0
    for batch in test_loader:
        img, label = batch
        img = img.to(device)
        label = label.to(device)
        with torch.no_grad():
            output = net(img)
        argmax = torch.argmax(output, 1)
        test_correct += (argmax == label).sum().item()
        del img, label
        torch.cuda.empty_cache()
    test_acc = test_correct / len(sample_test)
    print('Final Test Acc: {:.3f}'.format(test_acc))

