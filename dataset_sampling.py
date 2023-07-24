from torch.utils.data import Dataset,DataLoader
from torchvision import transforms
from PIL import Image
import os
import torch

class MyDataset(Dataset):
    """mode in ['train', 'test', 'valid']"""
    def __init__(self, main_dir,mode='train'):
        self.dataset = []
        data_filename = os.path.join(main_dir, "cat_dog_datasets", mode) 
        for img_data in os.listdir(os.path.join(main_dir,data_filename)):
            img_path = os.path.join(main_dir,data_filename,img_data)
            label = img_data.split(".")[0]
            self.dataset.append([img_path,label])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        data = self.dataset[index]
        image_data = self.image_preprocess(Image.open(data[0]))
        label_data = torch.tensor(int(data[1]))
        return image_data, label_data

    def image_preprocess(self,x):
    #    return transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.4914, 0.4822, 0.4465],[0.2023, 0.1994, 0.2010])])(x)
       return transforms.Compose([
           transforms.ToTensor(),
           transforms.Normalize(mean = [0.4914, 0.4822, 0.4465],std = [0.2023, 0.1994, 0.2010])])(x)

if __name__ == '__main__':
    data_path = r"E:\cat_dog_classify"
    dataset = MyDataset(data_path,'train')
    dataloader = DataLoader(dataset,128,shuffle=True,num_workers=0,drop_last=True)
    print(len(dataset))
    for data in dataloader:
        print(data[0].shape)
        print(data[1].shape)
        print(data[0].dtype)
        print(data[1].dtype)
        exit()
        # print(data[1])
