

# #TODO: Import dependencies for Debugging andd Profiling
#TODO: Import your dependencies.
#For instance, below are some dependencies you might need if you are using Pytorch'

import torch
from torchvision.transforms import Resize , Normalize
import matplotlib.pyplot as plt
import os 
from torchvision.io import read_image
import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
# from torchsummary  import summary

import torch.nn.functional as F 
import argparse

S3_BUCKET = ""


PATH_BINDER = "/"

TRAIN_PATH = "dogImages" +PATH_BINDER +"dogImages" +PATH_BINDER +"train"
TEST_PATH = "dogImages" +PATH_BINDER +"dogImages" +PATH_BINDER +"test"


class MyDogCustomImageDataset(Dataset):
    def __init__(self, meta_dict, img_dir,transform=None, target_transform=None, shape = (256, 256)):
        # print(img_dir)
        self.meta_data = meta_dict
        self.img_dir = img_dir
        self.transform = transform
        self.shape = shape
        self.target_transform = target_transform

    def __len__(self):
        return len(self.meta_data["idx"])

    def __getitem__(self, idx):
        img_path = self.img_dir  + self.meta_data["image_paths"][idx]
        # print(img_path)
        # print(self.img_dir)
        image = read_image(img_path).type('torch.DoubleTensor')/255
        # torch.reshape(image, self.shape)
        label = self.meta_data["class_idx"][idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

def test(model, test_loader, criterion):
    '''
    TODO: Complete this function that can take a model and a 
          testing data loader and will get the test accuray/loss of the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss, correct, len(test_loader.dataset), 100.0 * correct / len(test_loader.dataset)
        )
    )


def train(model, train_loader, criterion, optimizer, epoch):
    '''
    TODO: Complete this function that can take a model and
          data loaders for training and will get train the model
          Remember to include any debugging/profiling hooks that you might need
    '''
    size = len(train_loader)
    running_loss = 0.0
    running_corrects = 0
    print("Epoch; {}, Number of steps: {}".format(epoch, size))
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        
        # target = F.one_hot(target, 134)

        output = model(data.float())
        _, preds = torch.max(output, 1)
 

        # print("Outoput: {}".format(output))
        # print("Outoput: {}".format(output.shape))

        # print("Predictions: {}", preds)
        # print("targer: {}".format(target))
        # print("targertype: {}".format(target.shape))
        loss = criterion(output, target.long() -1)
        loss.backward()
        optimizer.step()
        loss, current = loss.item(), (batch_idx + 1) * data.size(0)
        running_loss +=loss * data.size(0)
        running_corrects += torch.sum(preds == target.data)
        
        print(
            f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}] batch: {batch_idx:>d}")
        if batch_idx == 10:
            break
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = running_corrects.double() / len(train_loader)
    print(f' Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')


def net():
    '''
    TODO: Complete this function that initializes your model
          Remember to use a pretrained model
    '''
    pretrained_model = models.resnet18(pretrained=True)
    for layer in pretrained_model.parameters():
        layer.requires_grad = False
    model = nn.Sequential(
        pretrained_model,
        nn.Linear(1000,700),
        nn.ReLU(),
        nn.Linear(700,500),
        nn.ReLU(),
        nn.Linear(500, 133),
        nn.LogSoftmax(dim =1)

    )
    print("The Model being Trained resnet18")
    print(model)
    return model

# def create_data_loaders(data, batch_size):
#     '''
#     This is an optional function that you may or may not need to implement
#     depending on whether you need to use data loaders or not
#     '''
#     pass

def main(args):
    
    parser = argparse.ArgumentParser(description="Pytorch Project Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="input batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=14,
        metavar="N",
        help="number of epochs to train (default: 14)",
    )
    parser.add_argument(
        "--lr", type=float, default=1.0, metavar="LR", help="learning rate (default: 1.0)"
    )
    args = parser.parse_args()
    

    train_class_folder_names = os.listdir(TRAIN_PATH)
    test_class_folder_names = os.listdir(TEST_PATH)


    train_class_map = {str(int(folder.split(".")[0])):folder.split(".")[1] for folder in train_class_folder_names}
    test_class_map = {str(int(folder.split(".")[0])):folder.split(".")[1] for folder in test_class_folder_names}
    def create_meta_dict(path, path_delimiter, folder_names):
        meta_data_dict = {
            "image_name":[],
            "image_paths":[],  
            "class_idx":[],  
            "class_name":[],
            "idx": []  
        }  
        idx = 0

        for folder in folder_names:
            tmp =  os.listdir(path + path_delimiter + folder)
            meta_data_dict["image_name"] += tmp
            meta_data_dict["image_paths"] += [ path_delimiter+ folder + path_delimiter + name for name in  tmp]
            meta_data_dict["class_idx"] += [int(folder.split(".")[0])] * len(tmp)
            meta_data_dict["class_name"] += [folder.split(".")[1]] * len(tmp)
            meta_data_dict["idx"] += [i for i in range(idx, idx + len(tmp), 1)]
            idx += len(tmp) 
        return meta_data_dict
    

    meta_data_dict_train = create_meta_dict(TRAIN_PATH, PATH_BINDER, train_class_folder_names)
    meta_data_dict_test = create_meta_dict(TEST_PATH, PATH_BINDER, test_class_folder_names)


    image_transformer = transforms.Compose([
    Resize((256, 256)), Normalize(
       mean=[0.485, 0.456, 0.406],
       std=[0.229, 0.224, 0.225])])
    train_dataset= MyDogCustomImageDataset(meta_data_dict_train, TRAIN_PATH, image_transformer)
    test_dataset = MyDogCustomImageDataset(meta_data_dict_test, TEST_PATH, image_transformer)
    


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=True, num_workers=0)

    
    '''
    TODO: Initialize a model by calling the net function
    '''
    model=net()
    model =  model.float()
    epochs = 1
    '''
    TODO: Create your loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.001)
    
    '''
    TODO: Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    for epoch in range(epochs):
        model=train(model, train_dataloader, loss_criterion, optimizer, epoch)
    
    '''
    TODO: Test the model to see its accuracy
    '''
    test(model, test_dataloader, loss_criterion)
    
    '''
    TODO: Save the trained model
    '''
    torch.save(model, "models\model.pth")

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    '''
    TODO: Specify any training args that you might need
    '''
    
    args=parser.parse_args()
    
    main(args)
