# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:49:55 2022

@author: User
"""

"""# Training"""

_exp_name = "sample"

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset, random_split
from torchvision.datasets import DatasetFolder, VisionDataset

# This is for the progress bar.
from tqdm.auto import tqdm
import random

# Cross validation
#from sklearn.model_selection import KFold
# Model
from torchvision.models import resnet50,densenet121,resnext50_32x4d
# Augmentation
#from Augmentations import AutoAugment
# Test time augmentation
# import ttach as tta

myseed = 15212530  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

"""## **Transforms**
Torchvision provides lots of useful utilities for image preprocessing, data wrapping as well as data augmentation.

Please refer to PyTorch official website for details about different transforms.
"""

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods

train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    #transforms.RandomInvert(p=0.5),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomCrop(350, 350),
    #transforms.RandomRotation(30, resample=Image.BICUBIC, expand=False),
    #AutoAugment(),
    transforms.ColorJitter(brightness=(0.3), contrast=(0.3), saturation=(0.3), hue=(0.1)),
    transforms.RandomAffine(degrees=(-60,60), translate=(0.1, 0.1), scale=(0.9, 0.9)),
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
'''
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    AutoAugment(),
    transforms.Resize((128, 128)),
    # You may add some transforms here.
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])'''

"""## Define L2 norm"""
def l2_norm(loss, model):
    l2_lambda = 0.00001
    l2_norm = torch.tensor(0.).to(device)
    l2_norm = sum(p.pow(2.0).sum()
                  for p in model.parameters())
    return loss + l2_lambda * l2_norm

"""## Split data"""
def train_valid_split(data_set, valid_ratio, seed):
    '''Split provided training data into training set and validation set'''
    valid_set_size = int(valid_ratio * len(data_set)) 
    train_set_size = len(data_set) - valid_set_size
    train_set, valid_set = random_split(data_set, [train_set_size, valid_set_size], generator=torch.Generator().manual_seed(seed))
    return np.array(train_set), np.array(valid_set)

"""## Define mixup"""
_dataset_dir = "./food11"
path = os.path.join(_dataset_dir,"training")
def mixup(file):
    #files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
    files = file
    fname = []
    label = ['0_','1_','2_','3_','4_','5_','6_','7_','8_','9_','10']
    for n in label:
        mix = [f for f in files if f[18:20] == n]
        c=len(mix)
        
        for i in range(12000):
            if i < c:
                fname1 = [mix[i], mix[i]]
                fname.append(fname1)
            else:
                fname2 = [mix[random.randrange(len(mix))], mix[random.randrange(len(mix))]]
                fname.append(fname2)
    return fname
        
"""## **Datasets**
The data is labelled by the name, so we load images and label while calling '__getitem__'
"""

class trainFoodDataset(Dataset):

    def __init__(self,path,file,tfm=test_tfm,files = None):
        super(trainFoodDataset).__init__()
        #self.path = path
        #self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        self.files = mixup(file)
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        
        fname1, fname2 = self.files[idx]
        if fname1==fname2:
            im = Image.open(fname1)
        else:
            im1 = Image.open(fname1)
            im1 = im1.resize((512,512))
            im2 = Image.open(fname2)
            im2 = im2.resize((512,512))
            im = Image.blend(im1, im2, 0.5)
        #im = Image.open(fname)
        im = self.transform(im)
        #im = self.data[idx]
        try:
            label = int(fname1.split("\\")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return im,label

class FoodDataset(Dataset):

    def __init__(self,path,file,tfm=test_tfm,files = None):
        super(FoodDataset).__init__()
        #self.path = path
        #self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        self.files = file
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        #im = self.data[idx]
        try:
            label = int(fname.split("\\")[-1].split("_")[0])
        except:
            label = -1 # test has no label
        return im,label


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)

batch_size = 128
_dataset_dir = "./food11"

#path = os.path.join(_dataset_dir,"dataset")
#dataset = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
train_path = os.path.join(_dataset_dir,"training")
valid_path = os.path.join(_dataset_dir,"validation")
train_dataset = sorted([os.path.join(train_path,x) for x in os.listdir(train_path) if x.endswith(".jpg")])
valid_dataset = sorted([os.path.join(valid_path,x) for x in os.listdir(valid_path) if x.endswith(".jpg")])
dataset = ConcatDataset([train_dataset, valid_dataset])
train_data, valid_data = train_valid_split(dataset, 0.2, myseed)

# Construct datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = trainFoodDataset(train_path, train_data, tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size*4, shuffle=True, num_workers=0, pin_memory=True)
valid_set = FoodDataset(valid_path, valid_data, tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

# The number of training epochs and patience.
n_epochs = 200
patience = 20 # If no improvement in 'patience' epochs, early stop

# Initialize a model, and put it on the device specified.
#model = Classifier().to(device)
#model = densenet121(pretrained=False).to(device)
model = resnext50_32x4d(pretrained=False, num_classes=11).to(device)
#model.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.00008, weight_decay=1e-5) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0

for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()
        #print(imgs.shape,labels.shape)
        
        #datas, targets_a, targets_b, lam = mixup_data(imgs, labels, 100, True)
        

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))
        #logits = model(datas.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))
        #loss = mixup_criterion(criterion, logits, targets_a.to(device), targets_b.to(device), lam)
        #loss = l2_norm(loss, model)
        
        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()
        scheduler.step(acc)

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)
        
        
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        #break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


    # update logs
    if valid_acc > best_acc:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break

"""# My TTA"""
class TTADataset(Dataset):

    def __init__(self,path,train_tfm=train_tfm,test_tfm=test_tfm,files = None):
        super(TTADataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
        print(f"One {path} sample",self.files[0])
        self.train_transform = train_tfm
        self.test_transform = test_tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        aug_im = torch.empty(6, 3, 128, 128)
        aug_im[0,:,:,:] = self.test_transform(im)
        for i in range(5):
            aug_im[i+1,:,:,:] = self.train_transform(im)
            
        return aug_im

aug_test_set = TTADataset(os.path.join(_dataset_dir,"test"), train_tfm=train_tfm, test_tfm=test_tfm)
aug_test_loader = DataLoader(aug_test_set, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

"""# Testing and generate prediction CSV"""

#model_best = Classifier().to(device)
model_best = resnext50_32x4d(pretrained=False, num_classes=11).to(device)
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
model_best.eval()
prediction = []
with torch.no_grad():
    for data in aug_test_loader:
        test_pred = model_best(data[0,:,:,:,:].to(device)) #[batch_size, probability of 11 labels]
        t2n = test_pred.cpu().data.numpy()
        tta_test_pred = 0.5*t2n[0,:]+0.5*np.mean(t2n[1:len(t2n),:], axis=0)
        test_label = np.argmax(tta_test_pred)
        prediction.append(test_label)


'''
"""# Other TTA"""
test_set = FoodDataset(os.path.join(_dataset_dir,"test"), tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

"""# Testing and generate prediction CSV"""
#model_best = Classifier().to(device)
model_best = densenet121(pretrained=False).to(device)
model_best.load_state_dict(torch.load(f"{_exp_name}_best.ckpt"))
tta_model = tta.SegmentationTTAWrapper(model_best, tta.aliases.d4_transform(), merge_mode='mean')
model_best.eval()
prediction = []
with torch.no_grad():
    for data,_ in test_loader:
        test_pred = tta_model(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()'''
        
        
#create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(1,len(aug_test_set)+1)]
df["Category"] = prediction
df.to_csv("submission.csv",index = False)
        


"""# Q1. Augmentation Implementation
## Implement augmentation by finishing train_tfm in the code with image size of your choice. 
## Directly copy the following block and paste it on GradeScope after you finish the code
### Your train_tfm must be capable of producing 5+ different results when given an identical image multiple times.
### Your  train_tfm in the report can be different from train_tfm in your training code.

"""

train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.Resize((128, 128)),
    # You need to add some transforms here.
    transforms.ToTensor(),
])

"""# Q2. Residual Implementation
![](https://i.imgur.com/GYsq1Ap.png)
## Directly copy the following block and paste it on GradeScope after you finish the code

"""

from torch import nn
class Residual_Network(nn.Module):
    def __init__(self):
        super(Residual_Network, self).__init__()
        
        self.cnn_layer1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )

        self.cnn_layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
        )

        self.cnn_layer3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.BatchNorm2d(128),
        )

        self.cnn_layer4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
        )
        self.cnn_layer5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.BatchNorm2d(256),
        )
        self.cnn_layer6 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(256* 32* 32, 256),
            nn.ReLU(),
            nn.Linear(256, 11)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        # input (x): [batch_size, 3, 128, 128]
        # output: [batch_size, 11]

        # Extract features by convolutional layers.
        x1 = self.cnn_layer1(x)
        
        x1 = self.relu(x1)
        
        x2 = self.cnn_layer2(x1)
        
        x2 = self.relu(x2)
        
        x3 = self.cnn_layer3(x2)
        
        x3 = self.relu(x3)
        
        x4 = self.cnn_layer4(x3)
        
        x4 = self.relu(x4)
        
        x5 = self.cnn_layer5(x4)
        
        x5 = self.relu(x5)
        
        x6 = self.cnn_layer6(x5)
        
        x6 = self.relu(x6)
        
        # The extracted feature map must be flatten before going to fully-connected layers.
        xout = x6.flatten(1)

        # The features are transformed by fully-connected layers to obtain the final logits.
        xout = self.fc_layer(xout)
        return xout