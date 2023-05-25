import cv2
import torch
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
import pandas as pd
import pytorch_lightning as pl
from torchvision.io import read_image
import matplotlib.pyplot as plt
from torchvision.transforms import ToTensor
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, df, img_dir_path, transforms=None):
        """
        You can set your custom dataset to take in more parameters than specified
        here. But, I recommend at least you start with the three I listed here,
        as these are standard

        csv_file (str): file path to the csv file you created /
        df (pandas df): pandas dataframe

        img_dir_path: directory path to your images
        transform: Compose (a PyTorch Class) that strings together several
          transform functions (e.g. data augmentation steps)

        One thing to note -- you technically could implement `transform` within
        the dataset. No one is going to stop you, but you can think of the
        transformations/augmentations you do as a hyperparameter. If you treat
        it as a hyperparameter, you want to be able to experiment with different
        transformations, and therefore, it would make more sense to decide those
        transformations outside the dataset class and pass it to the dataset!
        """
        self.img_labels = df
        self.img_dir = img_dir_path
        self.transforms = transforms
        
    def __len__(self):
        """
        Returns: (int) length of your dataset
        """
        return len(self.img_labels)

    def __getitem__(self, idx):
        """
        Loads and returns your sample (the image and the label) at the
        specified index

        Parameter: idx (int): index of interest

        Returns: image, label
        """

        img_path =  self.img_dir + self.img_labels.iloc[idx, 1]
        
        image = Image.open(img_path).convert('RGB')

        label = self.img_labels.iloc[idx, -1]
        
        imbalance = self.img_labels.iloc[idx, -2]

        if self.transforms:
            
            if imbalance and not label:
                image = transforms(image)
                image = imbalance_transform(image)
                
            else:
                image = transforms(image)
        else:
            image = tranform_test(image)
               
        return image, label


transforms = T.Compose(
    [
        T.Resize((224,224), antialias=None, interpolation=InterpolationMode.BICUBIC),
        T.RandomApply([
            T.GaussianBlur(kernel_size=(5,5), sigma=(0.1, 0.2))
        ], p=0.5),
        T.RandomEqualize(),
        T.ToTensor()
    ]
)

tranform_test = T.Compose(
    [
        T.Resize((224,224), antialias=None, interpolation=InterpolationMode.BICUBIC),
        T.ToTensor()
    ]
)
imbalance_transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    T.RandomErasing(p=0.2)
])