import os
import cv2
import numpy as np
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import torch
from tqdm import tqdm
from PIL import Image

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, enable_preload=False):
        self.label_file = pd.read_csv(csv_file, dtype={col: int for col in ["N", "D", "G", "C", "A", "H", "M", "O"]})
        self.img_dir = img_dir
        self.transform = transform
        self.enable_preload = enable_preload
        
        # 预加载所有标签数据
        print("Loading labels...")
        self.labels = torch.tensor(self.label_file.iloc[:, 2:].astype(float).values, dtype=torch.float32)
        
        if enable_preload:
            # 预加载所有图像到内存
            self.left_images = []
            self.right_images = []
            print("Loading left eye images...")
            for idx in tqdm(range(len(self.label_file))):
                left_path = os.path.join(img_dir, self.label_file.iloc[idx, 0])
                img = cv2.imread(left_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.left_images.append(img)
            
            print("Loading right eye images...")
            for idx in tqdm(range(len(self.label_file))):
                right_path = os.path.join(img_dir, self.label_file.iloc[idx, 1])
                img = cv2.imread(right_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.right_images.append(img)

    def __len__(self):
        return len(self.label_file)

    def __getitem__(self, idx):
        if self.enable_preload:
            # 从内存直接获取图像
            left_image = self.left_images[idx]
            right_image = self.right_images[idx]
        else:
            # 实时加载图像
            left_path = os.path.join(self.img_dir, self.label_file.iloc[idx, 0])
            right_path = os.path.join(self.img_dir, self.label_file.iloc[idx, 1])
            left_image = cv2.imread(left_path)
            right_image = cv2.imread(right_path)
            left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2RGB)
            right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2RGB)
        
        if self.transform:
            # Convert numpy array to PIL Image before applying transforms
            left_image = Image.fromarray(left_image)
            right_image = Image.fromarray(right_image)
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        labels = self.labels[idx]
        
        return left_image, right_image, labels