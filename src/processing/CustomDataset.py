from random import random

import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import os
import torch


def apply_transform(image, heatmap, transform):
    # Générer un état aléatoire et l'appliquer à la fois à l'image et à la heatmap
    seed = np.random.randint(2147483647)
    random.seed(seed)
    torch.manual_seed(seed)
    image = transform(image)

    random.seed(seed)
    torch.manual_seed(seed)
    heatmap = transform(heatmap)

    return image, heatmap


class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        # Explorer les sous-dossiers pour collecter les noms de fichiers
        self.image_names = []
        for emotion_dir in os.listdir(image_dir):
            for image in os.listdir(os.path.join(image_dir, emotion_dir)):
                self.image_names.append((emotion_dir, image))

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        emotion_dir, img_name = self.image_names[idx]
        img_path = os.path.join(self.image_dir, emotion_dir, img_name)
        heatmap_name = os.path.splitext(img_name)[0] + '_h.png'
        heatmap_path = os.path.join(self.image_dir, emotion_dir, heatmap_name)
        if not os.path.exists(heatmap_path):
            raise FileNotFoundError(f"Heatmap not found for {img_name}")

        image = Image.open(img_path).convert('RGB')
        heatmap = Image.open(heatmap_path).convert('RGB')

        if self.transform:
            image, heatmap = apply_transform(image, heatmap, self.transform)

        return image, heatmap


# Utilisation du CustomDataset
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(10),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])
current_dir = os.path.dirname(__file__)
project_root = os.path.dirname(current_dir)
train_dataset_path = os.path.join(project_root, 'misc', 'DATASET', 'train')
custom_dataset = CustomDataset(image_dir=train_dataset_path, transform=transform)
data_loader = DataLoader(custom_dataset, batch_size=16, shuffle=True)
