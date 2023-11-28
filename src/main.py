import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np
import cv2
import dlib
from processing.heatmapGenerator import HeatmapGenerator
from processing.preprocessing import Preprocessor
from Attribution_methods.gradientSimple import GradientAttribution
from train import train_model
from torchvision import datasets


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_epochs = 75
    learning_rate = 1e-4

    # Charger le modèle ResNet50
    model = models.resnet50(pretrained=True).to(device)
    model.train()

    # Optimiseur
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Fonction de perte
    criterion = torch.nn.CrossEntropyLoss()

    # Préprocesseur
    preprocessor = Preprocessor()

    # DataLoader pour l'entraînement
    train_loader = DataLoader(
        datasets.ImageFolder("path/to/DATASET/train", transform=preprocessor.base_transform),
        batch_size=4,
        shuffle=True
    )

    heatmap_generator = HeatmapGenerator("path/to/heatmap_dir")
    gradient_attrib = GradientAttribution(model)

    # Entraînement du modèle
    train_model(model, train_loader, num_epochs, criterion, optimizer, device, heatmap_generator, gradient_attrib)

    # Sauvegarde du modèle
    torch.save(model.state_dict(), "resnet50_pal.pth")


if __name__ == "__main__":
    main()
