import torch
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from PIL import Image
import os
import numpy as np
import cv2
import dlib

# Vos classes déjà définies
from processing.preprocessing import Preprocessor
from processing.heatmapGenerator import HeatmapGenerator
from Attribution_methods.gradientSimple import GradientAttribution

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_epochs = 10
learning_rate = 1e-4

# Charger le modèle ResNet50
model = models.resnet50(pretrained=True).to(device)
model.eval()

# Optimiseur
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Définir la perte (vous pouvez choisir une fonction de perte adaptée à votre problème)
criterion = torch.nn.CrossEntropyLoss()

# Initialiser vos classes
preprocessor = Preprocessor()
heatmap_generator = HeatmapGenerator(heatmap_dir="path/to/heatmaps")
gradient_attrib = GradientAttribution(model)


# Fonction de prétraitement pour l'image
def process_image(image_path):
    image = Image.open(image_path)
    image = preprocessor.process_image(image)
    return image


def custom_loss_function(original_loss, gradients, heatmap, alpha=0.5):
    """
    Custom loss function qui ajuste la perte originale en fonction des heatmaps et des gradients.

    :param original_loss: Perte calculée par la fonction de perte standard.
    :param gradients: Gradients d'attribution pour l'image.
    :param heatmap: Heatmap générée pour l'image.
    :param alpha: Poids pour l'ajustement de la perte en fonction des heatmaps et des gradients.
    :return: Perte ajustée.
    """
    # Normaliser les gradients et les heatmaps pour les rendre comparables
    normalized_gradients = torch.norm(gradients, p=2, dim=1)
    normalized_heatmap = heatmap / heatmap.max()

    # Calculer la différence entre les gradients et les heatmaps
    diff = torch.abs(normalized_gradients - normalized_heatmap)

    # Ajuster la perte originale en fonction de la différence
    adjusted_loss = original_loss + alpha * diff.mean()

    return adjusted_loss


def train_model(model, data_loader, num_epochs, criterion, optimizer, device, heatmap_generator, gradient_attrib):
    for epoch in range(num_epochs):
        for batch in data_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            for image, label in zip(images, labels):
                # Forward pass
                output = model(image.unsqueeze(0))
                loss = criterion(output, label.unsqueeze(0))

                # Backward pass et optimisation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Générer la heatmap
                heatmap = heatmap_generator.generate_heatmap(image.cpu().numpy(), "temp_filename")

                # Calculer les gradients d'attribution
                gradients = gradient_attrib.compute_gradients(image.unsqueeze(0), label.item())

                # Ajuster la perte en fonction des heatmaps et des gradients (PAL)
                adjusted_loss = custom_loss_function(loss, gradients, heatmap)
                adjusted_loss.backward()

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
            torch.save(model.state_dict(), "resnet50_pal.pth")
