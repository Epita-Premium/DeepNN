import cv2
import numpy as np
import os
from torchvision.transforms.functional import to_pil_image
import matplotlib as plt


class HeatmapGenerator:
    def __init__(self, heatmap_dir, lbfmodel_name='lbfmodel.yaml', haarcascade_name='haarcascade_frontalface_alt2.xml'):

        model_dir = os.path.dirname(os.path.abspath(__file__))  # Répertoire du fichier heatmapGenerator.py
        lbfmodel_path = os.path.join(model_dir, lbfmodel_name)
        haarcascade_path = os.path.join(model_dir, haarcascade_name)

        # Chargement des modèles
        self.face_detector = cv2.CascadeClassifier(haarcascade_path)
        self.landmark_detector = cv2.face.createFacemarkLBF()
        self.landmark_detector.loadModel(lbfmodel_path)

        # Répertoire pour les heatmaps
        self.heatmap_dir = heatmap_dir
        if not os.path.exists(heatmap_dir):
            os.makedirs(heatmap_dir)

    def generate_heatmap(self, preprocessed_image, original_filename):
        if preprocessed_image.ndimension() == 4:
            preprocessed_image = preprocessed_image.squeeze(0)
        gray = cv2.cvtColor(np.array(to_pil_image(preprocessed_image)), cv2.COLOR_RGB2GRAY)

        # Détection des visages
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        heatmap = np.zeros((224, 224), dtype=np.float32)

        for (x, y, w, h) in faces:
            face_roi = np.array([[x, y, x + w, y + h]], dtype=np.int32)

            # Détection des repères faciaux pour chaque visage
            _, landmarks = self.landmark_detector.fit(gray, face_roi)
            for landmark in landmarks:
                for x, y in landmark[0]:
                    if 0 <= x < 224 and 0 <= y < 224:
                        cv2.circle(heatmap, (int(x), int(y)), radius=2, color=1, thickness=-1)

        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)

        heatmap_filename = os.path.join(self.heatmap_dir, os.path.splitext(original_filename)[0] + '.png')
        cv2.imwrite(heatmap_filename, heatmap * 255)
        print("Heatmaps generate")

