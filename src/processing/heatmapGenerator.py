import cv2
import numpy as np
import os
from torchvision.transforms.functional import to_pil_image
import matplotlib as plt
import cv2
import numpy as np
import os
import dlib
from torchvision.transforms.functional import to_pil_image


class HeatmapGenerator:
    def __init__(self, heatmap_dir, shape_predictor_path = os.path.abspath("processing/shape_predictor_68_face_landmarks.dat")):
        # Chargement du modèle dlib pour la détection des repères faciaux
        self.landmark_detector = dlib.shape_predictor(shape_predictor_path)

        # Répertoire pour les heatmaps
        self.heatmap_dir = heatmap_dir
        if not os.path.exists(heatmap_dir):
            os.makedirs(heatmap_dir)

    def generate_heatmap(self, preprocessed_image, original_filename):
        if preprocessed_image.ndimension() == 4:
            preprocessed_image = preprocessed_image.squeeze(0)
        rgb_image = np.array(to_pil_image(preprocessed_image))

        # Utilisation de dlib pour la détection des visages
        face_detector = dlib.get_frontal_face_detector()
        detected_faces = face_detector(rgb_image, 1)

        heatmap = np.zeros((224, 224), dtype=np.float32)

        for face in detected_faces:
            landmarks = self.landmark_detector(rgb_image, face)

            for i in range(68):  # dlib détecte 68 repères
                x, y = landmarks.part(i).x, landmarks.part(i).y
                if 0 <= x < 224 and 0 <= y < 224:
                    cv2.circle(heatmap, (int(x), int(y)), radius=2, color=1, thickness=-1)

        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=3, sigmaY=3, borderType=cv2.BORDER_DEFAULT)

        heatmap_filename = os.path.join(self.heatmap_dir, os.path.splitext(original_filename)[0] + '.png')
        cv2.imwrite(heatmap_filename, heatmap * 255)
        print("Heatmap generated for", original_filename)

