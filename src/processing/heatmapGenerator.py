import cv2
import numpy as np
import dlib
import os
from torchvision.transforms.functional import to_pil_image


class HeatmapGenerator:
    def __init__(self, face_predictor_path, heatmap_dir):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(face_predictor_path)
        self.heatmap_dir = heatmap_dir
        if not os.path.exists(heatmap_dir):
            os.makedirs(heatmap_dir)

    def generate_heatmap(self, preprocessed_image, original_filename):
        gray = cv2.cvtColor(np.array(to_pil_image(preprocessed_image)), cv2.COLOR_RGB2GRAY)
        faces = self.detector(gray)

        heatmap = np.zeros((224, 224), dtype=np.float32)
        for face in faces:
            landmarks = self.predictor(gray, face)
            for n in range(0, landmarks.num_parts):
                x, y = landmarks.part(n).x, landmarks.part(n).y
                heatmap[y, x] = 1.0

        heatmap = cv2.GaussianBlur(heatmap, (0, 0), sigmaX=2, sigmaY=2, borderType=cv2.BORDER_DEFAULT)
        heatmap_filename = os.path.join(self.heatmap_dir, os.path.splitext(original_filename)[0] + '.png')
        cv2.imwrite(heatmap_filename, heatmap * 255)
