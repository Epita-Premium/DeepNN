from processing.heatmapGenerator import HeatmapGenerator
from processing.preprocessing import Preprocessor
import os
import cv2
import mediapipe as mp
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

test_data_dir = os.path.join(project_root, 'misc\DATASET\\train\\1')
heatmap_generator = HeatmapGenerator(
    heatmap_dir=os.path.join(project_root, 'misc/heatmaps'),
    model_path=os.path.join(project_root, 'src', 'processing', 'face_landmarker.task')
)


def test_goo():
    index = 0
    for dirpath, dirnames, filenames in os.walk(test_data_dir):
        for filename in filenames:
            if index <= 15 and (filename.endswith('.jpg') or filename.endswith('.png')):
                image_path = os.path.join(dirpath, filename)
                heatmap_generator.process_image(image_path, filename)
                index += 1


test_goo()
