from src.processing.heatmapGenerator import HeatmapGenerator
from src.processing.preprocessing import Preprocessor
import os
from torchvision.transforms.functional import to_tensor

preprocessor = Preprocessor()
heatmap_generator = HeatmapGenerator(face_predictor_path='src\processing\shape_predictor_68_face_landmarks.dat',
                                     heatmap_dir='misc/heatmaps')

for dirpath, dirnames, filenames in os.walk('misc/DATASET/test'):
    for filename in filenames:
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(dirpath, filename)

            with open(image_path, 'rb') as f:
                image = preprocessor.process_image(f).unsqueeze(0)  # Ajoutez une dimension de batch
                image_tensor = to_tensor(image)

            # Générez et enregistrez la heatmap
            heatmap_generator.generate_heatmap(image_tensor, filename)
