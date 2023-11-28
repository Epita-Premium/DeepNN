from src.processing.heatmapGenerator import HeatmapGenerator
from src.processing.preprocessing import Preprocessor
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_data_dir = os.path.join(project_root, 'misc/DATASET/test/1')

preprocessor = Preprocessor()
heatmap_generator = HeatmapGenerator(heatmap_dir=os.path.join(project_root, 'misc/heatmaps'))
index = 0

for dirpath, dirnames, filenames in os.walk(test_data_dir):
    for filename in filenames:
        if index <= 15 and (filename.endswith('.jpg') or filename.endswith('.png')):
            image_path = os.path.join(dirpath, filename)

            with open(image_path, 'rb') as f:
                image_tensor = preprocessor.process_image(f, augment=True)
                if image_tensor.ndimension() == 4:
                    image_tensor = image_tensor.squeeze(0)

            # Générez et enregistrez la heatmap
            heatmap_generator.generate_heatmap(image_tensor, filename)
            index += 1


