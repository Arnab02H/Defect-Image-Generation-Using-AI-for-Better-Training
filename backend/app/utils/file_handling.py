import zipfile
import io
import numpy as np
from PIL import Image
from pathlib import Path
import base64

GENERATED_DIR = Path("generated")
UPLOAD_DIR=Path("uploads")


def save_images_to_zip(images, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, img in enumerate(images):
            # Input images are NumPy arrays with shape [H, W, C], values in [0, 255], dtype uint8
            # No need for permute or normalization since generate_images already handled it
            pil_img = Image.fromarray(img)
            img_byte_arr = io.BytesIO()
            pil_img.save(img_byte_arr, format='PNG')
            zipf.writestr(f'generated_{i}.png', img_byte_arr.getvalue())

def get_sample_images(zip_path, num_samples=5):
    sample_images = []
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        image_files = [f for f in zipf.namelist() if f.endswith('.png')]
        for file_name in image_files[:num_samples]:
            try:
                with zipf.open(file_name) as img_file:
                    img_data = img_file.read()
                    # Convert to base64
                    base64_string = base64.b64encode(img_data).decode('utf-8')
                    sample_images.append(base64_string)
            except:
                continue
    return sample_images