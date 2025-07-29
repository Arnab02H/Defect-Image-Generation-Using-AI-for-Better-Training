import zipfile
import io
import numpy as np
from PIL import Image
from pathlib import Path
import base64

GENERATED_DIR = Path("generated")

def save_images_to_zip(images, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for i, img in enumerate(images):
            img = (img.permute(1, 2, 0).numpy() * 0.5 + 0.5) * 255
            img = img.astype(np.uint8)
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