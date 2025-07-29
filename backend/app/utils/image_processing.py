from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms
import zipfile
UPLOAD_DIR = Path("uploads")

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

def load_images_from_zip(zip_path):
    images = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(UPLOAD_DIR)
        for file_name in zip_ref.namelist():
            if file_name.endswith(('.png', '.jpg', '.jpeg')):
                img_path = UPLOAD_DIR / file_name
                try:
                    img = Image.open(img_path).convert('RGB')
                    img = transform(img)
                    images.append(img)
                except:
                    continue
    return torch.stack(images)