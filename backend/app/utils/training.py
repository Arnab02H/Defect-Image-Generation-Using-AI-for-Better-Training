import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision.transforms as transforms
from app.models import Generator
from app.models import Discriminator
from app.models.discriminator import weights_init
import zipfile
from PIL import Image
import io
import os

def train_gan(images, epochs=50, batch_size=4, lr=0.0002, beta1=0.5, lambda_cycle=5.0):
    # Set MPS fallback
    if torch.backends.mps.is_available():
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    # Device configuration
    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Validate input size and dataset size
    if images.size(2) < 64 or images.size(3) < 64:
        raise ValueError(f"Input images must be at least 64x64, got {images.size(2)}x{images.size(3)}")
    if len(images) < batch_size:
        raise ValueError(f"Dataset must have at least {batch_size} images, got {len(images)}")

    # Preprocess images: Normalize to [-1, 1]
    transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    images = transform(images)
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(images)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize models
    G_deface = Generator().to(device)
    G_restore = Generator().to(device)
    D = Discriminator().to(device)
    
    # Apply weight initialization
    G_deface.apply(weights_init)
    G_restore.apply(weights_init)
    D.apply(weights_init)

    # Optimizers
    optim_G = optim.Adam(list(G_deface.parameters()) + list(G_restore.parameters()), lr=lr, betas=(beta1, 0.999))
    optim_D = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))

    # Loss functions
    adversarial_loss = nn.BCELoss()
    cycle_loss = nn.L1Loss()

    # Training loop
    for epoch in range(epochs):
        for i, real_clean in enumerate(dataloader):
            real_clean = real_clean[0].to(device)
            batch_size = real_clean.size(0)

            # Labels for adversarial loss
            real_label = torch.ones(batch_size, 1, 1, 1, device=device) * 0.9
            fake_label = torch.zeros(batch_size, 1, 1, 1, device=device) + 0.1

            # --- Train Discriminator ---
            optim_D.zero_grad()
            real_d_output = D(real_clean)
            d_loss_real = adversarial_loss(real_d_output, real_label)
            fake_defect = G_deface(real_clean)
            fake_d_output = D(fake_defect.detach())
            d_loss_fake = adversarial_loss(fake_d_output, fake_label)
            d_loss = (d_loss_real + d_loss_fake) / 2
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1.0)
            optim_D.step()

            # --- Train Generators ---
            optim_G.zero_grad()
            fake_d_output = D(fake_defect)
            g_adv_loss = adversarial_loss(fake_d_output, real_label)
            restored_clean = G_restore(fake_defect)
            cycle_loss_deface = cycle_loss(restored_clean, real_clean)
            restored_defect = G_deface(restored_clean)
            cycle_loss_restore = cycle_loss(restored_defect, fake_defect)
            g_loss = g_adv_loss + lambda_cycle * (cycle_loss_deface + cycle_loss_restore)
            g_loss.backward()
            torch.nn.utils.clip_grad_norm_(list(G_deface.parameters()) + list(G_restore.parameters()), max_norm=1.0)
            optim_G.step()

            # Print progress with detailed losses
            if i % 50 == 0:
                print(f"Epoch [{epoch+1}/{epochs}] Batch [{i+1}/{len(dataloader)}] "
                      f"D_loss: {d_loss.item():.4f} G_loss: {g_loss.item():.4f} "
                      f"Adv_loss: {g_adv_loss.item():.4f} Cycle_deface: {cycle_loss_deface.item():.4f} "
                      f"Cycle_restore: {cycle_loss_restore.item():.4f}")

    return G_deface

def generate_images(images, generator, num_images):
    # Set MPS fallback
    if torch.backends.mps.is_available():
        os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

    device = torch.device("cuda" if torch.cuda.is_available() else
                         "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Validate input size
    if images.size(2) < 64 or images.size(3) < 64:
        raise ValueError(f"Input images must be at least 64x64, got {images.size(2)}x{images.size(3)}")
    
    # Preprocess images
    transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    images = transform(images)
    
    # Create dataset and dataloader
    dataset = torch.utils.data.TensorDataset(images)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    generator = generator.to(device)
    generator.eval()
    generated_images = []

    with torch.no_grad():
        for i, img in enumerate(dataloader):
            if i >= num_images:
                break
            img = img[0].to(device)
            defect_img = generator(img)
            # Denormalize: [-1, 1] -> [0, 255]
            defect_img = (defect_img * 0.5 + 0.5) * 255.0
            defect_img = defect_img.squeeze().cpu().numpy().astype(np.uint8)
            # Transpose from [C, H, W] to [H, W, C]
            if defect_img.shape[0] == 3:
                defect_img = defect_img.transpose(1, 2, 0)
            generated_images.append(defect_img)
            print(f"Generated image {i+1} shape: {defect_img.shape}, range: [{defect_img.min()}, {defect_img.max()}]")
    
    return generated_images
