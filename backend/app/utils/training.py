import torch
import torch.nn as nn
import torch.optim as optim
from app.models import Generator, Discriminator 
from app.models.discriminator import weights_init
import os 

### This Code is Working 
def train_gan(images, epochs=50, batch_size=32, latent_dim=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    generator = Generator(latent_dim).to(device)
    discriminator = Discriminator().to(device)
    
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    real_label = 1.0
    fake_label = 0.0
    
    dataset = torch.utils.data.TensorDataset(images)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        for real_imgs in dataloader:
            real_imgs = real_imgs[0].to(device)
            batch_size = real_imgs.size(0)
            
            # Train Discriminator
            discriminator.zero_grad()
            label = torch.full((batch_size,), real_label, device=device)
            output = discriminator(real_imgs).view(-1)
            err_d_real = criterion(output, label)
            err_d_real.backward()
            
            noise = torch.randn(batch_size, latent_dim, 1, 1, device=device)
            fake_imgs = generator(noise)
            label.fill_(fake_label)
            output = discriminator(fake_imgs.detach()).view(-1)
            err_d_fake = criterion(output, label)
            err_d_fake.backward()
            optimizer_d.step()
            
            # Train Generator
            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake_imgs).view(-1)
            err_g = criterion(output, label)
            err_g.backward()
            optimizer_g.step()
    
    return generator

def generate_images(generator, num_images, latent_dim=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    noise = torch.randn(num_images, latent_dim, 1, 1, device=device)
    with torch.no_grad():
        generated_imgs = generator(noise).cpu()
    return generated_imgs
