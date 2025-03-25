import torch
import torch.nn as nn
import torch.optim as optim
from models.generator import UNetGenerator
from models.discriminator import PatchGANDiscriminator
from dataset import get_dataloader

# Initialize models
generator = UNetGenerator()
discriminator = PatchGANDiscriminator()

# Define loss functions and optimizers
criterion_GAN = nn.BCELoss()
criterion_L1 = nn.L1Loss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Load dataset
dataloader = get_dataloader("data/low_res", "data/high_res", batch_size=8)

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    for low_res, high_res in dataloader:
        optimizer_D.zero_grad()
        generated_high_res = generator(low_res)
        real_loss = criterion_GAN(discriminator(low_res, high_res), torch.ones_like(high_res))
        fake_loss = criterion_GAN(discriminator(low_res, generated_high_res.detach()), torch.zeros_like(high_res))
        loss_D = (real_loss + fake_loss) / 2
        loss_D.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        loss_GAN = criterion_GAN(discriminator(low_res, generated_high_res), torch.ones_like(high_res))
        loss_L1 = criterion_L1(generated_high_res, high_res) * 100
        loss_G = loss_GAN + loss_L1
        loss_G.backward()
        optimizer_G.step()

    print(f"Epoch {epoch+1}/{num_epochs} - Loss D: {loss_D.item()} - Loss G: {loss_G.item()}")

# Save the trained model
torch.save(generator.state_dict(), "generator.pth")
torch.save(discriminator.state_dict(), "discriminator.pth")
