import config
import torch
import torchvision.utils as vutils
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import LegoDataset
from generator import Generator
from discriminator import Discriminator
import config

# Load dataset
dataset = LegoDataset(root_dir=config.data_path, transform=config.transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)

# Initialize networks
netG = Generator(config.noise, config.generator_filters, config.image_channels).to(config.device)
netD = Discriminator(config.image_channels, config.discriminator_filters).to(config.device)
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))


def train():
    for epoch in range(config.num_epochs):
        for i, data in enumerate(dataloader, 0):
            real_images = data.to(config.device)
            batch_size = real_images.size(0)
            label = torch.full((batch_size, 1, 1, 1), 1.0, device=config.device)
            noise = torch.randn(batch_size, config.nz, 1, 1, device=config.device)

            # Discriminator: maximize ln( D(x) ) + ln( 1 - D(G(z)) )
            config.netD.zero_grad()
            output_real = config.netD(real_images)
            errD_real = config.criterion(output_real, label)
            errD_real.backward()
            D_x = output_real.mean().item()

            fake_images = config.netG(noise)
            label.fill_(0.0)
            output_fake = config.netD(fake_images.detach())
            errD_fake = config.criterion(output_fake, label)
            errD_fake.backward()
            D_G_z1 = output_fake.mean().item()
            errD = errD_real + errD_fake
            config.optimizerD.step()

            # Generator: maximize ln( D(G(z)) )
            config.netG.zero_grad()
            label.fill_(1.0)
            output = config.netD(fake_images)
            errG = config.criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            config.optimizerG.step()

            if i % 50 == 0:
                print(
                    f"[{epoch}/{config.num_epochs}] "
                    f"Loss_D: {errD.item():.4f} "
                    f"Loss_G: {errG.item():.4f} "
                    f"D(x): {D_x:.4f} "
                    f"D(G(z)): {D_G_z1:.4f}/{D_G_z2:.4f}"
                )

        with torch.no_grad():
            fake = config.netG(config.fixed_noise).detach().cpu()
            vutils.save_image(
                fake, f"{config.output_dir}/fake_samples_epoch_{epoch + 1}.png",
                normalize=True,
            )
    torch.save(config.netG.state_dict(), config.save_path)