import torch
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import config

from generator import Generator
from discriminator import Discriminator

# Denormalize & transform into image ready for visualization
transform_vis = transforms.Compose([
    transforms.Normalize((0, 0, 0), (1, 1, 1)),
    transforms.ToPILImage()
])

# Instantiate new Generator object, and load the model from its saved state
netG = Generator(config.noise, config.generator_filters, config.image_channels).to(config.device)
netG.load_state_dict(torch.load(config.model_path, map_location=config.device))
netG.eval()

num_images = 256 # a nice square number
fixed_noise = torch.randn(num_images, config.noise, 1, 1, device=config.device)

with torch.no_grad():
    generated_images = netG(fixed_noise).detach().cpu()

grid = vutils.make_grid(generated_images, nrow=16, padding=2, normalize=True)
plt.figure(figsize=(10, 10))
plt.imshow(transform_vis(grid))
plt.axis("off")
plt.show()