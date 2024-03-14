import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torchvision.transforms as transforms

# Hyperparameters
noise = 100                   # latent vector size
generator_filters = 128       # more filters => more accurate images
discriminator_filters = 128   # ideally same as generator_filters
image_channels = 3            # 3 channels - RGB

device = torch.device("cpu")  # i don't have GPU :( give me job to buy gpu pls

data_path = "minifig_images"

transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

criterion = nn.BCELoss()

num_epochs = 1
trial = str(3)
output_dir = f"minifig_output/Trial_{trial}"
os.makedirs(output_dir, exist_ok=True)

model_path = f"models/Trial_{trial}"
os.makedirs(model_path, exist_ok=True)

fixed_noise = torch.randn(64, noise, 1, 1, device=device)