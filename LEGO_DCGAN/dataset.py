import os
import random
import torch
import torch.nn.parallel
import torch.utils.data
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError

# Set random seed for reproducibility - new results for every run
manualSeed = random.randint(1, 10000)
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True)

class LegoDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        img_name = os.path.join(self.root_dir, self.image_list[index])
        try:
            image = Image.open(img_name).convert("RGB")
        except (UnidentifiedImageError, OSError):
            # In case of truncated image or error in opening image
            dummy_image = Image.new("RGB", (64, 64))
            if self.transform is not None:
                dummy_image = self.transform(dummy_image)
            return dummy_image

        if self.transform is not None:
            image = self.transform(image)

        return image