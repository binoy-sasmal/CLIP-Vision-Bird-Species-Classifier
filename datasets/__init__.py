import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class BirdDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        for label, species in enumerate(os.listdir(self.root_dir)):
            species_dir = os.path.join(self.root_dir, species)
            if os.path.isdir(species_dir):
                for img_file in os.listdir(species_dir):
                    self.image_paths.append(os.path.join(species_dir, img_file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

def get_transforms(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
