import torch as th
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
import pytorch_lightning as pl
from utils.configuration import Configuration
from sklearn.model_selection import train_test_split
import os
from PIL import Image

class ImageDirectoryDataset(Dataset):
    def __init__(self, img_dir, train=True):
        self.img_dir = img_dir
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            transforms.ToTensor(),          # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalize with ImageNet stats
                                 std=[0.229, 0.224, 0.225])
        ])
        self.train = train

        # Get the file names of all images in the directory
        self.filenames = os.listdir(img_dir)
        # filter out non-image files
        self.filenames = [f for f in self.filenames if f.endswith('.png') or f.endswith('.jpg')]

        # Split filenames into train and test sets
        self.train_filenames, self.test_filenames = train_test_split(self.filenames, test_size=0.2, random_state=42)

    def __len__(self):
        # The length is the number of images in the train set if train=True, else the number of images in the test set
        if self.train:
            return len(self.train_filenames)
        else:
            return len(self.test_filenames)

    def __getitem__(self, idx):
        # Get the filename of the image at the given index, depending on whether train=True or not
        if self.train:
            filename = self.train_filenames[idx]
        else:
            filename = self.test_filenames[idx]

        # Load the image from the file
        img_path = os.path.join(self.img_dir, filename)
        image = Image.open(img_path).convert("RGB")

        # Apply the transform to the image
        image = self.transform(image)

        # Return the image
        return image, 0


class FFHQDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Configuration):
        super().__init__()
        self.num_workers = cfg.num_workers
        self.batch_size  = cfg.model.batch_size
        self.data_dir    = cfg.data.data_dir

    def setup(self, stage=None):
        # Load MNIST Data
        if stage == 'fit' or stage == 'val' or stage is None:
            self.ffhq_train = ImageDirectoryDataset(img_dir=self.data_dir, train=True)
            self.ffhq_val = ImageDirectoryDataset(img_dir=self.data_dir, train=False)
        
        if stage == 'test' or stage is None:
            self.ffhq_test = ImageDirectoryDataset(img_dir=self.data_dir, train=False)

    def train_dataloader(self):
        return DataLoader(self.ffhq_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.ffhq_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.ffhq_test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
