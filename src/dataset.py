import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

EMOTION_LABELS = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral",
}


def get_transforms(aug_mode="none"):
    if aug_mode in ("none", "noaug"):
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
    if aug_mode == "light":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
            ]
        )
    if aug_mode == "strong":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5]),
                transforms.RandomErasing(p=0.2),
            ]
        )
    raise ValueError(f"Unknown augmentation mode: {aug_mode}")


class FERDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        self.has_labels = "emotion" in dataframe.columns

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pixels = self.data.loc[idx, "pixels"]
        pixel_array = np.array(pixels.split(), dtype=np.uint8).reshape(48, 48)
        image = Image.fromarray(pixel_array, mode="L")

        if self.transform:
            image = self.transform(image)

        if self.has_labels:
            label = int(self.data.loc[idx, "emotion"])
            return image, label
        return image


def get_dataloaders(data_dir="Data", aug_mode="none", batch_size=64, val_split=0.2, num_workers=0):
    train_csv = os.path.join(data_dir, "train.csv")
    test_csv = os.path.join(data_dir, "test.csv")

    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    if "Usage" in train_df.columns:
        train_df = train_df[["emotion", "pixels"]]

    train_data, val_data = train_test_split(
        train_df,
        test_size=val_split,
        random_state=42,
        stratify=train_df["emotion"],
    )

    train_transform = get_transforms(aug_mode)
    val_transform = get_transforms("none")

    train_dataset = FERDataset(train_data, transform=train_transform)
    val_dataset = FERDataset(val_data, transform=val_transform)
    test_dataset = FERDataset(test_df, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader
