import os
import sys

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, os.path.dirname(__file__))
from generator import random_plate

# Index 37 (_) is the CTC blank — must match blank= in CTCLoss
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-_"
char_to_idx = {c: i for i, c in enumerate(CHARS)}
idx_to_char = {i: c for i, c in enumerate(CHARS)}


def encode(text):
    try:
        return [char_to_idx[c] for c in text]
    except KeyError as e:
        raise ValueError(f"Character {e} not in CHARS. Got text: {repr(text)}")


def decode(indices):
    return "".join([idx_to_char[i] for i in indices])


class LicensePlateDataset(Dataset):
    def __init__(self, size, country=None):
        """
        Args:
            size:    Number of samples per epoch.
            country: If None (default), the generator picks country per plate
                     based on the format (NL/DE/FR mix). Pass a string like
                     "NL" only if you want to force a single country.
        """
        self.size = size
        self.country = country  # None = let generator decide

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        img, text = random_plate(country=self.country)
        img = img.resize((188, 48))
        tensor = to_tensor(img)
        label = encode(text)
        return tensor, label
