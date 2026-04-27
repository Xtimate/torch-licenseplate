import os
import sys

from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, os.path.dirname(__file__))
from generator import random_plate

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ-_"
char_to_idx = {c: i for i, c in enumerate(CHARS)}
idx_to_char = {i: c for i, c in enumerate(CHARS)}


def encode(text):
    return [char_to_idx[c] for c in text]


def decode(indices):
    return "".join([idx_to_char[i] for i in indices])


class LicensePlateDataset(Dataset):
    def __init__(self, size, country="NL"):
        self.size = size
        self.country = country

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        img, text = random_plate(self.country)
        img = img.resize((188, 48))
        tensor = to_tensor(img)
        label = encode(text)
        return tensor, label
