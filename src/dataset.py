import os
import sys

from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms.functional import to_tensor

sys.path.insert(0, os.path.dirname(__file__))

# Index 37 (_) is the CTC blank — must match blank= in CTCLoss
CHARS = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ-_"
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
    def __init__(self, size=10000, country=None, data_dir=None):
        """
        Args:
            size:     Number of samples. Used for on-the-fly mode or to cap
                      the number of pre-generated samples loaded from disk.
            country:  Force a specific country label (None = mixed, default).
            data_dir: Path to a pre-generated dataset folder containing
                      index.txt (produced by generate_dataset.py).
                      If None, plates are generated on the fly.
        """
        self.country = country
        self.data_dir = data_dir

        if data_dir is not None:
            index_path = os.path.join(data_dir, "index.txt")
            if not os.path.exists(index_path):
                raise FileNotFoundError(
                    f"No index.txt found in {data_dir}. Run generate_dataset.py first."
                )
            with open(index_path) as f:
                lines = [l.strip() for l in f if l.strip()]
            # Cap to requested size
            self.entries = lines[:size]
            self.size = len(self.entries)
            self.mode = "disk"
        else:
            self.size = size
            self.mode = "live"

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if self.mode == "disk":
            filename, text = self.entries[index].split(" ", 1)
            img = Image.open(os.path.join(self.data_dir, filename)).convert("RGB")  # type: ignore
        else:
            from generator import random_plate

            img, text = random_plate(country=self.country)

        img = img.resize((188, 48))
        tensor = to_tensor(img)
        label = encode(text)
        return tensor, label
