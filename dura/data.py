"""Portable JSONL observation manifests, independent of RLDS/TensorFlow."""

from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


def load_image(path, size=None):
    with Image.open(path) as source:
        rgb = source.convert("RGB")
        if size is not None:
            rgb = rgb.resize((size, size), Image.Resampling.LANCZOS)
        array = np.array(rgb, dtype=np.float32) / 255
    return torch.from_numpy(array).permute(2, 0, 1)


def save_image(tensor, path):
    array = tensor.detach().float().cpu().clamp(0, 1)
    if array.ndim == 4:
        if array.shape[0] != 1:
            raise ValueError("save_image expects one image")
        array = array[0]
    Image.fromarray((array.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(path)


@dataclass
class ObservationBatch:
    images: torch.Tensor
    instructions: list[str]
    metadata: list[dict]

    def to(self, device):
        return ObservationBatch(self.images.to(device), self.instructions, self.metadata)


class ManifestDataset(Dataset):
    """Rows contain image, instruction, and optional state/wrist_image/box/quad.

    Coordinates refer to the loaded image after optional resize. Images must
    already have the policy's camera orientation. No action labels are needed.
    """

    def __init__(self, manifest, image_size=224):
        self.path = Path(manifest).resolve()
        self.image_size = image_size
        self.records = []
        for line in self.path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row.get("instruction"), str) or "image" not in row:
                raise ValueError("Each manifest row needs image and instruction")
            for key in ("image", "wrist_image"):
                if key in row:
                    row[key] = str((self.path.parent / row[key]).resolve())
                    if not Path(row[key]).is_file():
                        raise FileNotFoundError(row[key])
            self.records.append(row)
        if not self.records:
            raise ValueError("Manifest is empty")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        return load_image(row["image"], self.image_size), row["instruction"], dict(row)


def collate_observations(rows):
    images, instructions, metadata = zip(*rows)
    return ObservationBatch(torch.stack(images), list(instructions), list(metadata))


def cycle_batches(loader):
    while True:
        found = False
        for batch in loader:
            found = True
            yield batch
        if not found:
            raise ValueError("Observation loader yielded no batches")
