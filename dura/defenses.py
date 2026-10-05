"""Optional evaluation transformations from Appendix E."""

from io import BytesIO
import numpy as np
from PIL import Image
import torch


def transform(images, kind="none", strength=None, generator=None):
    if kind == "none":
        return images
    if kind == "gaussian":
        if strength is None or strength < 0:
            raise ValueError("Gaussian sigma must be nonnegative")
        noise = torch.randn(images.shape, generator=generator, device=images.device, dtype=images.dtype)
        return (images + float(strength)*noise).clamp(0, 1)
    if kind == "bit_depth":
        if strength is None or int(strength) != strength or not 1 <= strength <= 8:
            raise ValueError("Bit depth must be an integer in [1,8]")
        levels = 2**int(strength)-1
        return (images.clamp(0, 1)*levels).round()/levels
    if kind == "jpeg":
        if strength is None or int(strength) != strength or not 1 <= strength <= 100:
            raise ValueError("JPEG quality must be an integer in [1,100]")
        results = []
        for image in images:
            array = (image.detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)*255).round().astype(np.uint8)
            buffer = BytesIO()
            Image.fromarray(array).save(buffer, format="JPEG", quality=int(strength))
            buffer.seek(0)
            with Image.open(buffer) as decoded:
                tensor = torch.from_numpy(np.array(decoded.convert("RGB"), dtype=np.float32)/255)
            results.append(tensor.permute(2, 0, 1))
        return torch.stack(results).to(images)
    raise ValueError(f"Unknown defense: {kind}")
