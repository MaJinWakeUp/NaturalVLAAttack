"""Differentiable shared-patch rendering with per-frame homographies (Eq. 1)."""

import torch
import torch.nn.functional as F


class PatchRenderer:
    def __init__(self, box=(0, 174, 50, 50)):
        self.box = box  # x, y, width, height in the observation image

    def _grid(self, metadata, height, width, device):
        source = torch.tensor([[0., 0.], [1., 0.], [1., 1.], [0., 1.]], device=device)
        transforms = []
        for row in metadata:
            if "quad" in row:
                destination = torch.as_tensor(row["quad"], dtype=torch.float32, device=device)
                if destination.shape != (4, 2):
                    raise ValueError("quad must contain TL, TR, BR, BL pixel-boundary coordinates")
            else:
                x, y, w, h = row.get("box", self.box)
                if w <= 0 or h <= 0:
                    raise ValueError("Patch width and height must be positive")
                destination = torch.tensor([[x, y], [x+w, y], [x+w, y+h], [x, y+h]],
                                           dtype=torch.float32, device=device)
            # Solve the destination->unit-square homography.
            x, y = destination.unbind(-1)
            u, v = source.unbind(-1)
            zero, one = torch.zeros_like(x), torch.ones_like(x)
            a = torch.stack((torch.stack((x, y, one, zero, zero, zero, -u*x, -u*y), -1),
                             torch.stack((zero, zero, zero, x, y, one, -v*x, -v*y), -1)), 1)
            try:
                h = torch.linalg.solve(a.reshape(8, 8), source.reshape(8))
            except torch.linalg.LinAlgError as error:
                raise ValueError("Patch quadrilateral is degenerate") from error
            transforms.append(torch.cat((h, one[:1])).reshape(3, 3))
        yy, xx = torch.meshgrid(torch.arange(height, device=device) + .5,
                                torch.arange(width, device=device) + .5, indexing="ij")
        coordinates = torch.stack((xx, yy, torch.ones_like(xx)), -1)
        projected = torch.einsum("bij,hwj->bhwi", torch.stack(transforms), coordinates)
        denominator = projected[..., 2:]
        valid = denominator.abs() > 1e-8
        uv = projected[..., :2] / torch.where(valid, denominator, torch.ones_like(denominator))
        inside = valid & (uv >= 0).all(-1, keepdim=True) & (uv < 1).all(-1, keepdim=True)
        return uv * 2 - 1, inside.permute(0, 3, 1, 2).float()

    def __call__(self, images, patches, metadata):
        """Return [candidate, observation, channel, height, width]."""
        if images.ndim != 4 or patches.ndim != 4 or images.shape[1] != 3 or patches.shape[1] != 3:
            raise ValueError("Expected RGB BCHW images and NCHW patches")
        if len(metadata) != len(images):
            raise ValueError("One metadata row per observation is required")
        n, b = len(patches), len(images)
        height, width = images.shape[-2:]
        grid, mask = self._grid(metadata, height, width, images.device)
        expanded = patches[:, None].expand(n, b, *patches.shape[1:]).reshape(n*b, *patches.shape[1:])
        grid = grid[None].expand(n, b, height, width, 2).reshape(n*b, height, width, 2)
        warped = F.grid_sample(expanded.float(), grid, mode="bilinear", padding_mode="border",
                               align_corners=False).reshape(n, b, 3, height, width)
        return images[None] * (1-mask[None]) + warped * mask[None]
