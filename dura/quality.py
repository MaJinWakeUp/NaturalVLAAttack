"""Appendix D crop and artifact metrics; optional perceptual metric backend."""

import torch


def local_crop(images):
    if images.shape[-2:] != (224, 224):
        raise ValueError("Paper crop expects 224x224 scenes")
    return images[..., 32:224, 0:192]


def patch_tv(patches):
    if min(patches.shape[-2:]) < 2:
        raise ValueError("TV needs a patch of at least 2x2 pixels")
    return ((patches[..., 1:, :] - patches[..., :-1, :]).abs().mean(dim=(-3,-2,-1))
            + (patches[..., :, 1:] - patches[..., :, :-1]).abs().mean(dim=(-3,-2,-1)))


def rgb_to_lab(images):
    rgb = images.float().movedim(-3, -1).clamp(0, 1)
    linear = torch.where(rgb <= .04045, rgb/12.92, ((rgb+.055)/1.055).pow(2.4))
    matrix = rgb.new_tensor([[.4124564, .3575761, .1804375],
                             [.2126729, .7151522, .0721750], [.0193339, .1191920, .9503041]])
    xyz = (linear @ matrix.T)/rgb.new_tensor([.95047, 1., 1.08883])
    delta = 6/29
    f = torch.where(xyz > delta**3, xyz.clamp_min(0).pow(1/3), xyz/(3*delta**2)+4/29)
    x, y, z = f.unbind(-1)
    return torch.stack((116*y-16, 500*(x-y), 200*(y-z)), -1)


def boundary_seam_energy(images, box):
    """Mean CIE76 distance across valid boundaries of an axis-aligned patch."""
    x, y, w, h = box
    if any(int(v) != v for v in box):
        raise ValueError("Boundary metrics require integer pixel coordinates")
    x, y, w, h = map(int, box)
    height, width = images.shape[-2:]
    if not (0 <= x < x+w <= width and 0 <= y < y+h <= height):
        raise ValueError("Metric patch box must be inside the image")
    lab = rgb_to_lab(images)
    edges = []
    if y > 0:
        edges.append((lab[..., y, x:x+w, :] - lab[..., y-1, x:x+w, :]).norm(dim=-1))
    if y+h < height:
        edges.append((lab[..., y+h, x:x+w, :] - lab[..., y+h-1, x:x+w, :]).norm(dim=-1))
    if x > 0:
        edges.append((lab[..., y:y+h, x, :] - lab[..., y:y+h, x-1, :]).norm(dim=-1))
    if x+w < width:
        edges.append((lab[..., y:y+h, x+w, :] - lab[..., y:y+h, x+w-1, :]).norm(dim=-1))
    if not edges:
        raise ValueError("Patch has no boundary inside this image")
    return torch.cat(edges, dim=-1).mean(-1)


class PerceptualMetrics:
    """Install pyiqa separately for NIQE/DISTS/SSIM/CLIP-IQA.

    Checkpoint and library versions affect these values; record them for any
    comparison. The paper does not identify its exact metric checkpoints.
    """

    def __init__(self, device="cuda"):
        import pyiqa
        self.niqe = pyiqa.create_metric("niqe", device=device)
        self.dists = pyiqa.create_metric("dists", device=device)
        self.ssim = pyiqa.create_metric("ssim", device=device)
        self.clip = pyiqa.create_metric("clipiqa", device=device)
        from pyiqa.archs.clip_imports import clip
        self.clip.net.prompt_pairs = clip.tokenize(["natural photo", "synthetic photo"])

    @torch.no_grad()
    def __call__(self, patched, clean):
        from torch.nn.functional import interpolate
        patched, clean = local_crop(patched), local_crop(clean)
        clip_input = interpolate(patched, size=(224,224), mode="bicubic", align_corners=False).clamp(0,1)
        return {"niqe": self.niqe(patched), "dists": self.dists(patched, clean),
                "ssim": self.ssim(patched, clean), "clip_natural": self.clip(clip_input)}
