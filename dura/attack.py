"""Algorithm 1 and the white-/black-box directions in Eqs. 8–9."""

from dataclasses import dataclass
import math
import time
import torch

from .config import AttackConfig


@dataclass
class AttackResult:
    patch: torch.Tensor
    latent: torch.Tensor
    history: list[dict]


def score_direction(latent, alpha, samples, chunk_size, loss_fn, generator, baseline="none"):
    """Streaming Eq. 9 with O(chunk_size) latent memory and no autograd.

    loss_fn accepts noised candidates and returns one batch-averaged loss per
    candidate. Leave-one-out centering is unbiased; ordinary sample-mean
    centering without a K/(K-1) correction is not.
    """
    if not 0 < alpha < 1:
        raise ValueError("Query alpha must be strictly between zero and one")
    if samples < 1 or chunk_size < 1 or (baseline == "leave_one_out" and samples < 2):
        raise ValueError("Invalid score estimator sample count")
    if baseline not in {"none", "leave_one_out"}:
        raise ValueError("Unknown baseline")
    if latent.shape[0] != 1:
        raise ValueError("DURA optimizes a single shared latent")
    root_alpha, sigma = math.sqrt(alpha), math.sqrt(1-alpha)
    sum_noise = torch.zeros_like(latent, dtype=torch.float32)
    sum_loss_noise = torch.zeros_like(sum_noise)
    sum_loss = torch.zeros((), device=latent.device)
    with torch.no_grad():
        for start in range(0, samples, chunk_size):
            count = min(chunk_size, samples-start)
            noise = torch.randn((count, *latent.shape[1:]), device=latent.device,
                                dtype=torch.float32, generator=generator)
            losses = loss_fn(root_alpha * latent + sigma * noise).float()
            if losses.shape != (count,) or not torch.isfinite(losses).all():
                raise ValueError("Expected one finite query loss per candidate")
            sum_loss += losses.sum()
            sum_noise += noise.sum(0, keepdim=True)
            sum_loss_noise += (losses.reshape(count, *([1]*(latent.ndim-1))) * noise).sum(0, keepdim=True)
    if baseline == "leave_one_out":
        direction = (sum_loss_noise - sum_loss / samples * sum_noise) / (samples-1)
    else:
        direction = sum_loss_noise / samples
    return direction * (root_alpha / sigma), float(sum_loss / samples)


class DURAAttack:
    def __init__(self, backend, objective, config: AttackConfig):
        self.backend, self.objective, self.config = backend, objective, config

    def run(self, seed_patch, batches, callback=None, checkpoint_callback=None):
        cfg, backend = self.config, self.backend
        started = time.monotonic()
        print("Encoding seed and constructing clean DDIM anchors...", flush=True)
        generator = torch.Generator(device=backend.device).manual_seed(cfg.seed)
        timesteps = backend.timesteps(cfg)
        with torch.no_grad():
            clean = backend.encode(seed_patch)
            noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
            latent = backend.add_noise(clean, noise, timesteps[0])
            # Store anchors before their DDIM step so each is matched to z_t, not z_(t-1).
            anchor, anchors = latent.clone(), []
            for timestep in timesteps:
                anchors.append(anchor.detach().cpu())
                anchor = backend.step(anchor, timestep)
        print(f"Starting {len(timesteps)} {cfg.mode} updates", flush=True)
        history = []
        for index, timestep in enumerate(timesteps):
            batch = next(batches).to(backend.device)
            with torch.no_grad():
                mixed = (1-cfg.anchor_weight)*latent + cfg.anchor_weight*anchors[index].to(latent)
                denoised = backend.step(mixed, timestep).detach()
            if cfg.mode == "whitebox":
                with torch.enable_grad():
                    denoised.requires_grad_(True)
                    loss = self.objective(backend.decode(denoised), batch, "whitebox").mean()
                    direction, = torch.autograd.grad(loss, denoised)
                    mean_loss = float(loss.detach())
                direction = direction.detach()
            else:
                def query(candidates):
                    if cfg.candidate_denoise:
                        candidates = backend.step(candidates, timestep)
                    return self.objective(backend.decode(candidates), batch, "blackbox")
                direction, mean_loss = score_direction(
                    denoised, backend.query_alpha(timestep, cfg.query_alpha), cfg.samples,
                    cfg.candidate_batch_size, query, generator, cfg.baseline)
            if not math.isfinite(mean_loss) or not torch.isfinite(direction).all():
                raise FloatingPointError(f"Non-finite loss/gradient at update {index}")
            with torch.no_grad():
                # Raw gradient, not sign/Adam/clipping: Eq. 7.
                latent = denoised.detach() - cfg.step_size * direction
            record = {"update": index+1, "timestep": int(timestep), "loss": mean_loss,
                      "gradient_norm": float(direction.norm()),
                      "candidate_queries": self.objective.candidate_queries,
                      "observation_queries": self.objective.observation_queries}
            history.append(record)
            record["elapsed_seconds"] = time.monotonic() - started
            if latent.is_cuda:
                record["peak_gpu_memory_gib"] = torch.cuda.max_memory_allocated(latent.device) / 2**30
            if callback:
                callback(record)
            if checkpoint_callback:
                checkpoint_callback(record, latent)
        with torch.no_grad():
            return AttackResult(backend.decode(latent), latent.detach(), history)
