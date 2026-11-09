"""Frozen Stable Diffusion backend with a differentiable VAE decoder."""

import torch


class StableDiffusionBackend:
    def __init__(self, checkpoint, device="cuda", dtype="float32", prompt="", guidance_scale=1.0,
                 local_files_only=False, cache_dir=None, revision=None, load_safety_checker=True,
                 negative_prompt=""):
        from diffusers import DDIMScheduler, StableDiffusionPipeline

        self.device = torch.device(device)
        self.dtype = getattr(torch, dtype)
        # Direct VAE/UNet optimization does not call the pipeline's image checker.
        # Keep the default for existing workflows; the object pilot omits unused weights.
        components = {} if load_safety_checker else {'safety_checker':None, 'requires_safety_checker':False}
        self.pipe = StableDiffusionPipeline.from_pretrained(
            checkpoint, torch_dtype=self.dtype, local_files_only=local_files_only,
            cache_dir=cache_dir, revision=revision, **components).to(device)
        self.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config, clip_sample=False)
        for module in (self.pipe.vae, self.pipe.unet, self.pipe.text_encoder):
            module.eval().requires_grad_(False)
        self.set_prompt(prompt, negative_prompt, guidance_scale)

    def set_prompt(self, prompt, negative_prompt="", guidance_scale=1.0):
        self.guidance_scale = float(guidance_scale)
        with torch.no_grad():
            prompts = [negative_prompt, prompt] if self.guidance_scale > 1 else [prompt]
            tokens = self.pipe.tokenizer(prompts, padding="max_length",
                                         max_length=self.pipe.tokenizer.model_max_length,
                                         truncation=True, return_tensors="pt")
            kwargs = {}
            if getattr(self.pipe.text_encoder.config, "use_attention_mask", False):
                kwargs["attention_mask"] = tokens.attention_mask.to(self.device)
            self.embeddings = self.pipe.text_encoder(tokens.input_ids.to(self.device), **kwargs)[0]

    @torch.no_grad()
    def from_noise(self, noise, steps):
        """Full deterministic DDIM trajectory from standard-normal noise; no image encode."""
        self.scheduler.set_timesteps(steps, device=self.device)
        latent = noise.to(self.device, torch.float32) * self.scheduler.init_noise_sigma
        for timestep in self.scheduler.timesteps:
            latent = self.step(latent, timestep)
        return latent.detach()

    def timesteps(self, config):
        self.scheduler.set_timesteps(config.ddim_steps, device=self.device)
        return self.scheduler.timesteps[-config.updates:]

    @torch.no_grad()
    def encode(self, patch):
        if any(size % self.pipe.vae_scale_factor for size in patch.shape[-2:]):
            raise ValueError("Seed size must be divisible by the VAE downsampling factor")
        distribution = self.pipe.vae.encode(patch.to(self.device, self.dtype) * 2 - 1).latent_dist
        return (distribution.mode() * self.pipe.vae.config.scaling_factor).float()

    def decode(self, latent):
        # Intentionally NOT decorated with no_grad: WB gradients pass through the frozen decoder.
        decoded = self.pipe.vae.decode(latent.to(self.dtype) / self.pipe.vae.config.scaling_factor).sample
        return (decoded.float() / 2 + .5).clamp(0, 1)

    @torch.no_grad()
    def add_noise(self, clean, noise, timestep):
        return self.scheduler.add_noise(clean, noise, timestep.reshape(1))

    @torch.no_grad()
    def step(self, latent, timestep):
        model_input = latent.to(self.dtype)
        if self.guidance_scale > 1:
            model_input = torch.cat((model_input, model_input))
            embeddings = self.embeddings.repeat_interleave(len(latent), dim=0)
        else:
            embeddings = self.embeddings.expand(len(latent), -1, -1)
        model_input = self.scheduler.scale_model_input(model_input, timestep)
        prediction = self.pipe.unet(model_input, timestep, encoder_hidden_states=embeddings).sample
        if self.guidance_scale > 1:
            unconditional, conditional = prediction.chunk(2)
            prediction = unconditional + self.guidance_scale * (conditional - unconditional)
        return self.scheduler.step(prediction.float(), timestep, latent, eta=0).prev_sample

    def query_alpha(self, timestep, convention):
        t = int(timestep)
        if convention == "step":
            value = self.scheduler.alphas[t]
        elif convention == "cumulative":
            value = self.scheduler.alphas_cumprod[t]
        elif convention == "ddim_transition":
            previous = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            previous_alpha = (self.scheduler.alphas_cumprod[previous] if previous >= 0
                              else self.scheduler.final_alpha_cumprod)
            value = self.scheduler.alphas_cumprod[t] / previous_alpha
        else:
            raise ValueError("Unknown alpha convention")
        return float(value)
