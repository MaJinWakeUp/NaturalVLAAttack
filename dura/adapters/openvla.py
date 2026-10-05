"""OpenVLA target-token teacher forcing and decoded-action queries.

Targets are in the checkpoint's unnormalized action coordinates. Tokenization
normalizes them using that checkpoint's statistics, matching predict_action.
"""

from pathlib import Path
import json
import numpy as np
import torch


class OpenVLAAdapter:
    def __init__(self, checkpoint, unnorm_key=None, device="cuda", dtype="bfloat16",
                 local_files_only=False, trust_remote_code=False, center_crop_area=1.0,
                 cache_dir=None, revision=None, code_revision=None):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        self.device, self.dtype = torch.device(device), getattr(torch, dtype)
        self.model = AutoModelForVision2Seq.from_pretrained(
            checkpoint, torch_dtype=self.dtype, attn_implementation="eager",
            low_cpu_mem_usage=True, local_files_only=local_files_only,
            trust_remote_code=trust_remote_code, cache_dir=cache_dir, revision=revision,
            code_revision=code_revision).to(self.device).eval().requires_grad_(False)
        if hasattr(self.model, "vision_backbone_requires_grad"):
            self.model.vision_backbone_requires_grad = True
        self.processor = AutoProcessor.from_pretrained(
            checkpoint, local_files_only=local_files_only, trust_remote_code=trust_remote_code,
            cache_dir=cache_dir, revision=revision, code_revision=code_revision)
        stats_path = Path(checkpoint) / "dataset_statistics.json"
        if stats_path.is_file():
            self.model.norm_stats = json.loads(stats_path.read_text())
        self.unnorm_key = unnorm_key
        self.stats = self.model.get_action_stats(unnorm_key)
        self.action_dim = self.model.get_action_dim(unnorm_key)
        self.center_crop_area = center_crop_area
        if not 0 < center_crop_area <= 1:
            raise ValueError("center_crop_area must be in (0,1]")

    def _pixels(self, images):
        # Keep preprocessing in torch: PIL/NumPy would sever the white-box gradient.
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as VF

        images = images.to(self.device).float()
        if self.center_crop_area < 1:
            h, w = images.shape[-2:]
            side = self.center_crop_area ** .5
            images = VF.center_crop(images, [round(h*side), round(w*side)])
        processor = self.processor.image_processor
        if processor.tvf_do_letterbox:
            h, w = images.shape[-2:]
            px, py = (max(h,w)-w)//2, (max(h,w)-h)//2
            if px or py:
                fill = images.new_tensor(processor.tvf_letterbox_fill).view(1, 3, 1, 1)/255
                images = torch.nn.functional.pad(images-fill, (px, px, py, py)) + fill
        modes = {0: InterpolationMode.NEAREST, 2: InterpolationMode.BILINEAR,
                 3: InterpolationMode.BICUBIC}
        outputs = []
        for resize, crop, normalize in zip(processor.tvf_resize_params, processor.tvf_crop_params,
                                           processor.tvf_normalize_params):
            params = dict(resize)
            interpolation = params.pop("interpolation")
            if interpolation not in modes:
                raise ValueError(f"Unsupported differentiable resize interpolation: {interpolation}")
            resized = VF.resize(images, interpolation=modes[interpolation], **params).clamp(0, 1)
            outputs.append(VF.normalize(VF.center_crop(resized, **crop), **normalize))
        return torch.cat(outputs, dim=1).to(self.dtype)

    def _prompt_ids(self, instruction):
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
        ids = self.processor.tokenizer(prompt, add_special_tokens=True).input_ids
        if ids[-1] != 29871:
            ids.append(29871)
        return ids

    def _target_ids(self, target):
        target = target.detach().cpu().numpy()
        if target.shape != (self.action_dim,):
            raise ValueError(f"OpenVLA requires a single target action of size {self.action_dim}")
        low, high = np.asarray(self.stats["q01"]), np.asarray(self.stats["q99"])
        mask = np.asarray(self.stats.get("mask", np.ones_like(low, dtype=bool)))
        span = high-low
        if np.any(mask & (span <= 0)):
            raise ValueError("Invalid action normalization statistics")
        normalized = np.where(mask, 2*(target-low)/np.where(mask, span, 1)-1, target)
        if np.any((normalized < -1) | (normalized > 1)):
            raise ValueError("Target is outside checkpoint action bounds; specify an achievable target")
        bins = np.asarray(self.model.bins)
        return torch.as_tensor(self.model.vocab_size - np.digitize(normalized, bins),
                               dtype=torch.long, device=self.device)

    def target_logits(self, images, instructions, metadata, target, weights):
        del metadata
        target_ids = self._target_ids(target)
        prompts = [self._prompt_ids(text) for text in instructions]
        # Feed the target prefix only. Each selected output predicts the next target token.
        sequences = [p + target_ids[:-1].tolist() for p in prompts]
        length = max(map(len, sequences))
        ids = torch.full((len(sequences), length), self.processor.tokenizer.pad_token_id,
                         dtype=torch.long, device=self.device)
        attention = torch.zeros_like(ids)
        for index, sequence in enumerate(sequences):
            ids[index, :len(sequence)] = torch.tensor(sequence, device=self.device)
            attention[index, :len(sequence)] = 1
        output = self.model(input_ids=ids, attention_mask=attention, pixel_values=self._pixels(images),
                            use_cache=False, return_dict=True)
        visual_tokens = output.logits.shape[1] - ids.shape[1]
        if visual_tokens <= 0:
            raise ValueError("Unexpected OpenVLA visual token layout")
        positions = torch.tensor([len(p)-1+visual_tokens for p in prompts], device=self.device)
        positions = positions[:, None] + torch.arange(self.action_dim, device=self.device)[None]
        logits = output.logits[torch.arange(len(images), device=self.device)[:, None], positions]
        return (logits, target_ids[None].expand(len(images), -1),
                weights.to(self.device)[None].expand(len(images), -1))

    @torch.no_grad()
    def predict_actions(self, images, instructions, metadata):
        del metadata
        pixels = self._pixels(images)
        actions = []
        # The original OpenVLA generation implementation supports batch size one.
        for index, instruction in enumerate(instructions):
            ids = torch.tensor([self._prompt_ids(instruction)], device=self.device)
            action = self.model.predict_action(input_ids=ids, attention_mask=torch.ones_like(ids),
                                               pixel_values=pixels[index:index+1],
                                               unnorm_key=self.unnorm_key, do_sample=False)
            actions.append(np.asarray(action))
        return torch.as_tensor(np.stack(actions), device=images.device, dtype=torch.float32)
