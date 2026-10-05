"""Appendix-D local metrics and explicit, additional appearance acceptance limits.

The paper uses diffusion/anchor regularization, not these numerical limits.
Passing this gate is not a proof of perceptual realism; inspect the preview too.
"""
import math
import torch
import torch.nn.functional as F
from .quality import local_crop, patch_tv, boundary_seam_energy


def ssim(images, reference):
    """Per-image RGB SSIM: 11x11 Gaussian (sigma 1.5), valid windows, range 1."""
    if images.shape != reference.shape or images.ndim != 4 or min(images.shape[-2:]) < 11:
        raise ValueError('SSIM needs matching BCHW tensors at least 11x11')
    x = torch.arange(11,device=images.device,dtype=images.dtype)-5
    g = torch.exp(-x.square()/(2*1.5**2)); g = g/g.sum()
    kernel = (g[:,None]*g[None,:])[None,None].expand(images.shape[1],1,11,11)
    def avg(v): return F.conv2d(v,kernel,groups=images.shape[1])
    mu_a,mu_b = avg(images),avg(reference)
    var_a = (avg(images.square())-mu_a.square()).clamp_min(0)
    var_b = (avg(reference.square())-mu_b.square()).clamp_min(0)
    cov = avg(images*reference)-mu_a*mu_b
    score = ((2*mu_a*mu_b+.01**2)*(2*cov+.03**2))/((mu_a.square()+mu_b.square()+.01**2)*(var_a+var_b+.03**2))
    return score.mean((1,2,3))


FIXED_LIMITS = {'patch_rgb_rmse_max': .06, 'patch_tv_max': .04,
                'boundary_increase_max': 5., 'local_ssim_drop_max': .01}


def validate_fixed_limits(limits):
    """Animal-seed experiments must retain the approved numerical limits."""
    if limits != FIXED_LIMITS:
        raise ValueError(f'Naturalness limits changed; required values: {FIXED_LIMITS}')


def check_limits(metrics, limits):
    """Fail closed: every configured metric must exist and be finite."""
    expected = {'patch_rgb_rmse_max','patch_tv_max','boundary_increase_max','local_ssim_drop_max'}
    if set(limits) != expected:
        raise ValueError(f'Expected exactly these naturalness limits: {sorted(expected)}')
    violations = {}
    for key,limit in limits.items():
        if not math.isfinite(limit) or limit < 0:
            raise ValueError(f'Invalid naturalness limit: {key}')
        value = metrics[key]
        if not math.isfinite(value) or value > limit:
            violations[key] = {'value':value,'maximum':limit}
    return {'accepted':not violations,'violations':violations,'limits':dict(limits),
            'interpretation':'Additional measurable safeguards, not paper thresholds or a realism guarantee'}


@torch.no_grad()
def appearance_report(images, candidate, seed, renderer, metadata, box, limits):
    if any('quad' in row or ('box' in row and list(row['box']) != list(box)) for row in metadata):
        raise ValueError('Appearance gate currently requires the fixed configured rectangle')
    if any(int(v)!=v for v in box):
        raise ValueError('Appearance gate needs integer patch boundaries')
    x,y,w,h = map(int,box)
    rendered = renderer(images,candidate,metadata)[0]
    reference = renderer(images,seed,metadata)[0]
    region = rendered[:,:,y:y+h,x:x+w]
    reference_region = reference[:,:,y:y+h,x:x+w]
    if region.shape[-2:] != (h,w):
        raise ValueError('Patch must be entirely inside the scene')
    # Paper reference is the unmodified scene, not the seed-composited scene.
    similarity = ssim(local_crop(rendered),local_crop(images))
    seed_similarity = ssim(local_crop(reference),local_crop(images))
    seams = boundary_seam_energy(rendered,box)
    seed_seams = boundary_seam_energy(reference,box)
    tv = patch_tv(region)
    rmse = (region-reference_region).square().mean((1,2,3)).sqrt()
    metrics = {'patch_rgb_rmse_max':float(rmse.max()), 'patch_tv_max':float(tv.max()),
               'boundary_increase_max':float((seams-seed_seams).clamp_min(0).max()),
               'local_ssim_drop_max':float((seed_similarity-similarity).clamp_min(0).max())}
    values = {'ssim_to_clean':similarity,'boundary_seam_cie76':seams,'patch_tv':tv,
              'patch_rgb_rmse_to_seed':rmse}
    return {'frames':len(images),'crop_xywh':[0,32,192,192], 'patch_footprint_xywh':list(box),
            'statistics':{key:{'mean':float(v.mean()),'std':float(v.std(unbiased=False)),
                               'per_frame':v.cpu().tolist()} for key,v in values.items()},
            'gate_metrics':metrics, 'gate':check_limits(metrics,limits),
            'not_computed':['NIQE','DISTS','CLIP-Natural'],
            'note':'Patch TV is measured at the rendered footprint, not at the 512x512 VAE resolution.'}


def eligible_best(candidates):
    eligible = [c for c in candidates if c['appearance']['gate']['accepted'] and c.get('action_score')]
    if not eligible:
        return None
    return min(eligible,key=lambda c:(c['action_score']['target_weighted_mse'],c['action_score']['target_ce']))
