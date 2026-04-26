import torch
import torch.nn.functional as F
import random

# ---------------- Loss & Aggregation ----------------

def softmin_stack(tensors, temperature=0.5):
    if not tensors: 
        return torch.tensor(0., device=torch.device('cpu'))
    T = torch.stack(tensors)  # (K,)
    weights = torch.softmax(T / temperature, dim=0)
    return (weights * T).sum()

def tv_loss(x, mask=None):
    """Total variation on x (B,C,H,W). Weighted within mask if provided."""
    dx = x[:,:,1:,:] - x[:,:,:-1,:]
    dy = x[:,:,:,1:] - x[:,:,:,:-1]
    if mask is not None:
        m_x = mask[:,:,1:,:] * mask[:,:,:-1,:]
        m_y = mask[:,:,:,1:] * mask[:,:,:,:-1]
        dx = dx * m_x
        dy = dy * m_y
    return (dx.abs().mean() + dy.abs().mean())

def check_finite(tensors):
    """Utility to prevent NaNs from crashing the training loop."""
    return all(torch.isfinite(t).all() for t in tensors)

# ---------------- Image Quality Metrics ----------------

def psnr(x, y, data_range=2.0):
    mse = F.mse_loss(x, y)
    if mse.item() == 0:
        return torch.tensor(99.0, device=x.device)
    return 10.0 * torch.log10((data_range ** 2) / mse)

def ssim_approx(x, y, C1=0.01**2, C2=0.03**2):
    x = (x * 0.5 + 0.5).clamp(0,1)
    y = (y * 0.5 + 0.5).clamp(0,1)

    mu_x = x.mean(dim=[2,3], keepdim=True)
    mu_y = y.mean(dim=[2,3], keepdim=True)
    var_x = ((x - mu_x)**2).mean(dim=[2,3], keepdim=True)
    var_y = ((y - mu_y)**2).mean(dim=[2,3], keepdim=True)
    cov_xy = ((x - mu_x)*(y - mu_y)).mean(dim=[2,3], keepdim=True)
    ssim = ((2*mu_x*mu_y + C1)*(2*cov_xy + C2)) / ((mu_x**2 + mu_y**2 + C1)*(var_x + var_y + C2))
    return ssim.mean()

# ---------------- EOT Augmentations ----------------

def _clamp_unit(x):
    return x.clamp(-1.0, 1.0)

def _random_brightness_contrast_gamma(x, delta=0.1):
    if delta <= 0: return x
    b = (torch.rand(1, device=x.device) * 2*delta - delta)
    c = (torch.rand(1, device=x.device) * 2*delta - delta)
    g = 1.0 + (torch.rand(1, device=x.device) * 2*delta - delta)
    y = (x + 1)/2
    y = (y + b).clamp(0,1)
    y = ((y - 0.5)*(1+c) + 0.5).clamp(0,1)
    y = y.pow(g.clamp(0.7, 1.3))
    return _clamp_unit(y*2 - 1)

def _gaussian_blur(x, sigma=0.8):
    if sigma <= 0: return x
    radius = max(1, int(3*sigma))
    k = torch.arange(-radius, radius+1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-0.5*(k**2)/(sigma**2))
    kernel = kernel / kernel.sum()
    kernel_x = kernel.view(1,1,-1,1)
    kernel_y = kernel.view(1,1,1,-1)
    C = x.shape[1]
    x = F.conv2d(x, kernel_x.expand(C,1,-1,1), padding=(radius,0), groups=C)
    x = F.conv2d(x, kernel_y.expand(C,1,1,-1), padding=(0,radius), groups=C)
    return x

def _resize_center_scale(x, scale_min=0.98, scale_max=1.02):
    if scale_min == 1.0 and scale_max == 1.0: return x
    B, C, H, W = x.shape
    s = float(scale_min + (scale_max - scale_min) * random.random())
    newH, newW = int(H*s), int(W*s)
    y = F.interpolate(x, size=(newH, newW), mode="bilinear", align_corners=False)
    pad_h = max(0, H - newH); pad_w = max(0, W - newW)
    if pad_h > 0 or pad_w > 0:
        y = F.pad(y, (pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2))
    y = y[:, :, :H, :W]
    return y

def eot_augs(x, cfg, num_augs=2):
    outs = [x]
    for _ in range(max(0, num_augs-1)):
        y = x
        y = _resize_center_scale(y, cfg.eot_scale_min, cfg.eot_scale_max)
        if cfg.eot_color_delta > 0: y = _random_brightness_contrast_gamma(y, cfg.eot_color_delta)
        if cfg.eot_blur_sigma > 0:  y = _gaussian_blur(y, cfg.eot_blur_sigma)
        if cfg.eot_noise_std > 0:   y = _clamp_unit(y + torch.randn_like(y)*cfg.eot_noise_std)
        y = torch.nan_to_num(y, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1, 1)
        outs.append(y)
    return outs

# ---------------- Scheduling & Targets ----------------

def compute_asr(fr_name: str, cos_list, thresholds_dict):
    th01, th001, th0001 = thresholds_dict[fr_name]
    total = max(1, len(cos_list))
    s01   = sum(c > th01   for c in cos_list) / total
    s001  = sum(c > th001  for c in cos_list) / total
    s0001 = sum(c > th0001 for c in cos_list) / total
    return {"FAR@0.1": s01, "FAR@0.01": s001, "FAR@0.001": s0001}

def step_fr_weight(step_idx, total_steps, base=0.2, gamma=2.0):
    if total_steps <= 1: return 1.0
    frac = step_idx / (total_steps - 1)
    return base + (1.0 - base) * (frac ** gamma)

def get_hinge_threshold(fr_name, thresholds_dict, far_level=0.01):
    a01, a001, a0001 = thresholds_dict[fr_name]
    if abs(far_level - 0.1) < 1e-9:  return a01
    if abs(far_level - 0.01) < 1e-9: return a001
    if abs(far_level - 0.001) < 1e-9:return a0001
    return a001

