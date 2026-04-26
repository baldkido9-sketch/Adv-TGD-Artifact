import os, copy, random, json, cv2, shutil, re, glob, argparse
os.environ["HF_HUB_OFFLINE"] = '0'

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import open_clip

# --- Diffusers & HuggingFace ---
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from transformers import CLIPTokenizer, CLIPTextModel
from peft import LoraConfig, get_peft_model

# --- Local Modules ---
from config import AdvTGDConfig, THRESHOLDS, FACENET_TAU_1E3
from face_processor import FaceProcessor, _top_percent_soft_mask, _ellipse_prior, _fr_target_saliency_from_src, \
    _normalize01, save_sgsm_visualization
from losses import softmin_stack, tv_loss, eot_augs, check_finite, psnr, ssim_approx, get_hinge_threshold
from dataset_userT import SimpleSrcTgtDataset
from utils import get_fr_model

try:
    import face_alignment
except ImportError:
    print("Please run 'pip install face_alignment' to use the face alignment feature.")
    face_alignment = None

try:
    import lpips
except ImportError:
    lpips = None
    print("[INFO] 'lpips' not installed. LPIPS loss will be disabled.")

# ---------------- Globals & Simple Helpers ----------------
clip_model = None
clip_tokenizer = None


def ensure_dir(p): os.makedirs(p, exist_ok=True)


def get_scaling_factor(vae):
    return getattr(vae.config, "scaling_factor", 0.18215)


def tensor_to_pil(x_bchw):
    x = torch.nan_to_num(x_bchw[0], nan=0.0, posinf=1.0, neginf=-1.0)
    x = (x * 0.5 + 0.5).clamp(0, 1)
    arr = (x.permute(1, 2, 0).detach().cpu().numpy() * 255).astype("uint8")
    return Image.fromarray(arr)


def load_image_tensor(image_input, device: torch.device, size):
    if isinstance(image_input, str):
        if not os.path.exists(image_input): return torch.empty(0)
        img = Image.open(image_input).convert("RGB")
    else:
        img = image_input
    tfm = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    return tfm(img).unsqueeze(0).to(device)


def fr_size_from_name(name):
    if name in ("IR152", "IRSE50", "MobileFace"): return 112
    return 160


def prep_fr(x, size):
    return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)


def step_fr_weight(step_idx, total_steps, base=0.2, gamma=2.0):
    if total_steps <= 1: return 1.0
    frac = step_idx / (total_steps - 1)
    return base + (1.0 - base) * (frac ** gamma)


def prep_clip_masked_crop(in_img_bchw, src_img_bchw, soft_img_mask_bchw, target=224):
    assert in_img_bchw.shape[0] == 1, "expects batch=1 here"
    m = soft_img_mask_bchw.clamp(0, 1)
    comp = m * in_img_bchw + (1 - m) * src_img_bchw.detach()
    with torch.no_grad():
        mh = (m[0, 0] > 0.3)
        if mh.any():
            ys, xs = torch.where(mh.any(dim=1))[0], torch.where(mh.any(dim=0))[0]
            y0, y1 = int(ys[0].item()), int(ys[-1].item()) + 1
            x0, x1 = int(xs[0].item()), int(xs[-1].item()) + 1
        else:
            H, W = m.shape[-2], m.shape[-1]
            side = int(min(H, W) * 0.8)
            y0, y1 = (H - side) // 2, (H - side) // 2 + side
            x0, x1 = (W - side) // 2, (W - side) // 2 + side

    comp_crop = comp[:, :, y0:y1, x0:x1]
    comp_224 = F.interpolate(comp_crop, size=(target, target), mode="bilinear", align_corners=False)
    x01 = (comp_224 * 0.5 + 0.5).clamp(0, 1)

    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x01.device).view(1, 3, 1, 1)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x01.device).view(1, 3, 1, 1)
    return (x01 - mean) / std


# ---------------- Primary Training Loop ----------------
def train_one_pair(spec, base_out_dir, fr_ensemble, tokenizer, text_encoder,
                   vae, unet_base, noise_scheduler, alphas_cumprod, fa_model,
                   cfg: AdvTGDConfig):
    nan_stats = {"terms": 0, "lpips": 0, "backward": 0, "grads": 0, "skipped_steps": 0, "max_consec": 0}
    consec_skips, TEMP_DISABLE_LPIPS_STEPS = 0, 0

    out_dir_suffix = spec['out_id']
    out_dir = os.path.join(base_out_dir, out_dir_suffix)
    ensure_dir(out_dir)

    processor = FaceProcessor(fa_model, cfg)

    # ---- LoRA Setup ----
    lora_cfg = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, target_modules=cfg.lora_targets,
                          lora_dropout=cfg.lora_dropout, bias="none")
    unet = get_peft_model(copy.deepcopy(unet_base), lora_cfg).to(cfg.device)
    unet.train()
    trainable = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=cfg.lr)

    final_factor = float(cfg.final_lr) / float(cfg.lr)

    def _lr_lambda(step_idx):
        if step_idx <= cfg.decay_start_step: return 1.0
        tail = max(1, cfg.num_steps - cfg.decay_start_step)
        return 1.0 + (final_factor - 1.0) * min(1.0, (step_idx - cfg.decay_start_step) / tail)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)

    # ---- Image Alignment ----
    img_pil_src, M_src = processor.align_face(spec['src_path'])
    img_pil_tgt, M_tgt = processor.align_face(spec['tgt_path'])
    if img_pil_src is None or img_pil_tgt is None:
        print(f"⚠️ Skipping pair {out_dir_suffix} due to alignment failure.")
        return

    img_pil_src.save(os.path.join(out_dir, "src_aligned.png"))
    img_pil_tgt.save(os.path.join(out_dir, "tgt_aligned.png"))

    src_img = processor.tfm(img_pil_src).unsqueeze(0).to(cfg.device)
    tgt_img = processor.tfm(img_pil_tgt).unsqueeze(0).to(cfg.device)

    # ---- Masking Generation (SGSM logic) ----
    latent_mask, soft_img_mask = None, None
    if cfg.use_masking:
        if cfg.use_hybrid_mask:
            try:
                mask_parsing, contour = processor.get_landmark_mask(img_pil_src, expand_forehead=0.6)
                if mask_parsing is None: mask_parsing = torch.zeros(1, 1, cfg.resolution, cfg.resolution).to(cfg.device)

                with torch.no_grad():
                    e_tgt_dict = {
                        fr["name"]: F.normalize(fr["model"](prep_fr(tgt_img.to(torch.float32), fr["size"])), p=2, dim=1)
                        for fr in fr_ensemble}

                sal_raw = _fr_target_saliency_from_src(src_img, fr_ensemble, e_tgt_dict, prep_fr)
                prior = _ellipse_prior(cfg.resolution, cfg.resolution, rx_frac=cfg.ellipse_rx_frac,
                                       ry_frac=cfg.ellipse_ry_frac, device=sal_raw.device)

                gated_raw = cfg.prior_strength * (sal_raw * prior) + (1.0 - cfg.prior_strength) * sal_raw
                is_flat = (gated_raw.amax(dim=(2, 3)) - gated_raw.amin(dim=(2, 3))) < 1e-6
                gated = torch.where(is_flat, gated_raw, _normalize01(gated_raw))

                k_big = 21
                ker_big = torch.ones(1, 1, k_big, k_big, device=gated.device) / (k_big * k_big)
                gated = F.conv2d(gated, ker_big, padding=k_big // 2)

                mask_gradient = _top_percent_soft_mask(gated, keep_percent=cfg.keep_percent_early,
                                                       feather_k=cfg.feather_k, prior=prior)
                combined_mask = torch.maximum(mask_parsing, mask_gradient)
                soft_img_mask = processor.clean_and_feather_mask(combined_mask, dilate_ksize=15, blur_ksize=21)

                src_cv = ((src_img[0].permute(1, 2, 0).detach().cpu().numpy() * 0.5 + 0.5) * 255).astype(np.uint8)
                src_cv = cv2.cvtColor(src_cv, cv2.COLOR_RGB2BGR)
                sal_np = gated[0, 0].detach().cpu().numpy()
                final_mask_np = soft_img_mask[0, 0].detach().cpu().numpy()
                save_sgsm_visualization(src_cv, sal_np, contour, final_mask_np,
                                        os.path.join(out_dir, "showcase_SGSM_comparison.png"))

                transforms.ToPILImage()(soft_img_mask.squeeze(0).cpu()).save(os.path.join(out_dir, "mask_hybrid.png"))
                latent_mask = F.interpolate(soft_img_mask, size=(cfg.resolution // 8, cfg.resolution // 8),
                                            mode='area').clamp(0, 1)
            except Exception as e:
                print(f"[WARN] Hybrid mask failed: {e}")

        elif cfg.use_parsing_mask:
            try:
                soft_img_mask, _ = processor.get_landmark_mask(img_pil_src, expand_forehead=0.6)
                if soft_img_mask is not None:
                    latent_size = cfg.resolution // 8
                    latent_mask = F.interpolate(soft_img_mask, size=(latent_size, latent_size), mode='area').clamp(0, 1)
                    transforms.ToPILImage()(soft_img_mask.squeeze(0).cpu()).save(
                        os.path.join(out_dir, "mask_parsing.png"))
            except Exception as e:
                print(f"[ERR] Parsing mask failed: {e}")

        elif cfg.use_smart_masks:
            try:
                with torch.no_grad():
                    e_tgt_dict = {
                        fr["name"]: F.normalize(fr["model"](prep_fr(tgt_img.to(torch.float32), fr["size"])), p=2, dim=1)
                        for fr in fr_ensemble}
                sal_raw = _fr_target_saliency_from_src(src_img, fr_ensemble, e_tgt_dict, prep_fr)
                prior = _ellipse_prior(cfg.resolution, cfg.resolution, rx_frac=cfg.ellipse_rx_frac,
                                       ry_frac=cfg.ellipse_ry_frac, device=sal_raw.device)
                gated_raw = cfg.prior_strength * (sal_raw * prior) + (1.0 - cfg.prior_strength) * sal_raw
                is_flat = (gated_raw.amax(dim=(2, 3)) - gated_raw.amin(dim=(2, 3))) < 1e-6
                gated = torch.where(is_flat, gated_raw, _normalize01(gated_raw))

                k_big = 21
                ker_big = torch.ones(1, 1, k_big, k_big, device=gated.device) / (k_big * k_big)
                gated = F.conv2d(gated, ker_big, padding=k_big // 2)

                soft_img_mask = _top_percent_soft_mask(gated, keep_percent=cfg.keep_percent_early,
                                                       feather_k=cfg.feather_k, prior=prior)
                soft_img_mask = processor.clean_and_feather_mask(soft_img_mask, dilate_ksize=25, blur_ksize=31)

                latent_size = cfg.resolution // 8
                latent_mask = F.interpolate(soft_img_mask, size=(latent_size, latent_size), mode='area').clamp(0, 1)

                transforms.ToPILImage()(soft_img_mask.squeeze(0).cpu()).save(os.path.join(out_dir, "mask_soft.png"))
            except Exception as e:
                print(f"[WARN] Smart mask failed: {e}")
    else:
        print("[INFO] USE_MASKING is False -> training with full-image edits (no regional mask).")

    # ---- Latents & text ----
    sf = get_scaling_factor(vae)
    src_lat = vae.encode(src_img.to(vae.dtype)).latent_dist.sample().detach() * sf
    with torch.no_grad():
        text_inputs = tokenizer([spec['text'] or "portrait photo"], padding="max_length", truncation=True,
                                max_length=tokenizer.model_max_length, return_tensors="pt")
        text_embeds = text_encoder(text_inputs.input_ids.to(cfg.device))[0]
        if clip_model is not None and clip_tokenizer is not None:
            t_tok = clip_tokenizer(spec.get("text") or "portrait face").to(cfg.device)
            clip_text_feat = F.normalize(clip_model.encode_text(t_tok).float(), p=2, dim=-1)
        else:
            clip_text_feat = None

    precomp = []
    with torch.no_grad():
        for fr in fr_ensemble:
            ssz = fr["size"]
            precomp.append({"name": fr["name"], "tau": fr["tau"], "size": ssz, "model": fr["model"],
                            "e_src": F.normalize(fr["model"](prep_fr(src_img.to(torch.float32), ssz)), p=2, dim=1),
                            "e_tgt": F.normalize(fr["model"](prep_fr(tgt_img.to(torch.float32), ssz)), p=2, dim=1)})

    lpips_fn = lpips.LPIPS(net='vgg').to(cfg.device).eval() if lpips is not None and cfg.lambda_lpips > 0 else None
    t_enc = max(1, int(cfg.strength_train * noise_scheduler.config.num_train_timesteps))
    t_fixed = max(0, min(t_enc - 1, int(cfg.t_fixed_frac * t_enc))) if cfg.use_fixed_t else None

    stall_triggers, last_logged_cos_tgt = 0, -1.0
    step_history = []
    id_gain, dir_gain, src_gain = 1.0, 1.0, 1.0

    pbar = tqdm(range(cfg.num_steps), desc=f"Training '{out_dir_suffix}'")
    for step in pbar:
        frac = step / max(1, (cfg.num_steps - 1))
        warmup = (frac <= cfg.warmup_frac)

        w_eps = cfg.lambda_eps
        w_id = 0.35 if warmup else (cfg.lambda_id_hinge * id_gain)
        w_dir = (cfg.lambda_dir_warm if warmup else cfg.lambda_dir_main) * dir_gain
        w_src = 0.15 if warmup else (cfg.lambda_src / (src_gain * 2.0))
        w_cons, w_tv = (0.0, 0.0) if warmup else (cfg.lambda_cons, cfg.lambda_tv)
        w_lpips = 0.0 if lpips_fn is None or warmup else cfg.lambda_lpips
        w_bg = cfg.lambda_bg_mse

        src_margin_dynamic = cfg.src_margin
        if not warmup and frac > 0.60:
            w_bg *= 0.5
            src_margin_dynamic *= 1.4

        if TEMP_DISABLE_LPIPS_STEPS > 0:
            w_lpips = 0.0
            TEMP_DISABLE_LPIPS_STEPS -= 1

        t = torch.full((src_lat.shape[0],), t_fixed, device=cfg.device,
                       dtype=torch.long) if cfg.use_fixed_t else torch.randint(0, t_enc, (src_lat.shape[0],),
                                                                               device=cfg.device).long()
        noise = torch.randn_like(src_lat)
        noisy_lat = noise_scheduler.add_noise(src_lat, noise, t)

        pred = unet(noisy_lat, t, encoder_hidden_states=text_embeds).sample

        a_t = alphas_cumprod[t].view(-1, 1, 1, 1)
        x0_pred = (noisy_lat - (1 - a_t).sqrt() * pred) / a_t.sqrt()
        x0_pred_blended = x0_pred if latent_mask is None else (src_lat * (1 - latent_mask) + x0_pred * latent_mask)

        rec_for_fr = torch.nan_to_num(vae.decode(x0_pred_blended / sf).sample.to(torch.float32), nan=0.0).clamp(-1, 1)
        augs = eot_augs(rec_for_fr, cfg, num_augs=1 if warmup else (2 if frac > 2 / 3 else 1))

        chosen = precomp if cfg.ensemble_sample_k >= len(precomp) else random.sample(precomp, cfg.ensemble_sample_k)

        hinge_terms, dir_terms, cons_terms, cos_logs, cos_step_dict = [], [], [], [], {}

        for frk in chosen:
            e_preds = [F.normalize(frk["model"](prep_fr(aug, frk["size"])), p=2, dim=1, eps=1e-12) for aug in augs]
            cos_tgt_k = (e_preds[0] * frk["e_tgt"]).sum(dim=1)
            cos_mean_k = cos_tgt_k.mean()

            tau_k = frk["tau"]
            delta_to_fn = max(0.0, FACENET_TAU_1E3 - tau_k)
            base_m = cfg.base_margin_warm if warmup else cfg.base_margin_main
            margin = min(base_m + cfg.alpha_margin * delta_to_fn + cfg.blend_comp + (0.0 if warmup else 0.08 * frac),
                         cfg.margin_cap_warm if warmup else cfg.margin_cap_main)

            is_weakest = True if not cos_logs else (
                        torch.stack(cos_logs + [cos_mean_k.detach()]).argmin().item() == len(cos_logs))

            target_k = tau_k + margin
            hinge_core = torch.relu(target_k - cos_tgt_k)
            overshoot = torch.relu(target_k + 0.03 - cos_tgt_k) * (0.4 if is_weakest else 0.2)
            hinge_terms.append((hinge_core + overshoot).mean())

            cos_logs.append(cos_mean_k.detach())
            cos_step_dict[frk["name"]] = cos_mean_k.item()

            v_pred = F.normalize(e_preds[0] - frk["e_src"], p=2, dim=1)
            v_tgt = F.normalize(frk["e_tgt"] - frk["e_src"], p=2, dim=1)
            dir_terms.append((1.0 - (v_pred * v_tgt).sum(dim=1)).mean())

            if len(e_preds) > 1:
                base = e_preds[0].detach()
                cons_accum = sum(
                    (1.0 - (F.normalize(base, p=2, dim=1) * F.normalize(e_preds[j], p=2, dim=1)).sum(dim=1).mean()) for
                    j in range(1, len(e_preds)))
                cons_terms.append(cons_accum / (len(e_preds) - 1))

        L_id_hinge = softmin_stack(hinge_terms, temperature=0.1)
        L_dir = softmin_stack(dir_terms, temperature=0.1)
        L_cons = torch.stack(cons_terms).mean() if cons_terms else torch.tensor(0.0, device=cfg.device)
        avg_cos_tgt = torch.stack(cos_logs).mean() if cos_logs else torch.tensor(0.0, device=cfg.device)

        src_terms = [torch.relu(
            (F.normalize(frk["model"](prep_fr(augs[0], frk["size"])), p=2, dim=1) * frk["e_src"]).sum(
                dim=1) - src_margin_dynamic).mean() for frk in chosen]
        L_src = torch.stack(src_terms).mean() if src_terms else torch.tensor(0.0, device=cfg.device)

        if latent_mask is not None:
            L_eps = (F.mse_loss(pred, noise, reduction='none') * latent_mask).sum() / (
                        latent_mask.sum() * pred.shape[1] + 1e-8)
        else:
            L_eps = F.mse_loss(pred, noise)

        rec_vis = torch.nan_to_num(vae.decode(x0_pred_blended / sf).sample.to(torch.float32), nan=0.0).detach()
        in_img = augs[0]
        L_bg_mse = F.mse_loss(in_img * (1.0 - soft_img_mask).expand_as(src_img).detach(),
                              src_img * (1.0 - soft_img_mask).expand_as(
                                  src_img).detach()) if soft_img_mask is not None and w_bg > 0 else torch.tensor(0.0,
                                                                                                                 device=cfg.device)

        run_lpips_now = (not warmup) and (w_lpips > 0) and (lpips_fn is not None) and (step % cfg.lpips_every == 0)
        if run_lpips_now:
            in_m_small = F.interpolate((in_img * soft_img_mask.expand_as(in_img).detach()).clamp(-1, 1),
                                       size=(cfg.lpips_side, cfg.lpips_side),
                                       mode="area") if soft_img_mask is not None else in_img
            src_m_small = F.interpolate((src_img * soft_img_mask.expand_as(src_img).detach()).clamp(-1, 1),
                                        size=(cfg.lpips_side, cfg.lpips_side),
                                        mode="area") if soft_img_mask is not None else src_img
            try:
                with torch.autocast(device_type=('cuda' if in_m_small.is_cuda else 'cpu'), enabled=False):
                    lp = lpips_fn(in_m_small.to(torch.float32), src_m_small.to(torch.float32))
                L_lpips = lp.mean()
                if not torch.isfinite(L_lpips): raise FloatingPointError()
            except Exception:
                nan_stats["lpips"] += 1
                nan_stats["skipped_steps"] += 1
                consec_skips += 1
                TEMP_DISABLE_LPIPS_STEPS = max(TEMP_DISABLE_LPIPS_STEPS, 10)
                optimizer.zero_grad(set_to_none=True)
                continue
        else:
            L_lpips = torch.tensor(0.0, device=cfg.device)

        L_tv = tv_loss(in_img, mask=soft_img_mask) if soft_img_mask is not None and w_tv > 0 else torch.tensor(0.0,
                                                                                                               device=cfg.device)

        late_phase = (frac >= cfg.text_late_start_frac)
        w_text_phase = cfg.w_text_late if late_phase else cfg.w_text_early
        use_txt_step = (late_phase or not warmup) and (w_text_phase > 0) and (soft_img_mask is not None) and (
                    clip_text_feat is not None)

        if use_txt_step:
            s_mask_late = torch.maximum(soft_img_mask, torch.clamp(
                F.conv2d(soft_img_mask, torch.ones(1, 1, 9, 9, device=cfg.device) / 81.0, padding=4) * 1.2, 0,
                1)) if late_phase else soft_img_mask
            pixel_values = prep_clip_masked_crop(augs[0], src_img, s_mask_late)
            L_txt = (1.0 - (
                        F.normalize(clip_model.encode_image(pixel_values).float(), p=2, dim=-1) * clip_text_feat).sum(
                dim=1)).mean()
        else:
            L_txt = torch.tensor(0., device=cfg.device)

        if not check_finite([L_eps, L_id_hinge, L_dir, L_src, L_cons, L_lpips, L_tv, L_bg_mse, L_txt]):
            nan_stats["terms"] += 1
            nan_stats["skipped_steps"] += 1
            consec_skips += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        s_fw = step_fr_weight(step, cfg.num_steps, base=0.85, gamma=1.5)
        loss = (w_eps * L_eps + s_fw * (
                    w_id * L_id_hinge + w_dir * L_dir + w_src * L_src) + w_lpips * L_lpips + w_tv * L_tv + w_bg * L_bg_mse + w_text_phase * L_txt + w_cons * L_cons)

        if not torch.isfinite(loss):
            nan_stats["backward"] += 1
            nan_stats["skipped_steps"] += 1
            consec_skips += 1
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.zero_grad(set_to_none=True)
        try:
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in trainable):
                nan_stats["grads"] += 1
                raise FloatingPointError("non-finite grad")
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            consec_skips = 0
        except Exception:
            nan_stats["backward"] += 1
            nan_stats["skipped_steps"] += 1
            consec_skips += 1
            nan_stats["max_consec"] = max(nan_stats["max_consec"], consec_skips)
            optimizer.zero_grad(set_to_none=True)
            continue

        is_log_step = (step % cfg.log_every == 0)
        if is_log_step and step > 0:
            progress = avg_cos_tgt.item() - last_logged_cos_tgt
            if progress < cfg.progress_threshold and stall_triggers < cfg.stall_max_triggers:
                pbar.write(f"   [!] Stalled @ {step}. Increasing attack strength.")
                stall_triggers += 1
                id_gain = min(id_gain * cfg.stall_up_factor, 5.0)
                dir_gain = min(dir_gain * cfg.stall_up_factor, 1.5)
                src_gain = min(src_gain * cfg.stall_down_src, 10.0)
            last_logged_cos_tgt = avg_cos_tgt.item()
        elif step == 0:
            last_logged_cos_tgt = avg_cos_tgt.item()

        if is_log_step or step + 1 == cfg.num_steps:
            pbar.set_postfix({'Loss': f"{loss.item():.4f}", 'L_id': f"{L_id_hinge.item():.4f}",
                              'cos(avg)': f"{avg_cos_tgt.item():.3f}"})

        if (cfg.save_every > 0 and step % cfg.save_every == 0) or step + 1 == cfg.num_steps:
            tensor_to_pil(rec_vis).save(os.path.join(out_dir, f"step_{step:06d}.png"))


# ---------------- Re-evaluation ----------------
@torch.no_grad()
def reevaluate_one_pair_ensemble(pair_info, fr_ensemble, fa_model, cfg: AdvTGDConfig, topk_blend: int = 50):
    pair_dir, original_src_path, original_tgt_path = pair_info['out_dir'], pair_info['src_path'], pair_info['tgt_path']
    processor = FaceProcessor(fa_model, cfg)

    def _step_num(p):
        m = re.search(r'step_(\d+)\.png', os.path.basename(p))
        return int(m.group(1)) if m else -1

    step_image_paths_all = sorted(glob.glob(os.path.join(pair_dir, 'step_*.png')), key=_step_num)
    if not step_image_paths_all: return {"status": "Error", "cos_tgt_final": {}, "psnr_final": 0.0, "ssim_final": 0.0}

    filtered = [p for p in step_image_paths_all if
                _step_num(p) >= cfg.min_eval_step and (cfg.eval_stride <= 1 or _step_num(p) % cfg.eval_stride == 0)]
    if step_image_paths_all[-1] not in filtered: filtered.append(step_image_paths_all[-1])

    src_tensor = load_image_tensor(original_src_path, cfg.device, cfg.resolution)
    tgt_tensor = load_image_tensor(original_tgt_path, cfg.device, cfg.resolution)

    e_tgt = {fr["name"]: F.normalize(fr["model"](prep_fr(tgt_tensor.to(torch.float32), fr["size"])), p=2, dim=1) for fr
             in fr_ensemble}

    aligned_t_list, aligned_meta_steps = [], []
    for p in filtered:
        aligned_t_list.append(processor.tfm(Image.open(p).convert("RGB")).unsqueeze(0))
        aligned_meta_steps.append(_step_num(p))

    aligned_batch = torch.cat(aligned_t_list, dim=0).to(cfg.device)

    cos_pre_all = {}
    with torch.autocast(device_type='cuda' if cfg.device.type == 'cuda' else 'cpu'):
        for fr in fr_ensemble:
            emb = F.normalize(fr["model"](prep_fr(aligned_batch.to(torch.float32), fr["size"])), p=2, dim=1)
            cos_pre_all[fr["name"]] = (emb * e_tgt[fr["name"]].expand(emb.shape[0], -1)).sum(dim=1).detach()

    eval_name = fr_ensemble[0]["name"]
    K = min(topk_blend, aligned_batch.shape[0])
    topk_idx = torch.topk(cos_pre_all[eval_name], K, largest=True).indices.tolist()

    _, M_src = processor.align_face(original_src_path)

    blended_tensor_cpu_list, blended_meta = [], []
    for batch_i in topk_idx:
        blended_pil = processor.blend_image(filtered[batch_i], original_src_path, M_src)
        if blended_pil is None: continue
        blended_tensor_cpu_list.append(processor.tfm(blended_pil).unsqueeze(0))
        blended_meta.append({"step": aligned_meta_steps[batch_i], "path": filtered[batch_i]})

    if not blended_tensor_cpu_list: return {"status": "Error", "cos_tgt_final": {}, "psnr_final": 0.0,
                                            "ssim_final": 0.0}

    blended_batch = torch.cat(blended_tensor_cpu_list, dim=0).to(cfg.device)
    cos_post_all = {}
    with torch.autocast(device_type='cuda' if cfg.device.type == 'cuda' else 'cpu'):
        for fr in fr_ensemble:
            emb = F.normalize(fr["model"](prep_fr(blended_batch.to(torch.float32), fr["size"])), p=2, dim=1)
            cos_post_all[fr["name"]] = (emb * e_tgt[fr["name"]].expand(emb.shape[0], -1)).sum(dim=1).detach()

    tau = THRESHOLDS[eval_name][2]
    candidates = [{"step": blended_meta[i]["step"], "path": blended_meta[i]["path"],
                   "cos_per_fr": {n: float(cos_post_all[n][i].item()) for n in cos_post_all},
                   "psnr": psnr(blended_batch[i:i + 1], src_tensor).item(),
                   "ssim": ssim_approx(blended_batch[i:i + 1], src_tensor).item()} for i in range(len(blended_meta))]

    passers = [c for c in candidates if c["cos_per_fr"].get(eval_name, -1.0) >= tau]
    best = max(passers, key=lambda r: (r["psnr"], r["cos_per_fr"].get(eval_name, -1.0))) if passers else max(candidates,
                                                                                                             key=lambda
                                                                                                                 r: (r[
                                                                                                                         "cos_per_fr"].get(
                                                                                                                 eval_name,
                                                                                                                 -1.0),
                                                                                                                     r[
                                                                                                                         "psnr"]))

    shutil.copy2(best["path"], os.path.join(pair_dir, "final_best_REEVALUATED.png"))
    final_blended_pil = processor.blend_image(best["path"], original_src_path, M_src)
    if final_blended_pil: final_blended_pil.save(os.path.join(pair_dir, "final_best_blended_REEVALUATED.png"))

    return {"cos_tgt_final": best["cos_per_fr"], "psnr_final": best["psnr"], "ssim_final": best["ssim"],
            "status": "Success"}


# ---------------- Pipeline Execution ----------------
def run_adv_tgd(run_ablation=False):
    print(f"\n[INFO] Starting Adv-TGD Pipeline (Ablation Mode: {run_ablation})...")
    RUN_ID = "ccs_eval"
    base_cfg = AdvTGDConfig()

    experiments = [
        {"name": "SGSM_Proposed", "changes": {}},
        {"name": "Saliency_Mask", "changes": {"use_hybrid_mask": False, "use_smart_masks": True}},
        {"name": "Parsing_Mask", "changes": {"use_hybrid_mask": False, "use_parsing_mask": True}},
        {"name": "Full_Image_Editing", "changes": {"use_hybrid_mask": False, "use_masking": False}}
    ] if run_ablation else [{"name": "Adv_TGD_Main", "changes": {}}]

    print("[INFO] Loading models...")
    global clip_model, clip_tokenizer
    try:
        clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
        clip_model = clip_model.to(base_cfg.device).eval()
        for p in clip_model.parameters(): p.requires_grad = False
        clip_tokenizer = open_clip.get_tokenizer('ViT-B-32')
    except Exception as e:
        print(f"[WARN] OpenCLIP failed: {e}")

    tokenizer = CLIPTokenizer.from_pretrained(base_cfg.model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_cfg.model_id, subfolder="text_encoder").to(
        base_cfg.device).eval().requires_grad_(False)
    vae = AutoencoderKL.from_pretrained(base_cfg.model_id, subfolder="vae").to(base_cfg.device).eval().requires_grad_(
        False)
    unet_base = UNet2DConditionModel.from_pretrained(base_cfg.model_id, subfolder="unet").to(
        base_cfg.device).eval().requires_grad_(False)
    noise_scheduler = DDPMScheduler.from_pretrained(base_cfg.model_id, subfolder="scheduler")
    alphas_cumprod = noise_scheduler.alphas_cumprod.to(base_cfg.device)

    fa_model = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, flip_input=False, device=str(
        base_cfg.device)) if base_cfg.align_faces and face_alignment else None

    TRAIN_ENSEMBLE = [{"name": n, "model": get_fr_model(n, device=base_cfg.device).eval().requires_grad_(False),
                       "size": fr_size_from_name(n),
                       "tau": get_hinge_threshold(n, THRESHOLDS, base_cfg.hinge_far_level)} for n in
                      base_cfg.train_ensemble_frs]
    EVAL_ENSEMBLE = [{"name": n, "model": get_fr_model(n, device=base_cfg.device).eval().requires_grad_(False),
                      "size": fr_size_from_name(n), "tau": get_hinge_threshold(n, THRESHOLDS, base_cfg.hinge_far_level)}
                     for n in base_cfg.eval_frs]

    print("[INFO] Building dataset pairs...")
    manifest_path = "pairs_manifest_updated.json"
    subset_specs = []

    if os.path.exists(manifest_path):
        print(f"[INFO] Reading pairs from {manifest_path}...")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest_data = json.load(f)

        for pair in manifest_data.get("pairs", []):
            spec = {
                "out_id": pair.get("pair_id", f"pair_{len(subset_specs)}"),
                "src_path": pair.get("src_path"),
                "tgt_path": pair.get("tgt_path"),
                "text": pair.get("text", "portrait photo"),
                "heatmap_path": pair.get("heatmap_path", None)
            }
            if spec["src_path"] and spec["tgt_path"]:
                subset_specs.append(spec)
    else:
        print("[INFO] Manifest not found. Falling back to standard directory reading...")
        base_img_dir = "celeba-hq_sample"
        src_dir = os.path.join(base_img_dir, "src")
        tgt_dir = os.path.join(base_img_dir, "target")

        if not os.path.exists(src_dir) or not os.path.exists(tgt_dir):
            raise FileNotFoundError(f"Missing {src_dir} or {tgt_dir}. Please provide images or a manifest.")

        img_exts = ('.png', '.jpg', '.jpeg', '.webp')
        src_files = sorted([f for f in os.listdir(src_dir) if f.lower().endswith(img_exts)])
        tgt_files = sorted([f for f in os.listdir(tgt_dir) if f.lower().endswith(img_exts)])

        if not src_files or not tgt_files:
            raise ValueError(f"No images found in {src_dir} or {tgt_dir}.")

        # Round-robin pairing: cycle through targets if there are more sources than targets
        num_tgts = len(tgt_files)
        for i, src_name in enumerate(src_files):
            tgt_name = tgt_files[i % num_tgts]

            src_base = os.path.splitext(src_name)[0]
            tgt_base = os.path.splitext(tgt_name)[0]

            subset_specs.append({
                "out_id": f"{src_base}__to__{tgt_base}",
                "src_path": os.path.join(src_dir, src_name),
                "tgt_path": os.path.join(tgt_dir, tgt_name),
                "text": "portrait photo",  # Fallback default text
                "heatmap_path": None
            })

    print(f"[INFO] Loaded {len(subset_specs)} pairs for evaluation.")

    import pandas as pd
    results_table = []

    for exp in experiments:
        print(f"\n>>> Running Setup: {exp['name']} <<<")
        cfg = copy.deepcopy(base_cfg)
        for param, val in exp["changes"].items(): setattr(cfg, param, val)

        exp_dir = os.path.join(f"results_{RUN_ID}", f"run_{exp['name']}")
        ensure_dir(exp_dir)

        current_pair_log = []
        for spec in subset_specs:
            train_one_pair(spec, exp_dir, TRAIN_ENSEMBLE, tokenizer, text_encoder, vae, unet_base, noise_scheduler,
                           alphas_cumprod, fa_model, cfg)
            current_pair_log.append({"out_dir": os.path.join(exp_dir, spec['out_id']), "src_path": spec["src_path"],
                                     "tgt_path": spec["tgt_path"]})

        metric_sums = {"psnr": 0, "ssim": 0, "cos": 0, "asr_count": 0}
        eval_fr_name, tau_asr = EVAL_ENSEMBLE[0]["name"], THRESHOLDS[EVAL_ENSEMBLE[0]["name"]][1]

        for pair_info in current_pair_log:
            res = reevaluate_one_pair_ensemble(pair_info, EVAL_ENSEMBLE, fa_model, cfg)
            metric_sums["psnr"] += res["psnr_final"]
            metric_sums["ssim"] += res["ssim_final"]
            cos_val = res["cos_tgt_final"].get(eval_fr_name, 0.0)
            metric_sums["cos"] += cos_val
            if cos_val > tau_asr: metric_sums["asr_count"] += 1

        n = max(1, len(current_pair_log))
        results_table.append(
            {"Method": exp["name"], "ASR%": (metric_sums["asr_count"] / n) * 100, "Mean_Cos": metric_sums["cos"] / n,
             "PSNR": metric_sums["psnr"] / n, "SSIM": metric_sums["ssim"] / n})

    df = pd.DataFrame(results_table)
    print("\n=== FINAL RESULTS TABLE ===")
    print(df)
    df.to_csv(f"adv_tgd_{'ablation' if run_ablation else 'main'}_{RUN_ID}.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Adv-TGD Adversarial Face Generation Framework.")
    parser.add_argument("--ablation", action="store_true",
                        help="Run the full ablation study instead of just the main method.")
    args = parser.parse_args()
    run_adv_tgd(run_ablation=args.ablation)
