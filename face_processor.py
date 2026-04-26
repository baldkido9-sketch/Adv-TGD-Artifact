import os
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class FaceProcessor:
    def __init__(self, fa_model, config):
        self.fa = fa_model
        self.cfg = config
        self.tfm = transforms.Compose([
            transforms.Resize((self.cfg.resolution, self.cfg.resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def align_face(self, image_path):
        crop_scale = 0.65
        output_size = self.cfg.resolution

        if self.fa is None:
            return Image.open(image_path).convert("RGB").resize((output_size, output_size)), None

        try:
            if not os.path.exists(image_path):
                print(f"\\n[ALIGN ERROR] File does not exist at path: {image_path}")
                return None, None

            img_np = cv2.imread(image_path)
            if img_np is None:
                return None, None

            img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
            landmarks_list = self.fa.get_landmarks(img_np)

            if not landmarks_list:
                return Image.open(image_path).convert("RGB").resize((output_size, output_size)), None

            landmarks = max(landmarks_list, key=lambda l: np.linalg.norm(l[0] - l[-1]))
            src_landmarks = np.array([
                landmarks[36:42].mean(axis=0), landmarks[42:48].mean(axis=0),
                landmarks[30], landmarks[48], landmarks[54],
            ], dtype=np.float32)

            cx, cy = output_size / 2, output_size / 2
            eye_x, eye_y = 175.0 * crop_scale, -60.0 * crop_scale
            nose_y = 65.0 * crop_scale
            mouth_y, mouth_x = 140.0 * crop_scale, 150.0 * crop_scale
            ref_landmarks = np.array([
                [cx - eye_x, cy + eye_y], [cx + eye_x, cy + eye_y],
                [cx, cy + nose_y], [cx - mouth_x, cy + mouth_y], [cx + mouth_x, cy + mouth_y],
            ], dtype=np.float32)

            M, _ = cv2.estimateAffinePartial2D(src_landmarks, ref_landmarks, method=cv2.LMEDS)
            if M is None: return None, None

            warped = cv2.warpAffine(img_np, M, (output_size, output_size),
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
            return Image.fromarray(warped), M
        except Exception as e:
            print(f"\\n[ALIGN ERROR] Exception during alignment of {image_path}: {e}")
            return None, None

    def get_landmark_mask(self, img_pil_or_np, expand_forehead=0.6):
        if self.fa is None: return None, None
        img_np = np.array(img_pil_or_np) if isinstance(img_pil_or_np, Image.Image) else img_pil_or_np

        try:
            preds = self.fa.get_landmarks(img_np)
            if not preds: return None, None
            lms = preds[0]

            jaw = lms[0:17]
            lbrow = lms[17:22]
            rbrow = lms[22:27]

            face_width = np.linalg.norm(lms[16] - lms[0])
            brow_center = np.mean(np.concatenate([lbrow, rbrow], axis=0), axis=0)
            forehead_height = face_width * expand_forehead

            theta = np.linspace(np.pi, 0, 10)
            rx = face_width * 0.45
            ry = forehead_height

            forehead_arc = np.zeros((10, 2))
            forehead_arc[:, 0] = brow_center[0] + rx * np.cos(theta)
            forehead_arc[:, 1] = (np.min(lms[17:27, 1]) - ry * np.sin(theta))

            contour = np.array([*jaw, *np.flip(rbrow, axis=0), *forehead_arc, *np.flip(lbrow, axis=0)], dtype=np.int32)

            mask = np.zeros((self.cfg.resolution, self.cfg.resolution), dtype=np.uint8)
            cv2.fillPoly(mask, [contour], 255)
            mask = cv2.dilate(mask, np.ones((15, 15), np.uint8), iterations=1)
            mask = cv2.GaussianBlur(mask, (21, 21), 0)

            mask_tensor = torch.from_numpy(mask).float().div(255.0).unsqueeze(0).unsqueeze(0).to(self.cfg.device)
            return mask_tensor, contour
        except Exception:
            return None, None

    def clean_and_feather_mask(self, soft_mask_torch, dilate_ksize=25, blur_ksize=31):
        m = soft_mask_torch.detach().clone()[0, 0]
        H, W = m.shape[-2], m.shape[-1]
        m_np = m.cpu().numpy().astype(np.float32)
        bin_mask = (m_np > 0.5).astype(np.uint8)

        num_labels, labels_im = cv2.connectedComponents(bin_mask)
        bin_largest = bin_mask
        if num_labels > 1:
            largest_label = max(range(1, num_labels), key=lambda lab: (labels_im == lab).sum())
            bin_largest = (labels_im == largest_label).astype(np.uint8)

        bin_dil = cv2.dilate(bin_largest, np.ones((dilate_ksize, dilate_ksize), np.uint8),
                             iterations=1) if dilate_ksize > 1 else bin_largest

        if blur_ksize % 2 == 0: blur_ksize += 1
        bin_blur = np.clip(cv2.GaussianBlur(bin_dil.astype(np.float32), (blur_ksize, blur_ksize), 0), 0.0, 1.0)
        return torch.from_numpy(bin_blur).to(soft_mask_torch.device, dtype=soft_mask_torch.dtype).view(1, 1, H,
                                                                                                       W).clamp(0, 1)

    def blend_image(self, aligned_step_img_path, original_src_path, M_src):
        try:
            img_to_blend_cv = cv2.cvtColor(np.array(Image.open(aligned_step_img_path).convert("RGB")),
                                           cv2.COLOR_RGB2BGR)
            original_src_cv = cv2.imread(original_src_path)
            h, w, _ = original_src_cv.shape

            if M_src is None:
                return Image.fromarray(cv2.cvtColor(img_to_blend_cv, cv2.COLOR_BGR2RGB))

            M_inverse = cv2.invertAffineTransform(M_src)
            warped_final_img = cv2.warpAffine(img_to_blend_cv, M_inverse, (w, h))

            mask_template = np.ones((self.cfg.resolution, self.cfg.resolution), dtype=np.uint8) * 255
            warped_mask_hard = cv2.warpAffine(mask_template, M_inverse, (w, h))

            x, y, w_mask, h_mask = cv2.boundingRect(warped_mask_hard)
            center = (x + w_mask // 2, y + h_mask // 2)

            if self.cfg.blending_mode == 'alpha':
                warped_mask_soft = cv2.GaussianBlur(warped_mask_hard, (11, 11), 0)
                alpha = np.expand_dims(warped_mask_soft.astype(float) / 255.0, axis=2)
                blended = (warped_final_img * alpha + original_src_cv * (1.0 - alpha))
            elif self.cfg.blending_mode == 'poisson_normal':
                blended = cv2.seamlessClone(warped_final_img, original_src_cv, warped_mask_hard, center,
                                            cv2.NORMAL_CLONE)
            elif self.cfg.blending_mode == 'poisson_mixed':
                blended = cv2.seamlessClone(warped_final_img, original_src_cv, warped_mask_hard, center,
                                            cv2.MIXED_CLONE)
            else:
                blended = original_src_cv.copy()
                blended[warped_mask_hard > 127] = warped_final_img[warped_mask_hard > 127]

            return Image.fromarray(cv2.cvtColor(blended.astype(np.uint8), cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Blending error: {e}")
            return None


# ---------------- Mask Generation Math Helpers ----------------

def _normalize01(t):
    t = t - t.min()
    d = t.max() - t.min()
    return t / (d + 1e-8)


def _top_percent_soft_mask(sal01, keep_percent=0.15, feather_k=5, prior=None, flat_eps=1e-3):
    B, _, H, W = sal01.shape
    sal01 = sal01.clamp(0, 1)

    s_min = sal01.amin(dim=(2, 3), keepdim=True)
    s_max = sal01.amax(dim=(2, 3), keepdim=True)
    is_flat = (s_max - s_min) < flat_eps

    hard = torch.zeros_like(sal01)
    total_pix = H * W
    k_keep = max(1, int(round(total_pix * float(keep_percent))))

    v = sal01.view(B, -1)
    for b in range(B):
        if is_flat[b].item():
            if prior is not None:
                pv = prior[b:b + 1].reshape(-1)
                idx = torch.topk(pv, k_keep, largest=True).indices
            else:
                idx = torch.topk(v[b], k_keep, largest=True).indices
            hb = torch.zeros_like(v[b])
            hb[idx] = 1.0
            hard[b:b + 1] = hb.view(1, 1, H, W)
        else:
            q = max(0.0, min(1.0, 1.0 - float(keep_percent) - 1e-6))
            thr = torch.quantile(v[b], q)
            hb = (v[b] > thr).float()
            if int(hb.sum().item()) != k_keep:
                idx = torch.topk(v[b], k_keep, largest=True).indices
                hb.zero_()
                hb[idx] = 1.0
            hard[b:b + 1] = hb.view(1, 1, H, W)

    if feather_k <= 1: return hard

    ker = torch.ones(1, 1, feather_k, feather_k, device=sal01.device, dtype=sal01.dtype) / (feather_k ** 2)
    return F.conv2d(hard, ker, padding=feather_k // 2).clamp(0, 1)


def _ellipse_prior(H, W, rx_frac=0.34, ry_frac=0.42, device="cpu", dtype=torch.float32):
    yy = torch.linspace(0, H - 1, H, device=device, dtype=dtype).view(H, 1)
    xx = torch.linspace(0, W - 1, W, device=device, dtype=dtype).view(1, W)
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    rx = max(1.0, rx_frac * W)
    ry = max(1.0, ry_frac * H)
    dx2 = ((xx - cx) / rx) ** 2
    dy2 = ((yy - cy) / ry) ** 2
    d = dx2 + dy2
    hard = (d <= 1.0).float()
    soft = (1.1 - d).clamp(0, 1)
    return torch.maximum(hard, soft).view(1, 1, H, W)


def _fr_target_saliency_from_src(src_img_bchw, fr_list, e_tgt_dict, prep_fr_fn):
    img = src_img_bchw.clone().detach().requires_grad_(True)
    grads = []
    for fr in fr_list:
        size = fr["size"]
        e_pred = F.normalize(fr["model"](prep_fr_fn(img, size)), p=2, dim=1)
        cos_tgt = (e_pred * e_tgt_dict[fr["name"]]).sum()
        cos_tgt.backward(retain_graph=True)
        g = img.grad.detach().abs().mean(dim=1, keepdim=True)
        grads.append(g)
        img.grad.zero_()
    return torch.stack(grads, dim=0).mean(dim=0)


def save_sgsm_visualization(src_img, saliency_map, semantic_hull, final_mask, save_path):
    saliency_norm = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min())
    heatmap = cv2.applyColorMap((saliency_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)

    hull_viz = src_img.copy()
    cv2.polylines(hull_viz, [semantic_hull], isClosed=True, color=(0, 255, 0), thickness=2)

    mask_overlay = src_img.copy()
    mask_bool = final_mask > 0.5
    mask_overlay[mask_bool] = mask_overlay[mask_bool] * 0.5 + np.array([0, 0, 255]) * 0.5

    top_row = np.hstack([src_img, heatmap, hull_viz, mask_overlay])
    cv2.imwrite(save_path, top_row)
