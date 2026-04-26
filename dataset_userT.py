import os, random, json
from typing import Optional, Literal, Dict, Any, List
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np

IMG_EXTS = ('.png', '.jpg', '.jpeg', '.webp')


def _basename_wo_ext(p: str) -> str:
    return os.path.splitext(os.path.basename(p))[0]


class SimpleSrcTgtDataset(Dataset):
    """
    Directory layout:
      img_dir/
        src/        <-- many sources (e.g., 1000)
        target/     <-- few targets (e.g., 6)
      text_dir/     <-- optional; files named <target_basename>.txt
      heatmap_dir/  <-- optional; .npy files named <target_basename>.npy

    __getitem__(i) (for batch training) returns:
      {
        "src":  (3,H,W) in [-1,1],
        "tgt":  (3,H,W) in [-1,1],
        "text": str ('' if not found),
        "heatmap": (1,H,W) heatmap tensor,
        ...
      }

    build_specs(...) returns:
      list of dicts: {"src_path","tgt_path","text","out_id","heatmap_path"}

    You can also feed a pairs manifest via:
      - pairs_manifest_path=".../pairs_manifest.json"   (or a dataset_summary_*.json with {"pairs":[...]} )
      - pairs_manifest_pairs=[ {...}, {...}, ... ]
    And subset deterministically with:
      - manifest_limit / manifest_fraction / manifest_shuffle
    """

    def __init__(
        self,
        img_dir: str,
        text_dir: Optional[str] = None,
        heatmap_dir: Optional[str] = None,
        resolution: int = 768,
        target_mode: Literal["random", "cycle", "fixed"] = "random",
        fixed_target_name: Optional[str] = None,
        seed: int = 1337,
        # manifest controls ↓↓↓
        pairs_manifest_path: Optional[str] = None,
        pairs_manifest_pairs: Optional[List[Dict[str, Any]]] = None,
        manifest_limit: Optional[int] = None,
        manifest_fraction: Optional[float] = None,
        manifest_shuffle: bool = False,
    ):
        super().__init__()
        self.img_dir = img_dir
        self.src_dir = os.path.join(img_dir, "src")
        self.tgt_dir = os.path.join(img_dir, "target")
        self.text_dir = text_dir
        self.heatmap_dir = heatmap_dir
        self.resolution = int(resolution)
        self.target_mode = target_mode
        self.fixed_target_name = fixed_target_name

        # RNG for deterministic choices
        self.rng = random.Random(seed)

        # Manifest config
        self._manifest_path = pairs_manifest_path
        self._manifest_pairs_in = pairs_manifest_pairs
        self._manifest_limit = manifest_limit
        self._manifest_fraction = manifest_fraction
        self._manifest_shuffle = manifest_shuffle

        # Precomputed manifest specs (or None if not using manifest)
        self._manifest_specs_list: Optional[List[Dict[str, Any]]] = None

        # Validate folders for standard mode
        if not os.path.isdir(self.src_dir) or not os.path.isdir(self.tgt_dir):
            raise RuntimeError(f"Expected 'src' and 'target' folders under {img_dir}")

        self.src_names = sorted(
            f for f in os.listdir(self.src_dir) if f.lower().endswith(IMG_EXTS)
        )
        self.tgt_names = sorted(
            f for f in os.listdir(self.tgt_dir) if f.lower().endswith(IMG_EXTS)
        )
        if len(self.src_names) == 0 or len(self.tgt_names) == 0:
            raise RuntimeError("No images found in src/ or target/.")

        if self.target_mode == "fixed":
            if self.fixed_target_name is None:
                raise ValueError("target_mode='fixed' requires fixed_target_name")
            if self.fixed_target_name not in self.tgt_names:
                raise ValueError(f"fixed_target_name '{self.fixed_target_name}' not found in target/")

        # Basic image -> [-1,1] tensor
        self.tf = transforms.Compose([
            transforms.Resize((self.resolution, self.resolution), interpolation=Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        # If a manifest is provided, parse and pre-build its spec list (with subsetting)
        self._maybe_build_manifest_specs()

    # ------------------------------ standard dataset bits ------------------------------

    def __len__(self) -> int:
        # Length is defined for batch-style loading from folder pairs (not used by manifest specs).
        return len(self.src_names)

    @staticmethod
    def _load_rgb(path: str) -> Image.Image:
        return Image.open(path).convert("RGB")

    def _load_heatmap_for_target(self, tgt_name: str) -> torch.Tensor:
        if not self.heatmap_dir:
            return torch.zeros(1, self.resolution, self.resolution)

        base = os.path.splitext(tgt_name)[0]
        p = os.path.join(self.heatmap_dir, f"{base}.npy")
        if os.path.exists(p):
            try:
                heatmap = np.load(p)
                h_img = Image.fromarray(heatmap)
                h_img_resized = h_img.resize((self.resolution, self.resolution), Image.BILINEAR)
                h_tensor = torch.from_numpy(np.array(h_img_resized)).unsqueeze(0).float()
                return h_tensor
            except Exception:
                return torch.zeros(1, self.resolution, self.resolution)
        return torch.zeros(1, self.resolution, self.resolution)

    def _load_text_for_target(self, tgt_name: str) -> str:
        """
        Return the entire contents of <text_dir>/<target_basename>.txt exactly as stored.
        No line truncation and no stripping, so multi-line prompts are preserved.
        """
        if not self.text_dir:
            return ""
        base = os.path.splitext(tgt_name)[0]
        p = os.path.join(self.text_dir, f"{base}.txt")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    return f.read()  # <-- full file (unchanged)
            except Exception:
                return ""
        return ""

    def _pick_target_idx(self, src_idx: int) -> int:
        if self.target_mode == "random":
            return self.rng.randrange(len(self.tgt_names))
        elif self.target_mode == "cycle":
            return src_idx % len(self.tgt_names)
        else:  # "fixed"
            return self.tgt_names.index(self.fixed_target_name)

    # ---------- standard batch-style item ----------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        src_name = self.src_names[idx]
        tgt_idx = self._pick_target_idx(idx)
        tgt_name = self.tgt_names[tgt_idx]

        src_path = os.path.join(self.src_dir, src_name)
        tgt_path = os.path.join(self.tgt_dir, tgt_name)

        src_img = self._load_rgb(src_path)
        tgt_img = self._load_rgb(tgt_path)

        src_t = self.tf(src_img)
        tgt_t = self.tf(tgt_img)
        text = self._load_text_for_target(tgt_name)
        heatmap_t = self._load_heatmap_for_target(tgt_name)

        return {
            "src": src_t,
            "tgt": tgt_t,
            "text": text,
            "heatmap": heatmap_t,
            "src_name": src_name,
            "tgt_name": tgt_name,
            "src_idx": idx,
            "tgt_idx": tgt_idx,
        }

    # ------------------------------ manifest helpers ------------------------------

    def _canon_pairs_list(self, raw) -> List[Dict[str, Any]]:
        if isinstance(raw, dict) and "pairs" in raw:
            return raw["pairs"]
        if isinstance(raw, list):
            return raw
        raise ValueError("Unsupported manifest format (expect list or dict with key 'pairs').")

    def _maybe_build_manifest_specs(self) -> None:
        raw_pairs = None
        if self._manifest_pairs_in is not None:
            raw_pairs = self._manifest_pairs_in
        elif self._manifest_path is not None and os.path.exists(self._manifest_path):
            with open(self._manifest_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            raw_pairs = self._canon_pairs_list(raw)

        if raw_pairs is None:
            self._manifest_specs_list = None
            return

        idxs = list(range(len(raw_pairs)))
        if self._manifest_shuffle:
            self.rng.shuffle(idxs)

        K = len(raw_pairs)
        if self._manifest_fraction is not None:
            K = max(1, int(round(K * float(self._manifest_fraction))))
        if self._manifest_limit is not None:
            K = min(K, int(self._manifest_limit))

        chosen = [raw_pairs[i] for i in idxs[:K]]

        specs: List[Dict[str, Any]] = []
        for i, rec in enumerate(chosen):
            sp = rec.get("src_path") or rec.get("src") or rec.get("source_path")
            tp = rec.get("tgt_path") or rec.get("tgt") or rec.get("target_path")

            if not sp or not tp:
                out_dir = rec.get("out_dir", "")
                if not sp or not os.path.exists(sp):
                    src_name = rec.get("src_name")
                    if src_name:
                        sp = os.path.join(self.src_dir, src_name)
                if not tp or not os.path.exists(tp):
                    tgt_name = rec.get("tgt_name")
                    if tgt_name:
                        tp = os.path.join(self.tgt_dir, tgt_name)
            if not sp or not tp:
                continue

            out_id = (
                rec.get("pair_id")
                or os.path.basename(rec.get("out_dir", ""))
                or f"{_basename_wo_ext(sp)}__to__{_basename_wo_ext(tp)}"
            )

            specs.append(self._make_spec(sp, tp, out_id=out_id))

        self._manifest_specs_list = specs

    def _make_spec(self, src_path: str, tgt_path: str, out_id: Optional[str] = None) -> Dict[str, Any]:
        tgt_name = os.path.basename(tgt_path)
        text = self._load_text_for_target(tgt_name)

        if out_id is None:
            out_id = f"{_basename_wo_ext(os.path.basename(src_path))}__to__{_basename_wo_ext(os.path.basename(tgt_path))}"

        heatmap_path = None
        if self.heatmap_dir:
            base = os.path.splitext(tgt_name)[0]
            hp = os.path.join(self.heatmap_dir, f"{base}.npy")
            if os.path.exists(hp):
                heatmap_path = hp

        return {
            "src_path": os.path.normpath(src_path),
            "tgt_path": os.path.normpath(tgt_path),
            "text": text,
            "out_id": out_id,
            "heatmap_path": heatmap_path
        }

    # ------------------------------ spec builder (used by your trainer) ------------------------------

    def build_specs(
        self,
        pairing: Literal["round_robin", "zip", "cartesian"] = "round_robin",
        out_id_template: str = "{src_base}__to__{tgt_base}",
        limit_src: Optional[int] = None,
        limit_tgt: Optional[int] = None,
    ) -> List[Dict[str, Any]]:

        if self._manifest_specs_list is not None:
            return list(self._manifest_specs_list)

        srcs = self.src_names[:limit_src] if limit_src else self.src_names
        tgts = self.tgt_names[:limit_tgt] if limit_tgt else self.tgt_names

        items: List[Dict[str, Any]] = []

        def make_item(src_name: str, tgt_name: str) -> Dict[str, Any]:
            sp = os.path.join(self.src_dir, src_name)
            tp = os.path.join(self.tgt_dir, tgt_name)
            text = self._load_text_for_target(tgt_name)
            out_id = out_id_template.format(
                src_base=_basename_wo_ext(src_name),
                tgt_base=_basename_wo_ext(tgt_name),
            )

            heatmap_path = None
            if self.heatmap_dir:
                base = os.path.splitext(tgt_name)[0]
                hp = os.path.join(self.heatmap_dir, f"{base}.npy")
                if os.path.exists(hp):
                    heatmap_path = hp

            return {"src_path": sp, "tgt_path": tp, "text": text, "out_id": out_id, "heatmap_path": heatmap_path}

        if pairing == "round_robin":
            T = len(tgts)
            for i, s in enumerate(srcs):
                t = tgts[i % T]
                items.append(make_item(s, t))
        elif pairing == "zip":
            for s, t in zip(srcs, tgts):
                items.append(make_item(s, t))
        elif pairing == "cartesian":
            for s in srcs:
                for t in tgts:
                    items.append(make_item(s, t))
        else:
            raise ValueError(f"Unknown pairing mode: {pairing}")

        return items


def simple_collate(batch):
    """Stack tensors; keep texts as list."""
    srcs, tgts, texts, meta = [], [], [], []
    for b in batch:
        if not isinstance(b, dict) or ("src" not in b) or ("tgt" not in b):
            continue
        srcs.append(b["src"])
        tgts.append(b["tgt"])
        texts.append(b.get("text", ""))
        meta.append({k: b[k] for k in ("src_name", "tgt_name", "src_idx", "tgt_idx")})
    if len(srcs) == 0:
        return None
    return {
        "src": torch.stack(srcs, dim=0),
        "tgt": torch.stack(tgts, dim=0),
        "text": texts,
        "meta": meta,
    }


# ---- Example usage ----
if __name__ == "__main__":
    # Example 1: standard folder mode (FFHQ)
    ds = SimpleSrcTgtDataset(
        img_dir="ffhq_sample",
        text_dir="ffhq_sample/target_text",
        heatmap_dir="ffhq_sample/target_heatmap",
        resolution=768,
        target_mode="cycle",
        seed=1337,
    )
    item = ds[0]
    print("[getitem] src:", item["src"].shape)
    print("[getitem] heatmap:", item["heatmap"].shape, "sum:", float(item["heatmap"].sum()))
    print("[getitem] text (first 120 chars):", (item["text"] or "")[:120])

    # Example 2: manifest mode with half the pairs (deterministic) on FFHQ
    ds_m = SimpleSrcTgtDataset(
        img_dir="ffhq_sample",
        text_dir="ffhq_sample/target_text",
        heatmap_dir="ffhq_sample/target_heatmap",
        resolution=768,
        seed=1337,
        pairs_manifest_path="pairs_manifest.json",  # or dataset_summary_*.json with {"pairs":[...]}
        manifest_fraction=0.5,
        manifest_shuffle=True,
    )
    specs = ds_m.build_specs()
    print("Total in manifest subset:", len(specs))
    print("First spec:", specs[0] if specs else None)
