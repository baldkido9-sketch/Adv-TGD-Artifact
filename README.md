# Adv-TGD: Adversarial Text-Guided Diffusion for Face Recognition Impersonation Attacks

Official implementation of **Adv-TGD**, a generative adversarial framework that synthesizes photorealistic faces capable of impersonating target identities to deceive face recognition (FR) systems. Adv-TGD utilizes per-sample LoRA fine-tuning and a hybrid Salience-Guided Semantic Mask (SGSM) to achieve high attack success rates while maintaining superior visual fidelity.

---

## 📥 Datasets and Pre-trained Weights

To evaluate the artifacts and reproduce the pipeline, the following datasets, generative models, and evaluation models are required.

### Datasets
All experiments utilize the publicly available **CelebA-HQ** and **FFHQ** datasets. You can refer to [CelebAMask-HQ](https://github.com/switchablenorms/CelebAMask-HQ) for CelebA-HQ download.
1. Download the datasets from their standard academic distribution sources.
2. Extract the data and place the specific source and target images into the corresponding directories prior to execution:
   * `celeba-hq_sample/src/`
   * `celeba-hq_sample/target/`

### Generative Model Weights
* **Base Models:** The framework automatically fetches generative backbones from HuggingFace (e.g., `stabilityai/stable-diffusion-2-1`). 
### Face Recognition (FR) Surrogate Weights
To evaluate the Attack Success Rate (ASR) and reproduce the loss guidance, pre-trained FR models are required. 
* The framework utilizes **IR152**, **IRSE50**, **MobileFace**, and **FaceNet** as the surrogate ensemble and evaluation models.
* These pre-trained weights can be downloaded from the public repository of prior work: [kopperx/Adv-Diffusion](https://github.com/kopperx/Adv-Diffusion). google drive link [here](https://drive.google.com/file/d/1Vuek5-YTZlYGoeoqyM5DlvnaXMeii4O8/view).
* Once downloaded, place the FR weights in the `pretrained_model` directory.
---

## 🛠 Environment Setup

We recommend using a Conda environment for consistent dependency management.

### Option A: Create from environment.yml (Recommended)

```bash
conda env create -f environment.yml
conda activate adv-tgd
```

### Option B: Manual Installation

```bash
pip install torch torchvision torchaudio
pip install diffusers transformers peft
pip install face_alignment opencv-python lpips open_clip_torch pandas tqdm insightface onnxruntime-gpu
```

---

## 📂 Project Structure

* `main.py`: Central execution engine and training loop
* `config.py`: Configuration classes and experimental hyperparameters
* `losses.py`: Implementation of composite adversarial objectives
* `face_processor.py`: Face alignment, SGSM generation, and seamless blending logic
* `environment.yml`: dependency environment file
* `celeba-hq_sample/`: Directory for source (`src/`) and target (`target/`) images

---

## 🚀 Usage

### 1. Running the Primary Attack

To execute the Adv-TGD pipeline using the proposed SGSM method:

```bash
python main.py
```

---

### 2. Reproducing Ablation Studies (Table 3)

To run the full suite of masking variants (Saliency-Only, Parsing-Only, Full-Image, and SGSM):

```bash
python main.py --ablation
```

---

## 📊 Dataset Loading

The script supports two loading modes:

* **Manifest Mode**:
  If `pairs_manifest_updated.json` exists, the script uses it to pair specific source images with target identities and LLaVA-generated prompts.

* **Directory Mode (Fallback)**:
  If no manifest is found, the script automatically performs round-robin pairing between images in:

  * `celeba-hq_sample/src/`
  * `celeba-hq_sample/target/`

---

## 📜 Ethical Considerations

This research is intended for defensive security analysis and privacy protection. All experiments utilized publicly available research datasets (CelebA-HQ). We do not release pre-trained weights for specific individuals. Our methodology aims to inform the development of more robust biometric verification systems against generative threats.
