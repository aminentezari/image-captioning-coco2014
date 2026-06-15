<div align="center">

# 🖼️ Image Captioning on COCO 2014

### From a Transformer Built from Scratch to Parameter-Efficient Fine-Tuning with BLIP + LoRA

[![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/🤗%20Demo-Live-yellow)](https://huggingface.co/spaces/amin-en/image-captioning-coco)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

**Amin Entezari · Ali Sedghiye**

*Master's Degree in Data Science — Università degli Studi di Milano-Bicocca*
*Foundations of Deep Learning — A.Y. 2025/2026*

[🚀 Live Demo](https://huggingface.co/spaces/amin-en/image-captioning-coco) · [📄 Report](report/ImageCaptioning_Report.pdf) · [📊 Presentation](presentation/)

</div>

---

## 📌 Overview

This repository contains the full implementation of our final project for the Foundations of Deep Learning course. We build and compare two image captioning systems on the **COCO 2014** dataset:

| Model | BLEU-1 | BLEU-4 | CIDEr | Trainable Params |
|---|---|---|---|---|
| Custom — 5k samples | 29.76 | 4.34 | — | 29.3M |
| Custom — 100k samples | 37.91 | 9.39 | 0.85 | 29.3M |
| **BLIP + LoRA** | **76.00** | **36.54** | **1.35** | **1.77M (0.71%)** |

The **central finding**: pretraining scale dominates architectural design. A model fine-tuned from 129M image–text pairs achieves 4× better BLEU-4 while training only 0.71% of its parameters.

---

## 🏗️ Architecture

### Custom Model
```
Input Image (224×224)
       │
       ▼
EfficientNetV2-S         ← pretrained on ImageNet-1K
(visual encoder)
       │
       ▼
49 patches × 256-dim     ← 7×7 spatial grid + positional embedding
       │
       ▼
6-Layer Transformer      ← cross-attention + causal mask
Decoder
       │
       ▼
Caption (beam search, beam=5)
```

**Two-phase training strategy:**
- **Phase 1** (8 epochs, lr=3e-4): encoder frozen, decoder learns from scratch
- **Phase 2** (20 epochs, lr=1e-4): encoder blocks 6–7 unfrozen, joint fine-tuning

### BLIP + LoRA
- Base model: `Salesforce/blip-image-captioning-base` (249M params)
- LoRA rank r=8 injected into Q/K/V attention projections
- Only **1.77M parameters trained** — the rest are frozen
- Fine-tuned on COCO 2014 training captions

---

## 📂 Repository Structure

```
image-captioning-coco2014/
│
├── notebooks/
│   ├── Training.ipynb              # Quick test — 5k samples (Phase 1+2, 5 epochs)
│   ├── Traning2.ipynb              # Full training — 100k samples (8+20 epochs)
│   └── blip_lora_coco2014.ipynb    # BLIP + LoRA fine-tuning on COCO 2014
│
├── demo/
│   └── app.py                      # Gradio demo (deployed on HuggingFace Spaces)
│
├── report/
│   ├── ImageCaptioning_Report.tex  # LaTeX source
│   ├── ImageCaptioning_Report.pdf  # Compiled PDF
│   └── figures/                    # All figures used in the report
│       ├── architecture.png
│       ├── training_curves.png
│       ├── ablation_study.png
│       ├── qualitative_results.png
│       ├── quick_test_results.png
│       ├── heatmap.png
│       ├── final-comparision.png
│       └── effectoftraining.png
│
├── presentation/
│   └── ImageCaptioning_Presentation.pptx
│
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🧪 Ablation Study

We isolate the contribution of each design choice under identical conditions (50k samples, 5 epochs):

| Configuration | BLEU-4 | CIDEr |
|---|---|---|
| 2-layer decoder, frozen encoder | 5.38 | 0.52 |
| 6-layer decoder, frozen encoder | 6.21 | 0.59 |
| **6-layer decoder, fine-tuned encoder (ours)** | **7.84** | **0.77** |

**Key insights:**
- Deeper decoder: +0.83 BLEU-4
- Encoder fine-tuning: +1.63 BLEU-4 (the single largest gain)

---

## 👁️ Attention Heatmaps

The cross-attention mechanism is fully interpretable. For each generated word, we extract the 49 attention weights from the last decoder layer, reshape to 7×7, upsample to 224×224, and overlay as a heatmap.

- Content words (*urinals*, *window*, *bicycle*, *clock*) attend to the correct image regions
- Function words (*with*, *of*, *and*) attend diffusely — no specific region needed

This confirms the model learns genuine visual–semantic alignment.

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR-USERNAME/image-captioning-coco2014.git
cd image-captioning-coco2014
```

### 2. Install dependencies
```bash
pip install torch torchvision transformers peft gradio pillow-heif
pip install pycocoevalcap nltk
```

### 3. Run the demo locally
```bash
cd demo
python app.py
# Open http://localhost:7860
```

### 4. Run training notebooks
Open in **Google Colab** or **Kaggle**:
- `notebooks/Training.ipynb` — quick 5k test (~15 min on T4)
- `notebooks/Traning2.ipynb` — full 100k training (~6 hours on A100)
- `notebooks/blip_lora_coco2014.ipynb` — BLIP + LoRA fine-tuning (~3 hours on T4)

---

## 📦 Checkpoints

Checkpoints are **not stored in this repository** (too large for GitHub). They are hosted on HuggingFace:

| File | Size | Location |
|---|---|---|
| `best_phase1.pt` (5k) | ~112 MB | HuggingFace Space |
| `best_phase2.pt` (100k) | ~112 MB | HuggingFace Space |
| `blip_adapter/` (LoRA) | ~7 MB | HuggingFace Space |

To use them locally, download from the [HuggingFace Space files](https://huggingface.co/spaces/amin-en/image-captioning-coco/tree/main).

---

## 🛠️ Tech Stack

| Component | Tool |
|---|---|
| Deep Learning | PyTorch 2.5+ |
| Visual Encoder | EfficientNetV2-S (torchvision) |
| Foundation Model | BLIP (HuggingFace Transformers) |
| Parameter-Efficient FT | LoRA (PEFT) |
| Demo | Gradio 4.x |
| Deployment | HuggingFace Spaces (CPU) |
| Training | Google Colab Pro (A100) + Kaggle (T4) |
| Dataset | COCO 2014 |
| Evaluation | BLEU (NLTK), CIDEr (pycocoevalcap) |

---

## 🔗 Links

| Resource | Link |
|---|---|
| 🤗 Live Demo | https://huggingface.co/spaces/amin-en/image-captioning-coco |
| 📄 Full Report (PDF) | [report/ImageCaptioning_Report.pdf](report/ImageCaptioning_Report.pdf) |
| 📊 Presentation | [presentation/](presentation/) |
| 📦 COCO 2014 Dataset | https://cocodataset.org |

---

## 👥 Authors

<table>
  <tr>
    <td align="center"><b>Amin Entezari</b></td>
    <td align="center"><b>Ali Sedghiye</b></td>
  </tr>
  <tr>
    <td align="center">Master's in Data Science<br>Milano-Bicocca</td>
    <td align="center">Master's in Data Science<br>Milano-Bicocca</td>
  </tr>
</table>

---

## 📝 Citation

If you use this work, please cite:

```bibtex
@misc{entezari2026captioning,
  title  = {Image Captioning on COCO 2014: From a Transformer Built
             from Scratch to Parameter-Efficient Fine-Tuning with BLIP + LoRA},
  author = {Entezari, Amin and Sedghiye, Ali},
  year   = {2026},
  school = {Universit\`{a} degli Studi di Milano-Bicocca},
  note   = {Foundations of Deep Learning, A.Y. 2025/2026}
}
```

---

## 📜 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
<sub>Built with ❤️ · Università degli Studi di Milano-Bicocca · 2026</sub>
</div>
