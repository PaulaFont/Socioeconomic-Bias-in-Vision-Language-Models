# Socioeconomic Bias in CLIP for Satellite Imagery

**Author:** Paula Font Solà  
**Course:** Deep Learning for Visual Signal Processing 2, May 2026

Does CLIP encode socioeconomic bias when applied to satellite imagery of European cities, and which specific visual and linguistic features drive it?

---

## Overview

CLIP was trained on ~400 million image–text pairs scraped from the public internet; a corpus heavily skewed towards English-speaking, US-centric content. This project investigates what happens when you ask CLIP to judge whether a satellite image of a Spanish neighbourhood looks "wealthy" or "poor", and whether its answers reflect actual income levels or stereotypes learned from web data.

The study covers **8,044 census tracts** across Madrid and Barcelona, paired with official net household income data from the Spanish National Statistics Institute (INE).

---

## Repository Structure

```
├── 1-dataset-creation.ipynb   # Satellite image dataset generation pipeline
├── 2-main-analysis.ipynb      # Full experimental analysis (submit this one)
└── README.md
```

**`1-dataset-creation.ipynb`** —> Downloads Spanish census tract shapefiles and INE income data, computes geometric centroids, and fetches 512×512px satellite tiles from the IGN PNOA WMS API for each tract.

**`2-main-analysis.ipynb`** —> The main investigation. Self-contained and readable top to bottom. Covers five connected experiments (see below).

---

## Dataset

The satellite imagery dataset is publicly available on Kaggle:

**[🛰️ satellite-socioeconomic-spain](https://www.kaggle.com/datasets/paulafontsola/satellite-socioeconomic-spain)**

8,044 satellite images (512×512px) of Spanish census tracts in Madrid and Barcelona, paired with official INE net household income per household (€).

---

## Experiments

### I. Baseline Zero-Shot Retrieval
CLIP wealth scores vs. official INE income: global Pearson r = **0.107**. Weak, but not random: a Mann-Whitney test (p<0.001) confirms a statistically real ordering suppressed by a systematic positive-label bias. CLIP calls 65.7% of the *poorest* tracts "wealthy".

### II. Visual Interpretability
Attention rollout on false positives and false negatives, plus 8-feature cosine similarity probing. **Swimming pools (r=0.321) and vegetation (r=0.250)** are CLIP's dominant wealth proxies. These are valid in American suburbs, but systematically wrong in dense European cities like Barcelona's Eixample.

### III. Prompt Engineering
A 15+15 prompt ensemble improves Pearson r from 0.107 to **0.193**. A regression decomposition shows this gain is mostly explained by the same visual proxies. The ensemble is a more reliable measure of the same underlying bias, not evidence of better socioeconomic understanding.

### IV. Domain-Specific Model Comparison

| Model | Pearson r | Training data |
|---|---|---|
| OpenAI CLIP (single prompt) | 0.107 | Internet (WebImageText, ~400M) |
| OpenAI CLIP (15+15 ensemble) | 0.193 | Internet. Prompt-averaged |
| SigLIP (15+15 ensemble) | 0.236 | Internet (WebLI, sigmoid loss) |
| RemoteCLIP (single prompt) | 0.025 | Satellite captions (domain fine-tuned) |

RemoteCLIP's near-zero correlation is the most informative result: domain fine-tuning strips away the web-derived socioeconomic associations entirely.

### V. Text-Space Cultural Probing
Without any satellite image, CLIP's text encoder reveals its definition of wealth:

```
0.9399  an American suburb
0.9329  a typical American city
0.9242  a neighborhood in the United States
0.8943  a typical European city
0.8780  a European historic city center
0.8479  a neighborhood in Spain
0.8388  green lawns and separated houses
0.8191  dense concrete apartment blocks
```

The bias is linguistically encoded before any image is processed. It lives in training data composition, not in the visual structure of satellite imagery.

---

## Key Finding

CLIP applies a **suburban American morphological template** to European cities. It fails more severely where the urban form diverges most from that template (Barcelona > Madrid). This is not a vision problem; it is a language–image alignment problem rooted in the geographic skew of the training corpus.

---

## Requirements

```
transformers
open_clip_torch
torch torchvision
geopandas
codecarbon
huggingface_hub
```

---

## References

- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision*. ICML.
- Liu et al. (2023). *RemoteCLIP: A Vision Language Foundation Model for Remote Sensing*. arXiv:2306.11029.
- Zhai et al. (2023). *SigLIP: Sigmoid Loss for Language Image Pre-Training*. ICCV.
- Jean et al. (2016). Combining satellite imagery and machine learning to predict poverty. *Science*, 353(6301).
- INE (2021). *Atlas de distribución de renta de los hogares*. Instituto Nacional de Estadística, España.
