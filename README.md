# Plug-and-Play Fidelity Optimization for Diffusion Transformer Acceleration via Cumulative Error Minimization

:sparkles: **ICLR 2026** :sparkles:· [arXiv:2512.23258](https://arxiv.org/abs/2512.23258)

---

## :book: Introduction

Although Diffusion Transformer (DiT) has emerged as a predominant architecture for image and video generation, its iterative denoising process results in slow inference, which hinders broader applicability and development. Caching-based methods achieve training-free acceleration, while suffering from considerable computational error. Existing methods typically incorporate error correction strategies such as pruning or prediction to mitigate it. However, their fixed caching strategy fails to adapt to the complex error variations during denoising, which limits the full potential of error correction.

<p align="center">
  <img src="imgs/iclr_head.svg" alt="CEM: overview of plug-and-play fidelity optimization for cache-based DiT acceleration" width="95%" />
</p>

To tackle this challenge, we propose a novel fidelity-optimization plugin for existing error correction methods via cumulative error minimization, named CEM. CEM predefines the error to characterize the sensitivity of model to acceleration jointly influenced by timesteps and cache intervals. Guided by this prior, we formulate a dynamic programming algorithm with cumulative error approximation for strategy optimization, which achieves the caching error minimization, resulting in a substantial improvement in generation fidelity. CEM is model-agnostic and exhibits strong generalization, which is adaptable to arbitrary acceleration budgets. It can be seamlessly integrated into existing error correction frameworks and quantized models without introducing any additional computational overhead. Extensive experiments conducted on nine generation models and quantized methods across three tasks demonstrate that CEM significantly improves generation fidelity of existing acceleration models, and outperforms the original generation performance on FLUX.1-dev, PixArt, StableDiffusion1.5 and Hunyuan.

<p align="center">
  <img src="imgs/iclr_model.svg" alt="CEM: formulation and integration with the denoising loop" width="95%" />
</p>

---

## :rocket: Code

### 2.1 Class-to-Image Models (DiT-XL/2)

**Plug-in on DuCa**

- CEM is a **plug-in**: We use **[DuCa](https://github.com/shenyi-z/duca)** as the representative host method.

**Environment**

- Base: CUDA 11.8, PyTorch 2.6.0, torchvision 0.21.0.
- Installation: We provide a list of our environment packages for your reference during installation; some redundant packages can be ignored. See [`CEM/DiT/requirements.txt`](DiT/requirements.txt)
- Checkpoints: We test performance at 256 resolution on DiT, requiring the **[DiT model](https://github.com/facebookresearch/DiT)**.

**Layout**

- `CEM.py`: main script for **batched** image generation.
- `models.py`: main **DiT** model code.
- `CEM_utils/`: CEM **module** code (e.g. PEM, DCS).
- `outputs/`: `generations/` stores samples; `priors/` stores priors for error modeling; `visualizations/` holds plots and other visualization outputs.

**Acceleration**

- *Prior Error Modeling*. Use the code below in random generation to build cache-interval error priors. Finally, `priors.pt` is found in folder [`DiT/outputs/priors`](DiT/outputs/priors).  
We have already provided the generated files in this path, which you can use directly.
  ```
  # generate priors of random samples
  bash DiT/PEM.sh
  # Aggregate samples and save prior files
  python DiT/CEM_utils/PEM_module.py
  ```
- *Dynamic Caching Strategy*. We embed this function before batch generation. If you want to test it separately, you can run [`DiT/CEM_utils/DCS_module.py`](DiT/CEM_utils/DCS_module.py) directly.
- *Plug-and-Play Deployment*. We replace the cache intervals in the original method with the optimized cache intervals. This batch generation is a benchmark of 50,000 images from ImageNet categories:
  ```
  bash CEM.sh
  ```

### 2.2 Text-to-Image Models (FLUX.1-dev)

**Plug-in on TeaCache**

- We take **[TeaCache](https://github.com/ali-vilab/TeaCache)** as an example.

**Environment**

- Base: CUDA 12.4 + PyTorch 2.6.0 + torchvision 0.21.0 + diffusers 0.32.0
- Checkpoints: Use a recent `diffusers` build with **FLUX.1-dev** support and [Hugging Face](https://huggingface.co/docs/huggingface_hub/quick-start#login) to access [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev).

**Layout**

- `CEM.py`: main **batched** T2I generation scripts.
- `teacache_flux.py`: teacache baseline.
- `CEM_utils/`: PEM and DCS code.
- `outputs/`: `generations/`, `priors/`, and `visualizations/` (same roles as in §2.1).

**Acceleration**

- *Prior Error Modeling*: Run random T2I (we generate random prompts from GPT) to fit cache-interval priors, and the aggregated file is `priors.pt` under [`FLUX/outputs/priors`](FLUX/outputs/priors), or use the files we have already generated.
  ```
  # generate priors of random samples
  bash FLUX/PEM.sh
  # Aggregate samples and save prior files
  python FLUX/CEM_utils/PEM_module.py
  ```
- *Dynamic Caching Strategy*: Invoked before batch generation; optional standalone test: [`FLUX/CEM_utils/DCS_module.py`](FLUX/CEM_utils/DCS_module.py).
  > Additional insight: As can be seen from the visualization, the model has special stages that can self-correct errors, and the cache interval budget for these stages can be set separately.
- *Plug-and-Play Deployment*: We batch generated the data on drawbench to verify the acceleration effect.
  ```
  bash FLUX/CEM.sh
  ```

### 2.3 Text-to-Video Models (Wan2.1)

**Plug-in on TaylorSeer**

- We take **[TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer)** as an example.

**Environment**

- Base: CUDA 12.4 + PyTorch 2.8.0 + torchvision 0.23.0
- Installation: The specific environment is as follows: [`Wan21/requirements.txt`](Wan21/requirements.txt)
- Checkpoints: place official [Wan2.1](https://github.com/Wan-Video/Wan2.1) T2V [weights](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B) locally.

**Layout**

- `CEM.py`: main **batched** T2V scipts.
- `CEM_utils/`: PEM and DCS code.
- `outputs/`: `generations/`, `priors/`, and `visualizations/`.

**Acceleration**

- *Prior Error Modeling*: Run **random T2V** (e.g. prompts from `random_GPT.json`) to fit cache-interval priors; the merged file is `priors.pt` under [`Wan21/outputs/priors`](Wan21/outputs/priors), or use our pre-bundled priors.
  ```
  # generate priors of random samples
  bash Wan21/PEM.sh
  # Aggregate samples and save prior files
  python Wan21/CEM_utils/PEM_module.py
  ```
- *Dynamic Caching Strategy*: Invoked before batch generation; optional standalone: [`Wan21/CEM_utils/DCS_module.py`](Wan21/CEM_utils/DCS_module.py).
- *Plug-and-Play Deployment*: Batched T2V with the optimized cache schedule (edit `ERROR_ROOT` / `OUT_ROOT` and hyper-parameters in the script, then run):
  ```
  bash Wan21/CEM.sh
  ```

---

## :clap: Acknowledgement

Many thanks to the open-source projects of outstanding visual generative models such as [DiT](https://github.com/facebookresearch/DiT), [FLUX](https://github.com/black-forest-labs/flux), and [Wan](https://github.com/Wan-Video/Wan2.1). Our code is based on previous caching acceleration work using [DuCa](https://github.com/shenyi-z/duca), [TeaCache](https://github.com/ali-vilab/TeaCache), and [TaylorSeer](https://github.com/Shenyi-Z/TaylorSeer).

## :dart: Citation

If you find this project useful, please consider citing:

```
@inproceedings{
shao2026plugandplay,
title={Plug-and-Play Fidelity Optimization for Diffusion Transformer Acceleration via Cumulative Error Minimization},
author={Tong Shao and Yusen Fu and Guoying Sun and Jingde Kong and Zhuotao Tian and Jingyong Su},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=pt4iKnAm0M}
}
```
