# 4D Entangled Volumetric Scattering via Neural Importance Sampling

## TL;DR

- Explored a **stylized (non-physical) 4D volumetric scattering function** that *jointly* samples:
  - Outgoing direction (2D)
  - Wavelength (1D)
  - Scattering distance (1D) [Hence the non-physical behavior]
- Compared **4D grid-based sampling** vs. a **4D neural inverse transform sampler**.
- For sufficiently complex / fluctuating scattering functions:
  - The neural sampler achieved **~3× lower Sliced Wasserstein Distance** to ground truth.
  - Used **~33× less memory** (1.5 MB vs. 50 MB).
- Results suggest **neural importance sampling clearly outperforms grids** in high-dimensional, entangled sampling spaces.
- Project is **paused** due to renderer integration complexity and limited support for joint sampling in modern renderers.



---

## Overview

This project explores a **stylized, non-photorealistic volumetric scattering model** that jointly samples multiple scattering variables *at once*, rather than independently. The goal was to understand when **traditional grid-based sampling breaks down** in higher-dimensional, entangled scattering spaces—and when **neural inverse transform sampling** becomes a better alternative.

## Motivation

Given the **incoming direction** of a light ray, I wanted to simulate a stylized volumetric scattering effect that jointly samples:

- **Outgoing direction** (2 variables)
- **Wavelength** (1 variable)
- **Scattering distance** (1 variable) [Hence the non-physical behavior]

This forms a **4D entangled volumetric scattering function**.

> Note: Jointly sampling scattering distance is not physically meaningful in a traditional rendering context, since we cannot physically predict the next interaction position. This project intentionally operates in a **stylized / non-photorealistic** domain.

At the same time, I was investigating **neural importance sampling**, with the specific question:

> At what level of complexity does a 4D scattering function become impractical for grid-based sampling and instead require a neural sampler?

## Approach

To study this, I constructed a **synthetic proxy scattering function** with controlled complexity and compared two sampling approaches:

### 1. Neural Inverse Transform Sampler

- **Architecture**:
  - 4D MLP
  - Fourier feature encoding
- **Input**:
  - 4 random uniform samples
- **Output**:
  - 50,000 samples of 4D scattering variables:
    - Outgoing direction
    - Wavelength
    - Non-physical scattering distance

### 2. 4D Grid-Based Sampling

- **Method**: Sequential sampling over 4D grids
- **Output**:
  - 50,000 samples of the same 4D variables

## Evaluation

### Metric

- **Sliced Wasserstein Distance (SWD)**
- Computed between:
  - Sampled 4D distributions
  - Rejection-sampled **ground truth** 4D distribution

This enabled comparison at the **distribution level**, independent of rendering.

## Results (Partial)

For sufficiently **highly fluctuating** scattering functions:

- The **4D neural inverse transform sampler**:
  - Achieved **~3× lower SWD** to ground truth
  - Used **~1.5 MB** of memory

- The **4D grid-based sampler**:
  - Required **~50 MB** of memory
  - Produced significantly worse distributional accuracy

| Method | Memory | SWD (relative) |
|------|--------|----------------|
| 4D Neural Sampler | ~1.5 MB | 3× better |
| 4D Grid Sampling | ~50 MB | Baseline |

## Limitations & Current Status

This project is **shelved for now** due to:

- The time required to modify or build a renderer capable of exploiting joint sampling.
- Limited support in modern renderers (e.g., Mitsuba 3) for:
  - Jointly sampling outgoing direction, wavelength, and scattering distance
  - Wavelength sampling more than once per path (in spectral rendering)
  - Joint sampling of scattering distance and direction is primarily meaningful in an **artistic / non-photorealistic** context.

