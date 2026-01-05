


## 4D Stylized Volumetric Scattering via Neural Importance Sampling

- Explored a **stylized (non-physical) 4D volumetric scattering function** that *jointly* samples:
  - Outgoing direction (2D)
  - Wavelength (1D)
  - Scattering distance (1D) [Hence the non-physical behavior]
- Started with a synthetic/proxy scattering function, so that I could later extend this neural sampler to different scattering functions with **similar levels of complexity/entanglement**.
- Compared **4D grid-based sampling** vs. a **4D neural inverse transform sampler**.
- For sufficiently complex / fluctuating scattering functions:
  - The neural sampler achieved **~3× lower Sliced Wasserstein Distance** to ground truth.
  - Used **~33× less memory** (1.5 MB vs. 50 MB).
- Results empirically show that **neural importance sampling clearly outperforms grids** in high-dimensional, entangled sampling spaces.
- Project is paused due to renderer integration complexity and limited support for joint sampling in modern renderers.