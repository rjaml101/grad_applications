## Incorporating Semantic Interpretability for Generative BRDF Model [MORE DETAILS IN "Interpretable BRDF Model" FOLDER]

- I’m creating a generative model trained on a family of measured BRDFs. Previous models of this kind can synthesize new materials based on free movement within the latent space, but they don’t have human-interpretable “axes of manipulation” (e.g. roughness, color, glossiness), which limits their usefulness in graphics applications, where intuitive parameter manipulation is essential.
- **My primary goal is to incorporate semantic interpretability into the latent space(s) of this generative model, enabling direct manipulation of interpretable material properties, so one can have an intuitive, easily interpretable set of “material” knobs with which to control the appearance of the generated BRDF material.**


## Euler Elastica [MORE DETAILS IN "Euler Elastica" FOLDER]

- The goal was to come up with the proper numerical/iterative method to generate discrete closed 3D Euler Elastic curves, by starting with randomly initialized points in 3D space, and using a simple length minimization method while constraining the Area vector and Volume vector
- **Result: The code now generates closed 3D Euler Elastic curves based on length minimizer with area vector and volume vector constraints (using an Augmented Lagrangian), as opposed to the more complex method of minimizing total bending energy with length and total torsion constraints.**
- Extensions: We can also try to extend this to open curves. This also has implications for splines in CAD



## 4D Stylized Volumetric Scattering via Neural Importance Sampling [CODE AND MORE DETAILS IN "Stylized Scattering Sampler" FOLDER]

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