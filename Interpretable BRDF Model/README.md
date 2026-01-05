# Incorporating Semantic Interpretability for Generative BRDF Model

## Overview

This project focuses on **incorporating semantic interpretability into generative models of measured BRDFs**. Previous models can synthesize new materials via free movement in latent space, but they **lack human-interpretable “axes of manipulation”** (e.g., roughness, color, glossiness), limiting their usefulness in graphics applications where **intuitive parameter control** is essential.  

The primary goal is to enable **direct manipulation of interpretable material properties**, creating an intuitive set of “material knobs” to control the appearance of generated BRDF materials.

Some papers informing this project:

- [Sztrajman]: Neural BRDF Representation and Importance Sampling  
- [Harkonen]: GANSpace: Discovering Interpretable GAN Controls  

## Current Plans / Experiments

1. **Data and Pretrained Models**  
   - Use Neural BRDFs (NBRDFs) based on measured MERL BRDF data (code and pretrained models provided by Sztrajman).

2. **Autoencoder Construction**  
   - Build an autoencoder that maps NBRDFs to an **encoded latent space** and reconstructs them back to NBRDFs.  
   - The encoder of this trained autoencoder (rather than training from scratch or using random latent variables) provides a **consistent latent space structure**, guided by reconstruction objectives with image-MSE loss.  
   - This structured latent space **facilitates interpretability** and **stabilizes training** of downstream generative models.

3. **Incorporating Interpretability**  
   - Apply **Principal Component Analysis (PCA)** on the input and/or intermediate latent space(s) of the generative model.  
   - PCA generates **meaningful basis vectors** that allow users to manipulate perceptual material properties (e.g., roughness, color) intuitively.

4. **Overall Setup Options**  

   - **PCA on Autoencoder/Variational Autoencoder latent space**  
     - Does not replace decoder.  
     - Simpler but may produce less creative outputs.  

   - **Replace decoder with a generative model**  
     - **Standard MLP** distinct from the decoder MLP (more expressive than simple reconstruction).  
     - **StyleGAN-like generator**: StyleGAN latent spaces are somewhat linear, which is particularly suitable for PCA (as discussed in Harkonen / GANSpace).

## Future Plans

- **Residual Variables in Latent Space**  
  - PCA identifies the most consequential directions, but less consequential “residual” directions may still control **fine-grained perceptual effects**.

- **Importance Sampling**  
  - **Evaluation distribution**: decoded NBRDF output of the Autoencoder or generative model.  
  - **Sampling distribution**: latent space of the Autoencoder mapped to either:
    - An **analytical distribution** for which importance sampling is easier (as reported in Sztrajman), or  
    - A **neural sampling distribution** for better distribution match.