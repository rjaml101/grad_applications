import mitsuba as mi
import drjit as dr
from drjit.auto import Float
mi_variant = "cuda_ad_rgb" if "cuda_ad_rgb" in mi.variants() else "llvm_rgb"
mi.set_variant(mi_variant)

import numpy as np
import os, sys
import torch

################################
# Evaluation Utilities
################################

# Normalization not necessary, because CDF maps [0,1] to [0,1]
def pdf_grid_to_cdf_flat(pdf_grid):
    cdf_flat = np.cumsum(pdf_grid)
    cdf_flat /= cdf_flat[-1]
    return cdf_flat



################################
# Evaluating Images
################################


def compute_mse_images(img1, img2):
    assert img1.shape == img2.shape 
    diff = img1 - img2
    return np.mean(np.square(diff))

def load_exr_image(image_subdir, image_name):
    bmp = mi.Bitmap(os.path.join(image_subdir, image_name+".exr"))
    image_data = np.array(bmp.convert(pixel_format=mi.Bitmap.PixelFormat.RGB,
                            component_format=mi.Struct.Type.Float32))
    return image_data





################################
# Evaluating Distributions
################################

'''RECONSTRUCT PDF FROM DATA USING HISTOGRAM OR KDE'''




'''COMPUTE EVALUATION METRICS'''

# COMPUTE MSE BETWEEN DISTRIBUTIONS
def compute_mse_distributions(distr_ref, distr_test):
    mse = np.mean((distr_ref - distr_test)**2)
    print(f"Mean Squared Error (MSE): {mse}")
    return mse

# COMPUTE KULLBACK-LIEBLER DIVERGENCE
def compute_kl_divergence(distr_ref, distr_test):
    # kl = np.sum(distr_ref * np.log((distr_ref + epsilon) / (distr_test + epsilon)))
    kl = np.sum(distr_ref * np.log((distr_ref) / (distr_test)))
    # kl = np.sum(distr_test * np.log((distr_test + epsilon) / (distr_ref + epsilon)))
    # kl = np.sum(distr_test * np.log((distr_test) / (distr_ref)))
    print(f"KL Divergence: {kl:.6f}")
    return kl

# [EXTEND THIS TO WASSERSTEIN N-DISTANCE, OR SLICED WASSERSTEIN]
def compute_wasserstein_N_distance(distr_ref, distr_test):

    # WASSERSTEIN-1 DISTANCE 
    def compute_wasserstein_1_distance(num_dims):
        cdf_true = np.cumsum(distr_ref)
        cdf_est = np.cumsum(distr_test)
        wasserstein = np.sum(np.abs(cdf_true - cdf_est)) / len(cdf_true)
        print(f"Wasserstein-{num_dims} Distance: {wasserstein:.6f}")
        return wasserstein

    num_dims = len((np.array(distr_ref)).shape)
    if num_dims == 1:
        wasserstein = compute_wasserstein_1_distance(num_dims)
    else:
        wasserstein = compute_wasserstein_1_distance(num_dims)
    return wasserstein

# JENSEN-SHANNON DIVERGENCE
def compute_jensen_shannon_divergence(distr_ref, distr_test):
    epsilon = 1e-8
    m = 0.5 * (distr_ref + distr_test)
    js = 0.5 * np.sum(distr_ref * np.log((distr_ref + epsilon) / (m + epsilon))) + \
        0.5 * np.sum(distr_test * np.log((distr_test + epsilon) / (m + epsilon)))
    print(f"Jensen-Shannon Divergence: {js:.6f}\n")
    return js





################################
# Evaluating/Comparing Sample Sets
#  Average MCMC Chains and report MEAN, Standard Deviation
################################

# Wasserstein-1


# Sliced Wasserstein (comparing sample "geometry" distribution)


import numpy as np
from scipy.stats import wasserstein_distance

def sliced_wasserstein(samples_a, samples_b, num_projections=100):
    dim = samples_a.shape[1]
    distances = []

    for _ in range(num_projections):
        # Sample a random unit vector in 4D
        direction = np.random.randn(dim)
        direction /= np.linalg.norm(direction)

        # Project both sample sets onto this direction
        proj_a = samples_a @ direction
        proj_b = samples_b @ direction

        # Compute 1D Wasserstein on the projection
        wd = wasserstein_distance(proj_a, proj_b)
        distances.append(wd)

    return np.mean(distances)

# Energy Distance



# MMD






















# KL Divergence between Sample Sets

from scipy.stats import wasserstein_distance

def compute_wasserstein_samples_1D(samples_ref, samples_test):
    return wasserstein_distance(samples_ref, samples_test)


from scipy.spatial.distance import cdist
import numpy as np

def compute_energy_distance(samples_ref, samples_test):
    a = samples_ref
    b = samples_test
    dist_ab = np.mean(cdist(a, b))
    dist_aa = np.mean(cdist(a, a))
    dist_bb = np.mean(cdist(b, b))
    return 2 * dist_ab - dist_aa - dist_bb





import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
#import ot  # POT: Python Optimal Transport

# --- 1D Wasserstein ---
def wasserstein_1D(samples_ref, samples_test):
    return wasserstein_distance(samples_ref.flatten(), samples_test.flatten())

# --- Energy Distance ---
def energy_distance(samples_ref, samples_test):
    dist_ref_test = cdist(samples_ref, samples_test, metric='euclidean')
    dist_ref_ref = cdist(samples_ref, samples_ref, metric='euclidean')
    dist_test_test = cdist(samples_test, samples_test, metric='euclidean')

    m, n = len(samples_ref), len(samples_test)
    term1 = 2.0 * np.sum(dist_ref_test) / (m * n)
    term2 = np.sum(dist_ref_ref) / (m * m)
    term3 = np.sum(dist_test_test) / (n * n)
    return term1 - term2 - term3

# --- MMD ---
def gaussian_kernel(x, y, sigma=1.0):
    x_norm = np.sum(x ** 2, axis=1).reshape(-1, 1)
    y_norm = np.sum(y ** 2, axis=1).reshape(1, -1)
    dist_sq = x_norm + y_norm - 2 * np.dot(x, y.T)
    return np.exp(-dist_sq / (2 * sigma ** 2))

def mmd(samples_ref, samples_test, sigma=1.0):
    K_XX = gaussian_kernel(samples_ref, samples_ref, sigma)
    K_YY = gaussian_kernel(samples_test, samples_test, sigma)
    K_XY = gaussian_kernel(samples_ref, samples_test, sigma)

    m, n = len(samples_ref), len(samples_test)
    mmd_sq = np.sum(K_XX) / (m * m) + np.sum(K_YY) / (n * n) - 2 * np.sum(K_XY) / (m * n)
    return np.sqrt(mmd_sq)

# --- Sinkhorn OT ---
def sinkhorn_wasserstein(samples_ref, samples_test, reg=0.01):
    n = len(samples_ref)
    M = ot.dist(samples_ref, samples_test, metric='euclidean')
    M /= M.max()  # normalize cost
    a = np.ones(n) / n
    b = np.ones(n) / n
    return ot.sinkhorn2(a, b, M, reg)[0]  # return only the value


# === AUTO EVALUATION WRAPPER ===
def evaluate_sample_distributions(samples_ref, samples_test, metric=None, fast=False):
    """
    Automatically evaluate distance between two sample sets using best-suited metric.
    
    Parameters:
        samples_ref: (N, D) array of reference samples (e.g., from MCMC)
        samples_test: (N, D) array of test samples (e.g., from inverse sampler)
        metric: Optional override ['wasserstein', 'energy', 'mmd', 'sinkhorn']
        fast: If True, prefers faster (energy) metric in higher dimensions

    Returns:
        distance_value (float), metric_name (str)
    """
    samples_ref = np.asarray(samples_ref)
    samples_test = np.asarray(samples_test)

    if samples_ref.shape != samples_test.shape:
        raise ValueError("Sample sets must have the same shape")

    dim = samples_ref.shape[1] if samples_ref.ndim > 1 else 1

    if metric is not None:
        metric = metric.lower()

    if metric == 'wasserstein':
        if dim != 1:
            raise ValueError("Wasserstein distance is only implemented for 1D here.")
        dist = wasserstein_1D(samples_ref, samples_test)
        return dist, "wasserstein_1D"

    elif metric == 'energy' or (metric is None and (dim > 1 and fast)):
        dist = energy_distance(samples_ref, samples_test)
        return dist, "energy"

    elif metric == 'mmd' or (metric is None and dim <= 3 and not fast):
        dist = mmd(samples_ref, samples_test, sigma=1.0)
        return dist, "mmd"

    elif metric == 'sinkhorn' or (metric is None and dim > 3 and not fast):
        dist = sinkhorn_wasserstein(samples_ref, samples_test, reg=0.01)
        return dist, "sinkhorn"

    elif metric is None and dim == 1:
        dist = wasserstein_1D(samples_ref, samples_test)
        return dist, "wasserstein_1D"

    else:
        raise ValueError(f"Unsupported or unrecognized metric: {metric}")




'''
# 2D example
ref_samples = np.random.normal(0, 1, (1000, 2))
test_samples = np.random.normal(0.1, 1.1, (1000, 2))

dist_val, metric_used = evaluate_sample_distributions(ref_samples, test_samples)
print(f"{metric_used} Distance: {dist_val:.6f}")

# 4D, force fast mode
ref4d = np.random.uniform(-1, 1, (1000, 4))
test4d = np.random.uniform(-0.8, 1, (1000, 4))

dist_val, metric_used = evaluate_sample_distributions(ref4d, test4d, fast=True)
print(f"{metric_used} Distance (fast): {dist_val:.6f}")
'''
