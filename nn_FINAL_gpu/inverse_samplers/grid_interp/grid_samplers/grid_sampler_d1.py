import numpy as np
import torch
import os, sys
import torch
import time

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def d1_histogram_sampler(u, grid_edges, cdf):

    u = u.clone().detach().to(device, dtype=torch.float32)
    grid_edges = grid_edges.clone().detach().to(device, dtype=torch.float32)
    cdf = torch.tensor(cdf, dtype=torch.float32).to(device)


    # Find the upper bin index for each u (returns values in [1, N])
    bin_idx = torch.searchsorted(cdf, u, right=True)

    # Get corresponding CDF bounds
    cdf_lower = cdf[bin_idx - 1]
    cdf_upper = cdf[bin_idx]

    # Get grid x-bounds for interpolation
    x_lower = grid_edges[bin_idx - 1]
    x_upper = grid_edges[bin_idx]

    # Linear interpolation factor
    denom = (cdf_upper - cdf_lower).clamp(min=1e-8)
    t = (u - cdf_lower) / denom

    # Final interpolated sample
    samples = x_lower + t * (x_upper - x_lower)
    return samples 



def multilinear_grid_interp(grid_pdf, num_samples, g_val=0.4, epsilon_val=1.0, alpha_val=1.0):

    num_dims = len(grid_pdf.shape)
    assert num_dims == 1, "This version supports only 1D grids."

    print(f"{g_val=}")
    print(f"{epsilon_val=}")
    print(f"{alpha_val=}")

    BOUNDS = [(-1,1)]
    num_dims = grid_pdf.ndim
    grid_pdf_shape = grid_pdf.shape

    # Generate random uniform samples u from [0,1]^num_dims, and grid edges 
    uniform_samples = torch.rand(num_samples, num_dims)  # [u1] for 1D sampling
    grid_edges_cos_theta = torch.linspace(-1.0, 1.0, grid_pdf_shape[0]+1)
    sampling_vars = [grid_edges_cos_theta]

    # Compute CDF of the given PDF
    grid_cdf_flat = np.cumsum(grid_pdf)
    grid_cdf_flat /= grid_cdf_flat[-1]
    grid_cdf_flat = np.insert(grid_cdf_flat, 0, 0) # prepend a zero
    print(grid_cdf_flat)


    # Potentially store generated 50K samples for ease of loading/comparison
    cos_theta_samples = d1_histogram_sampler(
        uniform_samples, grid_edges_cos_theta, grid_cdf_flat)
    cos_theta_samples = np.array(cos_theta_samples.cpu().detach().numpy()).transpose()[0]
    print(cos_theta_samples.shape)
    
    # Stack samples in list
    out_of_bounds = [0] * num_dims
    out_of_bounds_samples = []
    for cos_theta in cos_theta_samples:
        # print(cos_theta)
        dim_current = 0
        if not (BOUNDS[dim_current][0] <= cos_theta <= BOUNDS[dim_current][1]):
            out_of_bounds[0] += 1
            out_of_bounds_samples.append(cos_theta)
            # print(cos_theta)
    print(out_of_bounds)

    # For 1D output, expand dimensions to output range
    sampling_interp = np.expand_dims(cos_theta_samples, axis=-1)
    return sampling_interp




