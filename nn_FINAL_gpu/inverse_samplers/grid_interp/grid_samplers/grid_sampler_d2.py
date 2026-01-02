import numpy as np
import torch
import os, sys
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
def load_grid_data(num_dims, points_per_dim, root_dir=None):
    if root_dir == None:
        root_dir = os.path.dirname(__file__)
    dir_grid_data = os.path.join(root_dir, "grid_data")
    grid_dims_string = f"_{points_per_dim}" * num_dims
    filepath_grid_data = os.path.join(dir_grid_data, 
        f"grid_seq_d{num_dims}{grid_dims_string}.npy")
    grid_interp = np.load(filepath_grid_data)
    return grid_interp
'''


def load_grid_data(num_dims, points_per_dim, root_dir=None, g_val=None):
    if root_dir == None:
        root_dir = os.path.dirname(__file__)
    dir_grid_data = os.path.join(root_dir, "grid_data")
    if g_val != None:
        dir_grid_data = os.path.join(dir_grid_data, f"g_{g_val}")
    grid_dims_string = f"_{points_per_dim}" * num_dims
    grid_filename = f"grid_seq_d{num_dims}{grid_dims_string}.npy"
    filepath_grid_data = os.path.join(dir_grid_data, 
        f"grid_seq_d{num_dims}{grid_dims_string}.npy")           
    print(filepath_grid_data)
    grid_interp = np.load(filepath_grid_data)
    return grid_interp



def compute_cdf(pdf: torch.Tensor) -> torch.Tensor:
    """
    Compute the CDF from a given PDF along the last dimension, prepending a zero.
    """
    pdf = pdf / pdf.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    cdf = torch.cumsum(pdf, dim=-1)
    zero = torch.zeros_like(cdf[..., :1])
    return torch.cat([zero, cdf], dim=-1)  # [..., N+1]



def sample_from_cdf(cdf: torch.Tensor, edges: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Inverse transform sampling from a CDF.

    cdf: [N+1] or [batch, N+1]
    edges: [N+1]
    u: [num_samples] or [batch_size] uniform samples in [0, 1]

    Returns:
        samples: [num_samples]
    """
    device = cdf.device
    N = cdf.shape[-1] - 1

    if cdf.ndim == 1:
        # 1D case: marginal
        idx_upper = torch.searchsorted(cdf, u, right=True)
        idx_upper = torch.clamp(idx_upper, 1, N)
        idx_lower = idx_upper - 1

        cdf_lower = cdf[idx_lower]
        cdf_upper = cdf[idx_upper]
        edge_lower = edges[idx_lower]
        edge_upper = edges[idx_upper]

    elif cdf.ndim == 2:
        # 2D case: batch of conditional CDFs
        idx_upper = torch.searchsorted(cdf, u.unsqueeze(-1), right=True).squeeze(-1)
        idx_upper = torch.clamp(idx_upper, 1, N)
        idx_lower = idx_upper - 1

        # Gather per-batch values
        batch_indices = torch.arange(cdf.shape[0], device=device)

        cdf_lower = cdf[batch_indices, idx_lower]
        cdf_upper = cdf[batch_indices, idx_upper]
        edge_lower = edges[idx_lower]
        edge_upper = edges[idx_upper]
    else:
        raise ValueError(f"Unsupported CDF dimensions: {cdf.shape}")

    t = (u - cdf_lower) / (cdf_upper - cdf_lower + 1e-8)
    return edge_lower + t * (edge_upper - edge_lower)



def sample_2d_pdf_grid(pdf_grid, x_edges, y_edges, u1, u2):
    """
    2D inverse CDF sampling from a joint PDF.

    Args:
        pdf_grid: [Nx, Ny] joint PDF
        x_edges: [Nx+1] bin edges for x
        y_edges: [Ny+1] bin edges for y
        u1: [num_samples] uniform samples for x
        u2: [num_samples] uniform samples for y | x

    Returns:
        samples: [num_samples, 2]
    """
    device = pdf_grid.device
    Nx, Ny = pdf_grid.shape
    num_samples = u1.shape[0]

    # === Normalize joint PDF ===
    pdf_grid = pdf_grid / pdf_grid.sum()

    # === Step 1: Sample X ===
    marginal_pdf = pdf_grid.sum(dim=1)  # [Nx]
    marginal_cdf = compute_cdf(marginal_pdf)  # [Nx+1]
    x_samples = sample_from_cdf(marginal_cdf, x_edges, u1)  # [num_samples]

    # === Step 2: Conditional CDFs for Y ===
    conditional_cdfs = compute_cdf(pdf_grid)  # [Nx, Ny+1]

    # Find X bin index for interpolation
    marginal_cdf_inner = marginal_cdf[1:]  # [Nx]
    idx_upper = torch.searchsorted(marginal_cdf_inner, u1, right=True)
    idx_upper = torch.clamp(idx_upper, 1, Nx - 1)
    idx_lower = idx_upper - 1

    # Interpolation weights
    cdf_low = marginal_cdf[idx_lower]
    cdf_high = marginal_cdf[idx_upper]
    tx = (u1 - cdf_low) / (cdf_high - cdf_low + 1e-8)  # [num_samples]

    # Gather and interpolate conditional CDFs
    cdf_y_l = conditional_cdfs[idx_lower]  # [num_samples, Ny+1]
    cdf_y_h = conditional_cdfs[idx_upper]  # [num_samples, Ny+1]
    cdf_y_interp = (1 - tx.unsqueeze(-1)) * cdf_y_l + tx.unsqueeze(-1) * cdf_y_h
    cdf_y_interp /= cdf_y_interp[:, -1:].clamp(min=1e-8)  # normalize

    # === Step 3: Sample Y from conditional CDF ===
    y_samples = sample_from_cdf(cdf_y_interp, y_edges, u2)  # [num_samples]

    return torch.stack([x_samples, y_samples], dim=-1)  # [num_samples, 2]











def multilinear_grid_interp(grid_interp, num_samples):

    # Full pipeline: computes marginal and conditional CDFs and performs vectorized 2D sequential sampling.

    num_dims = len(grid_interp.shape)
    assert num_dims == 2, "This version supports only 2D grids."

    # Grid edges for x and y (add extra dimension for CDF dimension matching)
    grid_edges_cos_theta = torch.linspace(-1.0, 1.0, grid_interp.shape[0]+1, device=device)
    grid_edges_phi = torch.linspace(0.0, 2 * np.pi, grid_interp.shape[1]+1, device=device)

    # Random uniform samples [u1, u2]
    u1 = torch.rand(num_samples, device=device)
    u2 = torch.rand(num_samples, device=device)

    # Convert grid_interp to tensor on GPU and normalize
    grid_interp = torch.tensor(grid_interp, dtype=torch.float32, device=device)
    grid_interp /= grid_interp.sum()
    # grid_interp /= torch.sum(grid_interp)


    # === TIMING AND SAMPLING ===
    torch.cuda.synchronize()
    start_time = time.time()

    samples = sample_2d_pdf_grid(grid_interp, 
        grid_edges_cos_theta, grid_edges_phi, u1, u2)
    print(type(samples))
    samples = samples.cpu().detach().numpy()
    print(type(samples))

    torch.cuda.synchronize()
    print("Sampling Time Elapsed:", time.time() - start_time)
    print(samples.shape)
    return samples



# MAIN
if __name__ == "__main__":

    num_dims = 2 
    points_per_dim = 27

    # Import Phase Function Grid of specified dimension and resolution
    grid_interp = load_grid_data(num_dims, points_per_dim)
    print(grid_interp.shape)

    # Incorporate multilinear interpolation
    num_samples = 50
    samples = multilinear_grid_interp(grid_interp, num_samples)
    print(samples.shape)




