import numpy as np
import torch
import os, sys
import torch
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



'''
def load_grid_data(num_dims, points_per_dim, root_dir=None, g_val=None):
    if root_dir == None:
        root_dir = os.path.dirname(__file__)
    dir_grid_data = os.path.join(root_dir, "grid_data")
    
    if g_val == "WORKING":
        dir_grid_data = os.path.join(dir_grid_data, "WORKING_DIR")
    elif g_val != None:
        dir_grid_data = os.path.join(dir_grid_data, f"g_{g_val}")

    grid_dims_string = f"_{points_per_dim}" * num_dims
    grid_filename = f"grid_seq_d{num_dims}{grid_dims_string}.npy"
    filepath_grid_data = os.path.join(dir_grid_data, 
        f"grid_seq_d{num_dims}{grid_dims_string}.npy")
    print(filepath_grid_data)
    grid_interp = np.load(filepath_grid_data)
    return grid_interp
'''





def load_grid_data(num_dims, points_per_dim, root_dir=None, g_val=None):
    if root_dir == None:
        root_dir = os.path.dirname(__file__)
    dir_grid_data = os.path.join(root_dir, "grid_data")
    print(dir_grid_data)

    if g_val == "WORKING":
        print(dir_grid_data)
        #dir_grid_data = os.path.join(dir_grid_data, "WORKING_DIR")
        dir_grid_data = "/home/ubuntu/Importance_Sampling/nn_RESULTS_gpu/inverse_samplers/grid_interp/grid_data/WORKING_DIR"
        print(dir_grid_data)
    elif g_val != None:
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
    Normalize and compute cumulative distribution function along the last axis.
    """
    pdf = pdf / pdf.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    cdf = torch.cumsum(pdf, dim=-1)
    zero = torch.zeros_like(cdf[..., :1])
    return torch.cat([zero, cdf], dim=-1)



def batched_searchsorted(cdf: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Row-wise searchsorted: finds indices i in each row of cdf where u < cdf[i]
    Inputs:
        cdf: [B, N] (e.g. CDF values per sample)
        u:   [B]    (one uniform sample per row)
    Returns:
        idx_upper: [B] (index of first cdf > u)
    """
    B, N = cdf.shape
    # Add small noise to avoid equality issues
    u_expanded = u.unsqueeze(-1).expand(B, N)
    mask = cdf >= u_expanded
    idx = mask.float().cumsum(dim=1).eq(1).float().argmax(dim=1)
    return idx



def sample_from_cdf(cdf: torch.Tensor, edges: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    """
    Vectorized inverse CDF sampling.
    
    Args:
        cdf: Tensor of shape [N+1] (marginal) or [B, N+1] (batch of conditionals)
        edges: Tensor of shape [N+1] or [B, N+1] (bin edges)
        u: Tensor of shape [B] (uniform samples in [0,1])

    Returns:
        samples: [B] interpolated samples
    """
    N = cdf.shape[-1] - 1  # number of bins

    if cdf.ndim == 1:
        # --- 1D Marginal case ---
        idx_upper = torch.searchsorted(cdf, u, right=True)
        idx_upper = torch.clamp(idx_upper, 1, N)
        idx_lower = idx_upper - 1

        # Slice CDF + edges
        cdf_lower = cdf[idx_lower]
        cdf_upper = cdf[idx_upper]
        edge_lower = edges[idx_lower]
        edge_upper = edges[idx_upper]

    elif cdf.ndim == 2:
        # --- 2D Batched case (e.g. conditional CDFs) ---
        B = cdf.shape[0]
        device = cdf.device
        idx_upper = torch.searchsorted(cdf, u.unsqueeze(-1), right=True).squeeze(-1)
        idx_upper = torch.clamp(idx_upper, 1, N)
        idx_lower = idx_upper - 1

        batch_indices = torch.arange(B, device=device)

        cdf_lower = cdf[batch_indices, idx_lower]
        cdf_upper = cdf[batch_indices, idx_upper]

        # Handle static vs batched edges
        if edges.ndim == 1:
            edge_lower = edges[idx_lower]
            edge_upper = edges[idx_upper]
        else:
            edge_lower = edges[batch_indices, idx_lower]
            edge_upper = edges[batch_indices, idx_upper]

    else:
        raise ValueError(f"Unsupported CDF dimensions: {cdf.shape}")

    # Linear interpolation
    t = (u - cdf_lower) / (cdf_upper - cdf_lower + 1e-8)
    samples = edge_lower + t * (edge_upper - edge_lower)

    return samples



'''
def sample_3d_pdf_grid(pdf_grid, x_edges, y_edges, z_edges, u1, u2, u3):
    """
    Sequential 3D sampling:
    1. Sample x from marginal PDF
    2. Sample y | x
    3. Sample z | x,y

    Args:
        pdf_grid: [Nx, Ny, Nz] normalized or unnormalized PDF
        x_edges, y_edges, z_edges: each [N+1]
        u1, u2, u3: [B] uniform samples in [0, 1]

    Returns:
        samples: [B, 3]
    """
    device = pdf_grid.device
    Nx, Ny, Nz = pdf_grid.shape
    B = u1.shape[0]

    # === Step 1: Sample x from marginal ===
    pdf_x = pdf_grid.sum(dim=(1, 2))             # [Nx]
    cdf_x = compute_cdf(pdf_x)                   # [Nx+1]
    x_samples = sample_from_cdf(cdf_x, x_edges, u1)

    # === Step 2: Sample y | x ===
    pdf_xy = pdf_grid.sum(dim=2)                 # [Nx, Ny]
    cdf_y_given_x = compute_cdf(pdf_xy)          # [Nx, Ny+1]

    idx_x = torch.searchsorted(cdf_x[1:], u1, right=True).clamp(1, Nx - 1)
    idx_x_l = idx_x - 1
    idx_x_h = idx_x

    cdf_x_l = cdf_x[idx_x_l]
    cdf_x_h = cdf_x[idx_x_h]
    tx = (u1 - cdf_x_l) / (cdf_x_h - cdf_x_l + 1e-8)

    cdf_y_l = cdf_y_given_x[idx_x_l]             # [B, Ny+1]
    cdf_y_h = cdf_y_given_x[idx_x_h]             # [B, Ny+1]
    cdf_y_interp = (1 - tx).unsqueeze(1) * cdf_y_l + tx.unsqueeze(1) * cdf_y_h
    cdf_y_interp /= cdf_y_interp[:, -1:].clamp(min=1e-8)

    y_samples = sample_from_cdf(cdf_y_interp, y_edges, u2)

    # === Step 3: Sample z | x,y ===
    cdf_z_given_xy = compute_cdf(pdf_grid)       # [Nx, Ny, Nz+1]

    idx_y = torch.searchsorted(cdf_y_interp[:, 1:], u2.unsqueeze(-1), right=True).clamp(1, Ny - 1).squeeze(-1)
    idx_y_l = idx_y - 1
    idx_y_h = idx_y

    cdf_y_l = cdf_y_interp[torch.arange(B), idx_y_l]
    cdf_y_h = cdf_y_interp[torch.arange(B), idx_y_h]
    ty = (u2 - cdf_y_l) / (cdf_y_h - cdf_y_l + 1e-8)

    cdf_ll = cdf_z_given_xy[idx_x_l, idx_y_l]     # [B, Nz+1]
    cdf_lh = cdf_z_given_xy[idx_x_l, idx_y_h]
    cdf_hl = cdf_z_given_xy[idx_x_h, idx_y_l]
    cdf_hh = cdf_z_given_xy[idx_x_h, idx_y_h]

    # Bilinear interpolation in x-y
    tx = tx.unsqueeze(1)
    ty = ty.unsqueeze(1)
    cdf_z_interp = (
        (1 - tx) * (1 - ty) * cdf_ll +
        (1 - tx) * ty * cdf_lh +
        tx * (1 - ty) * cdf_hl +
        tx * ty * cdf_hh
    )
    cdf_z_interp /= cdf_z_interp[:, -1:].clamp(min=1e-8)

    z_samples = sample_from_cdf(cdf_z_interp, z_edges, u3)

    return torch.stack([x_samples, y_samples, z_samples], dim=1)
'''





def sample_3d_pdf_grid(pdf_grid, x_edges, y_edges, z_edges, u1, u2, u3):
    """
    3D inverse CDF sampling from a joint PDF using sequential sampling.

    Args:
        pdf_grid: [Nx, Ny, Nz] joint PDF
        x_edges, y_edges, z_edges: bin edges
        u1, u2, u3: uniform samples in [0,1], shape [num_samples]

    Returns:
        samples: [num_samples, 3]
    """
    device = pdf_grid.device
    Nx, Ny, Nz = pdf_grid.shape
    num_samples = u1.shape[0]

    # === Normalize ===
    pdf_grid = pdf_grid / pdf_grid.sum()

    # === Step 1: Sample X from marginal ===
    pdf_x = pdf_grid.sum(dim=(1, 2))  # [Nx]
    cdf_x = compute_cdf(pdf_x)        # [Nx+1]
    x_samples = sample_from_cdf(cdf_x, x_edges, u1)  # [num_samples]
    print(cdf_x.sum(dim=-1).min(), cdf_x.sum(dim=-1).max())

    # === Step 2: Sample Y | X ===
    pdf_xy = pdf_grid.sum(dim=2)  # [Nx, Ny]
    cdf_y_given_x = compute_cdf(pdf_xy)  # [Nx, Ny+1]

    # Interpolate over cdf_y_given_x using u1
    idx_x_upper = torch.searchsorted(cdf_x[1:], u1, right=True).clamp(1, Nx - 1)
    idx_x_lower = idx_x_upper - 1

    cdf_x_low = cdf_x[idx_x_lower]
    cdf_x_high = cdf_x[idx_x_upper]
    tx = (u1 - cdf_x_low) / (cdf_x_high - cdf_x_low + 1e-8)

    cdf_y_l = cdf_y_given_x[idx_x_lower]  # [num_samples, Ny+1]
    cdf_y_h = cdf_y_given_x[idx_x_upper]
    cdf_y_interp = (1 - tx.unsqueeze(-1)) * cdf_y_l + tx.unsqueeze(-1) * cdf_y_h
    cdf_y_interp /= cdf_y_interp[:, -1:].clamp(min=1e-8)
    print(cdf_y_interp.sum(dim=-1).min(), cdf_y_interp.sum(dim=-1).max())
    y_samples = sample_from_cdf(cdf_y_interp, y_edges, u2)  # [num_samples]

    # === Step 3: Sample Z | X,Y ===
    cdf_z_given_xy = compute_cdf(pdf_grid)  # [Nx, Ny, Nz+1]
    #cdf_z_given_xy = compute_cdf(pdf_grid, dim=2)  # [Nx, Ny, Nz+1]

    # Get interpolation indices for x and y
    idx_y_upper = batched_searchsorted(cdf_y_interp[:, 1:], u2).clamp(1, Ny - 1)
    #idx_y_upper = torch.searchsorted(cdf_y_interp[:, 1:], u2, right=True).clamp(1, Ny - 1)
    idx_y_lower = idx_y_upper - 1

    cdf_y_low = cdf_y_interp.gather(1, idx_y_lower.unsqueeze(-1)).squeeze(-1)
    cdf_y_high = cdf_y_interp.gather(1, idx_y_upper.unsqueeze(-1)).squeeze(-1)
    ty = (u2 - cdf_y_low) / (cdf_y_high - cdf_y_low + 1e-8)

    # Bilinear interpolation of conditional CDFs for z
    cdf_00 = cdf_z_given_xy[idx_x_lower, idx_y_lower]  # [num_samples, Nz+1]
    cdf_01 = cdf_z_given_xy[idx_x_lower, idx_y_upper]
    cdf_10 = cdf_z_given_xy[idx_x_upper, idx_y_lower]
    cdf_11 = cdf_z_given_xy[idx_x_upper, idx_y_upper]

    tx = tx.unsqueeze(-1)
    ty = ty.unsqueeze(-1)

    cdf_z_interp = (
        (1 - tx) * (1 - ty) * cdf_00 +
        (1 - tx) * ty * cdf_01 +
        tx * (1 - ty) * cdf_10 +
        tx * ty * cdf_11
    )
    cdf_z_interp /= cdf_z_interp[:, -1:].clamp(min=1e-8)
    print(cdf_z_interp.sum(dim=-1).min(), cdf_z_interp.sum(dim=-1).max())
    z_samples = sample_from_cdf(cdf_z_interp, z_edges, u3)  # [num_samples]

    return torch.stack([x_samples, y_samples, z_samples], dim=-1)  # [num_samples, 3]




























def multilinear_grid_interp(grid_interp, num_samples, g_bound_abs, alpha_max):

    # Full pipeline: computes marginal and conditional CDFs and performs vectorized 2D sequential sampling.

    # Ignore "alpha_max"

    num_dims = len(grid_interp.shape)
    assert num_dims == 3, "This version supports only 3D grids."
    #g_bound_abs = 0.99
    #g_bound_abs = 0.9
    #g_bound_abs = 0.7
    #g_bound_abs = 0.4
    assert(0.0 < g_bound_abs < 1.0) # ensure |g| < 1 and nonzero

    # Grid edges for x and y (add extra dimension for CDF dimension matching)
    grid_edges_cos_theta = torch.linspace(-1.0, 1.0, grid_interp.shape[0]+1, device=device)
    grid_edges_phi = torch.linspace(0.0, 2 * np.pi, grid_interp.shape[1]+1, device=device)
    grid_edges_g = torch.linspace(-g_bound_abs, g_bound_abs, grid_interp.shape[2]+1, device=device)

    # Random uniform samples [u1, u2]
    u1 = torch.rand(num_samples, device=device)
    u2 = torch.rand(num_samples, device=device)
    u3 = torch.rand(num_samples, device=device)

    # Convert grid_interp to tensor on GPU and normalize
    grid_interp = torch.tensor(grid_interp, dtype=torch.float32, device=device)
    grid_interp /= grid_interp.sum()
    # grid_interp /= torch.sum(grid_interp)


    # === TIMING AND SAMPLING ===
    torch.cuda.synchronize()
    start_time = time.time()

    samples = sample_3d_pdf_grid(grid_interp, 
        grid_edges_cos_theta, grid_edges_phi, grid_edges_g, u1, u2, u3)
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




