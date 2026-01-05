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
'''


def compute_cdf(pdf: torch.Tensor) -> torch.Tensor:
    pdf = pdf / pdf.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    cdf = torch.cumsum(pdf, dim=-1)
    zero = torch.zeros_like(cdf[..., :1])
    return torch.cat([zero, cdf], dim=-1)

def batched_searchsorted(cdf: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    B, N = cdf.shape
    u_expanded = u.unsqueeze(-1).expand(B, N)
    mask = cdf >= u_expanded
    idx = mask.float().cumsum(dim=1).eq(1).float().argmax(dim=1)
    return idx




def sample_from_cdf(cdf: torch.Tensor, edges: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
    N = cdf.shape[-1] - 1

    if cdf.ndim == 1:
        idx_upper = torch.searchsorted(cdf, u, right=True).clamp(1, N)
        idx_lower = idx_upper - 1
        cdf_lower = cdf[idx_lower]
        cdf_upper = cdf[idx_upper]
        edge_lower = edges[idx_lower]
        edge_upper = edges[idx_upper]
    elif cdf.ndim == 2:
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


def sample_4d_pdf_grid(pdf_grid, x_edges, y_edges, g_edges, a_edges, u1, u2, u3, u4):
    """
    4D inverse transform sampling from joint PDF using sequential sampling.

    Args:
        pdf_grid: [Nx, Ny, Ng, Na] joint PDF
        x_edges, y_edges, g_edges, a_edges: [N+1] bin edges for each axis
        u1, u2, u3, u4: [num_samples] uniform samples in [0, 1]

    Returns:
        samples: [num_samples, 4] samples from the 4D distribution
    """
    device = pdf_grid.device
    Nx, Ny, Ng, Na = pdf_grid.shape
    num_samples = u1.shape[0]

    # === Normalize joint PDF ===
    pdf_grid = pdf_grid / pdf_grid.sum()

    # === Step 1: Sample X ===
    pdf_x = pdf_grid.sum(dim=(1, 2, 3))           # [Nx]
    cdf_x = compute_cdf(pdf_x)                    # [Nx+1]
    print(cdf_x.sum(dim=-1).min(), cdf_x.sum(dim=-1).max())
    x_samples = sample_from_cdf(cdf_x, x_edges, u1)

    # === Step 2: Sample Y | X ===
    pdf_xy = pdf_grid.sum(dim=(2, 3))             # [Nx, Ny]
    cdf_y_given_x = compute_cdf(pdf_xy)           # [Nx, Ny+1]

    idx_x_upper = torch.searchsorted(cdf_x[1:], u1, right=True).clamp(1, Nx - 1)
    idx_x_lower = idx_x_upper - 1

    tx = (u1 - cdf_x[idx_x_lower]) / (cdf_x[idx_x_upper] - cdf_x[idx_x_lower] + 1e-8)

    cdf_y_l = cdf_y_given_x[idx_x_lower]
    cdf_y_h = cdf_y_given_x[idx_x_upper]
    cdf_y_interp = (1 - tx.unsqueeze(-1)) * cdf_y_l + tx.unsqueeze(-1) * cdf_y_h
    cdf_y_interp /= cdf_y_interp[:, -1:].clamp(min=1e-8)
    print(cdf_y_interp.sum(dim=-1).min(), cdf_y_interp.sum(dim=-1).max())

    y_samples = sample_from_cdf(cdf_y_interp, y_edges, u2)

    # === Step 3: Sample G | X, Y ===
    pdf_xyg = pdf_grid.sum(dim=3)                 # [Nx, Ny, Ng]
    cdf_g_given_xy = compute_cdf(pdf_xyg)         # [Nx, Ny, Ng+1]

    idx_y_upper = batched_searchsorted(cdf_y_interp[:, 1:], u2).clamp(1, Ny - 1)
    idx_y_lower = idx_y_upper - 1

    cdf_y_l = cdf_y_interp[torch.arange(num_samples), idx_y_lower]
    cdf_y_h = cdf_y_interp[torch.arange(num_samples), idx_y_upper]
    ty = (u2 - cdf_y_l) / (cdf_y_h - cdf_y_l + 1e-8)

    cdf_g_00 = cdf_g_given_xy[idx_x_lower, idx_y_lower]
    cdf_g_01 = cdf_g_given_xy[idx_x_lower, idx_y_upper]
    cdf_g_10 = cdf_g_given_xy[idx_x_upper, idx_y_lower]
    cdf_g_11 = cdf_g_given_xy[idx_x_upper, idx_y_upper]

    tx = tx.unsqueeze(-1)
    ty = ty.unsqueeze(-1)

    cdf_g_interp = (
        (1 - tx) * (1 - ty) * cdf_g_00 +
        (1 - tx) * ty * cdf_g_01 +
        tx * (1 - ty) * cdf_g_10 +
        tx * ty * cdf_g_11
    )
    cdf_g_interp /= cdf_g_interp[:, -1:].clamp(min=1e-8)
    print(cdf_g_interp.sum(dim=-1).min(), cdf_g_interp.sum(dim=-1).max())

    g_samples = sample_from_cdf(cdf_g_interp, g_edges, u3)

    # === Step 4: Sample Alpha | X, Y, G ===
    cdf_alpha_given_xyg = compute_cdf(pdf_grid)   # [Nx, Ny, Ng, Na+1]

    idx_g_upper = batched_searchsorted(cdf_g_interp[:, 1:], u3).clamp(1, Ng - 1)
    idx_g_lower = idx_g_upper - 1

    cdf_g_l = cdf_g_interp[torch.arange(num_samples), idx_g_lower]
    cdf_g_h = cdf_g_interp[torch.arange(num_samples), idx_g_upper]
    tg = (u3 - cdf_g_l) / (cdf_g_h - cdf_g_l + 1e-8)

    cdf_a_000 = cdf_alpha_given_xyg[idx_x_lower, idx_y_lower, idx_g_lower]
    cdf_a_001 = cdf_alpha_given_xyg[idx_x_lower, idx_y_lower, idx_g_upper]
    cdf_a_010 = cdf_alpha_given_xyg[idx_x_lower, idx_y_upper, idx_g_lower]
    cdf_a_011 = cdf_alpha_given_xyg[idx_x_lower, idx_y_upper, idx_g_upper]
    cdf_a_100 = cdf_alpha_given_xyg[idx_x_upper, idx_y_lower, idx_g_lower]
    cdf_a_101 = cdf_alpha_given_xyg[idx_x_upper, idx_y_lower, idx_g_upper]
    cdf_a_110 = cdf_alpha_given_xyg[idx_x_upper, idx_y_upper, idx_g_lower]
    cdf_a_111 = cdf_alpha_given_xyg[idx_x_upper, idx_y_upper, idx_g_upper]

    tg = tg.unsqueeze(-1)

    # Trilinear interpolation
    cdf_a_interp = (
        (1 - tx) * (1 - ty) * (1 - tg) * cdf_a_000 +
        (1 - tx) * (1 - ty) * tg * cdf_a_001 +
        (1 - tx) * ty * (1 - tg) * cdf_a_010 +
        (1 - tx) * ty * tg * cdf_a_011 +
        tx * (1 - ty) * (1 - tg) * cdf_a_100 +
        tx * (1 - ty) * tg * cdf_a_101 +
        tx * ty * (1 - tg) * cdf_a_110 +
        tx * ty * tg * cdf_a_111
    )
    cdf_a_interp /= cdf_a_interp[:, -1:].clamp(min=1e-8)
    print(cdf_a_interp.sum(dim=-1).min(), cdf_a_interp.sum(dim=-1).max())

    a_samples = sample_from_cdf(cdf_a_interp, a_edges, u4)

    return torch.stack([x_samples, y_samples, g_samples, a_samples], dim=-1)





def multilinear_grid_interp(grid_interp, num_samples, g_val=0.5, epsilon_val=1, alpha_val=1):

    # Full pipeline: computes marginal and conditional CDFs and performs vectorized 2D sequential sampling.

    num_dims = len(grid_interp.shape)
    assert num_dims == 4, "This version supports only 4D grids."
    print(f"{g_val=}")
    print(f"{epsilon_val=}")
    print(f"{alpha_val=}")

    # Grid edges for x and y (add extra dimension for CDF dimension matching)
    grid_edges_cos_theta = torch.linspace(-1.0, 1.0, grid_interp.shape[0]+1, device=device)
    grid_edges_phi = torch.linspace(0.0, 2 * torch.pi, grid_interp.shape[1]+1, device=device)
    grid_edges_lambda = torch.linspace(0, 1, grid_interp.shape[2]+1, device=device)
    grid_edges_pol = torch.linspace(-1, 1, grid_interp.shape[3]+1, device=device)

    # Random uniform samples [u1, u2]
    u1 = torch.rand(num_samples, device=device)
    u2 = torch.rand(num_samples, device=device)
    u3 = torch.rand(num_samples, device=device)
    u4 = torch.rand(num_samples, device=device)

    # Convert grid_interp to tensor on GPU and normalize
    grid_interp = torch.tensor(grid_interp, dtype=torch.float32, device=device)
    grid_interp /= grid_interp.sum()
    # grid_interp /= torch.sum(grid_interp)


    # === TIMING AND SAMPLING ===
    torch.cuda.synchronize()
    start_time = time.time()

    samples = sample_4d_pdf_grid(grid_interp, grid_edges_cos_theta, 
            grid_edges_phi, grid_edges_lambda, grid_edges_pol, u1, u2, u3, u4)
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




