import numpy as np
import mitsuba as mi
import drjit as dr
import os, time, sys
from pathlib import Path
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from phase_functions.phase_functions import *
from utils.file_info import print_file_size_info



def array_sampling_variables(grid_num_samples : tuple):
    
    # Sampling variables
    max_N = 4
    num_dims = len(grid_num_samples)
    sampling_N = max_N * [0]
    for dim in range(num_dims):
        sampling_N[dim] = grid_num_samples[dim]
    
    N_cos_theta, N_phi, N_lambda, N_pol = sampling_N
    cos_theta_vals = torch.linspace(-1, 1, N_cos_theta)
    phi_vals = torch.linspace(0, 2*torch.pi, N_phi) 
    lambda_vals = torch.linspace(0, 1, N_lambda)
    pol_vals = torch.linspace(-1, 1, N_pol) 

    # Create and return coordinate tensor
    sampling_vars = [cos_theta_vals, phi_vals, lambda_vals, pol_vals]
    sampling_vars_selected = sampling_vars[:num_dims]
    mesh = torch.meshgrid(*sampling_vars_selected, indexing='ij')
    coord_tensor = torch.stack(mesh, axis=-1)
    return coord_tensor.to(device)


def generate_phase_func_grid(grid_num_samples : tuple, g_val, epsilon_val, alpha_val):

    # Loop through each dimension
    num_dims = len(grid_num_samples)
    grid_main = torch.empty(grid_num_samples, dtype=torch.float32)
    g_val = torch.tensor([g_val]).to(device)
    epsilon_val = torch.tensor([epsilon_val]).to(device)
    alpha_val = torch.tensor([alpha_val]).to(device)

    sampling_coord_tensor = array_sampling_variables(grid_num_samples)
    phase_function_map = {
        1: henyey_greenstein_torch, # first test structured vs unstructured for this phase function
        4: hg_polar_azimuth_lambda_pol_4d_torch # change to 4D function, and use normalizing flows
    }

    # Get the corresponding function
    if num_dims in phase_function_map:
        func = phase_function_map[num_dims]
    else: raise ValueError(f"No function defined for {num_dims}-dimensional arrays.")

    # Construct array, iterate over all indices, and call the function
    start_time = time.time() 
    coords_flat = sampling_coord_tensor.reshape(-1, sampling_coord_tensor.shape[-1])  # shape: [N_total, D]
    with torch.no_grad():
        args_main = [coords_flat[:, i] for i in range(coords_flat.shape[1])]
        if num_dims == 1:
            args_const = [g_val]
        else: args_const = [g_val, epsilon_val, alpha_val]
        args_full = args_main + args_const
        values = func(*args_full)  # assume func returns tensor
        values = torch.clamp(values, min=0.0)
        grid_main = values.reshape(grid_num_samples)
    end_time = time.time()
    print(end_time - start_time)

    print(type(grid_main))
    grid_main = grid_main.cpu()
    print(type(grid_main))
    grid_main = grid_main.numpy()
    print(type(grid_main))
    return grid_main




if __name__ == "__main__":

        
    # Define dimensions and number of samples per dimension (50-100 MB imposed limit)
    num_dims = 4

    # Define sampling range of input arguments
    g_val = 0.99 # 0.4, 0.95, 0.99
    epsilon_val = 0.5 # 0.5, 1
    alpha_val = 10 # 0.1, 1, 10

    # Loop through generation of grids from 1, 4, 24, 60 in 4D
    samples_per_dim_counts = [1, 4, 24, 60]
    for samples_per_dim in samples_per_dim_counts: 
    
        grid_num_samples = (samples_per_dim,) * num_dims
        print(f"{grid_num_samples=}")
        
        grid_name = f"grid_seq_d{num_dims}_{samples_per_dim}_p2.npy"
        dir_current = os.path.dirname(__file__)
        dir_grid_data = "grid_data"
        
        dir_g_value = f"g{g_val}"
        dir_epsilon_value = f"epsilon{epsilon_val}"
        dir_alpha_value = f"alpha{alpha_val}"
        dir_full_value = dir_g_value + "_" + dir_epsilon_value + "_" + dir_alpha_value
        grid_seq_path = os.path.join(dir_current, dir_grid_data, dir_full_value, grid_name)

        start_time = time.time()
        grid_seq = generate_phase_func_grid(grid_num_samples, g_val=g_val, epsilon_val=epsilon_val, alpha_val=alpha_val)
        print("Total Time to Generate:", time.time() - start_time)
        np.save(grid_seq_path, grid_seq)
        print_file_size_info(grid_seq_path)
        print(f"{g_val=}")
        print(f"{epsilon_val=}")
        print(f"{alpha_val=}")



