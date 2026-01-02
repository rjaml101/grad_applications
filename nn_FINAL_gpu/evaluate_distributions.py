

import numpy as np
import os, sys
import torch

from utils.evals import sliced_wasserstein
from inverse_samplers.neural.models import InverseSamplerNet




def sample_from_model(model, num_dims, n_samples=100_000):
    model.eval()
    with torch.no_grad():
        # Feed in random uniform samples
        u = torch.rand(n_samples, num_dims)
        samples = model(u).numpy()
    return samples



def count_out_of_bounds(samples, g_val=0.5, epsilon_val=1, alpha_val=1):
   
    print(samples.ndim)
    if samples.ndim > 1:
        num_dims = samples.shape[1]
    else: 
        num_dims = 1
        samples = np.expand_dims(samples, axis=-1)
        print(samples.shape)
    bounds_cos_theta = (-1,1)
    bounds_phi = (0, 2*np.pi)
    bounds_lambda = (-1, 1)
    bounds_p = (-1, 1)

    sampling_bounds = [
            bounds_cos_theta, bounds_phi, bounds_lambda, bounds_p]
    sampling_bounds = sampling_bounds[:num_dims]
    out_of_bounds_counts = [
        np.sum((samples[:, dim] < lower) | (samples[:, dim] > upper))
        for dim, (lower, upper) in enumerate(sampling_bounds)
    ]

    return out_of_bounds_counts


if __name__ == "__main__":

    # NUMBER OF DIMENSIONS
    num_dims = 4

    # Conditioning Parameters
    g_val = 0.99
    epsilon_val = 1
    alpha_val = 10
    print(f"{g_val=}")
    print(f"{epsilon_val=}")
    print(f"{alpha_val=}")

    # Settings to be dealt with later (Grid/MLP)
    mode_idx = 0
    points_per_grid = 60 # 1, 4, 24, 32, 60, 80, 235, 590


    # Import appropriate grid-interpolation code
    if num_dims==1:
        from inverse_samplers.grid_interp.grid_samplers.grid_sampler_d1 import *
    elif num_dims==4:
        from inverse_samplers.grid_interp.grid_samplers.grid_sampler_d4 import *



    # Ground Truth Sample Set from INDEPENDENT/DISJOINT Rejection sampling
    dir_inverse_samplers = "inverse_samplers"
    dir_ground_truth = "ground_truth"
    dir_gt_data = "gt_data"

    gt_filename = f"g{g_val}_epsilon{epsilon_val}_alpha{alpha_val}"
    dir_rej_ground_truth = os.path.join(dir_inverse_samplers, dir_ground_truth, dir_gt_data, gt_filename)
    filepath_rej_gt = os.path.join(dir_rej_ground_truth, 
        f"d{num_dims}_sample_set_rej_gt.npy") 
    print("GROUND TRUTH FILEPATH:", filepath_rej_gt)
    samples_gt = np.load(filepath_rej_gt)
    print("Ground Truth (Rejection Sampling):", samples_gt.shape)





    # Compare Ground Truth with Grid, MLP+Fourier, or Normalizing Flow
    modes = ["grid_interp", "mlp"]
    mode_comp = modes[mode_idx]
    print(mode_comp)
    dir_current = os.path.dirname(__file__)
    samples_test = None


    num_samples_test = len(samples_gt)
    if mode_comp == "grid_interp":
        
        # Import Grid Multilinear Interpolation and Compare
        print(f"{points_per_grid=}") 

        num_samples_grid = num_samples_test
        grid_data_path = "/home/ubuntu/Importance_Sampling/nn_FINAL_gpu/inverse_samplers/grid_interp/grid_data" 
        dir_g_value = f"g{g_val}"
        dir_epsilon_value = f"epsilon{epsilon_val}"
        dir_alpha_value = f"alpha{alpha_val}"
        dir_full_value = dir_g_value + "_" + dir_epsilon_value + "_" + dir_alpha_value
        grid_data_file = f"grid_seq_d{num_dims}_{points_per_grid}_p2.npy"
        grid_data_file = os.path.join(grid_data_path, dir_full_value, grid_data_file)
        print(grid_data_file)

        grid_interp = np.load(grid_data_file) # actual phase function values for grid
        samples_grid = multilinear_grid_interp(grid_interp, num_samples_grid, g_val=g_val, epsilon_val=epsilon_val, alpha_val=alpha_val)

        samples_test = samples_grid
        print(samples_test.shape)
        

    elif mode_comp == "mlp":

        loss_type = "mmd" # "mmd", "mse", "swd", etc... 
        models_trained_path = "/home/ubuntu/Importance_Sampling/nn_FINAL_gpu/inverse_samplers/neural/models_trained"

        print(f"{loss_type=}") 
        filename_neural = f"d{num_dims}_{mode_comp}_{loss_type}.pth"
        dir_g_value = f"g{g_val}"
        dir_epsilon_value = f"epsilon{epsilon_val}"
        dir_alpha_value = f"alpha{alpha_val}"
        dir_full_value = dir_g_value + "_" + dir_epsilon_value + "_" + dir_alpha_value
        filepath_neural_net = os.path.join(models_trained_path, dir_full_value, filename_neural)
        print(f"{filepath_neural_net=}")

        # Load model path
        num_frequencies = 10 # 10 is default
        print(f"{num_frequencies=}")
        model = InverseSamplerNet(num_dims=num_dims, num_frequencies=num_frequencies, g_val=g_val, epsilon_val=epsilon_val, alpha_val=alpha_val)
        model.load_state_dict(torch.load(filepath_neural_net))
        model.eval()


        n_samples_mlp = num_samples_test
        samples_mlp_fourier = sample_from_model(model, num_dims=num_dims, n_samples=n_samples_mlp)
        samples_test = samples_mlp_fourier
        print(samples_mlp_fourier.shape)





    ##################################################################
    
    # Print metrics for Ground Truth (Rejection Sampled)
    print(type(samples_gt))
    print(f"{samples_gt.shape=}")
    print(samples_gt)
    print(f"{count_out_of_bounds(samples_gt, g_val, epsilon_val, alpha_val)=}")

    # Print metrics for Testing Distribution (Grid or MLP)
    print(type(samples_test))
    print(f"{samples_test.shape=}")
    print(samples_test)
    print("Shape-->", samples_test.shape)
    print(f"{count_out_of_bounds(samples_test, g_val, epsilon_val, alpha_val)=}")

    # Compute Evaluation Metrics including Sliced Wasserstein Distance (SWD)
    wstn_dist = sliced_wasserstein(samples_test, samples_gt)
    print(f"{wstn_dist=}")






