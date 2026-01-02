import numpy as np
import os
import torch
import time

from phase_functions.phase_functions import *


def rejection_tight_upper_bound(num_dims, g_val=0.5, epsilon_val=1.0, alpha_val=1.0):

    # g value that returns maximum of HG
    g_val = abs(g_val) # make g_val positive to calculate maximum
    epsilon_val = abs(epsilon_val) # make epsilon positive to calculate maximum
    alpha_val = 0 # temporarily shut off alpha, maximum is covered by cos(phi=0)

    if num_dims == 1:
        cos_theta_argmax = torch.tensor(1) 
        max_pdf = henyey_greenstein_torch(cos_theta_argmax, g_const=g_val)
    elif num_dims == 4:
        cos_theta_argmax = torch.tensor(1)
        phi_argmax = torch.tensor(0)
        lambda_argmax = torch.tensor(1)
        pol_argmax = torch.tensor(1)
        alpha_argmax = torch.tensor(alpha_val)
        max_pdf = hg_polar_azimuth_lambda_pol_4d_torch(cos_theta_argmax, phi_argmax, 
            lambda_argmax, pol_argmax, g_const=g_val, epsilon_const=epsilon_val, alpha_const=alpha_val)

    return max_pdf



def generate_rejection_samples_nd(n_samples, num_dims, max_pdf, g_val=0.5, epsilon_val=1, alpha_val=1, device='cuda', seed=0):
   
    np.random.seed(i)
    COS_THETA_RANGE = (-1.0, 1.0)
    PHI_RANGE = (0.0, 2 * torch.pi)
    LAMBDA_RANGE = (0, 1)
    POL_RANGE = (-1, 1)

    start_time = time.time()
    samples = []
    batch_size = 1_000_000

    while len(samples) < n_samples:
        cos_theta = torch.empty(batch_size, device=device).uniform_(*COS_THETA_RANGE)
        phi = torch.empty(batch_size, device=device).uniform_(*PHI_RANGE)
        wavelength = torch.empty(batch_size, device=device).uniform_(*LAMBDA_RANGE)
        pol = torch.empty(batch_size, device=device).uniform_(*POL_RANGE)
        sampling_vars = [cos_theta, phi, wavelength, pol]
        sampling_vars = sampling_vars[:num_dims]

        if num_dims == 1:
            pdf = hg_polar_1d(cos_theta, g_const=g_val)
        elif num_dims == 4:
            pdf = hg_polar_azimuth_lambda_pol_4d_torch(cos_theta, phi, wavelength, pol, 
                        g_const=g_val, epsilon_const=epsilon_val, alpha_const=alpha_val)

        accept = torch.rand(batch_size, device=device) < (pdf / max_pdf)
        accepted_samples = torch.stack(sampling_vars, dim=1)[accept]
        samples.append(accepted_samples)

        if sum(s.shape[0] for s in samples) >= n_samples:
            break

    result = torch.cat(samples, dim=0)[:n_samples]
    end_time = time.time()
    return result.cpu(), end_time - start_time


if __name__ == "__main__":

    num_dims = 4

    g_val=0.99
    epsilon_val=0.5
    alpha_val=10

    max_pdf = rejection_tight_upper_bound(num_dims=num_dims, g_val=g_val, epsilon_val=epsilon_val, alpha_val=alpha_val)  # Precomputed tight bound
    print(f"{max_pdf=}")
    print(f"{g_val=}")
    print(f"{epsilon_val=}")
    print(f"{alpha_val=}")

    # Measure times for various sample sizes
    total_samples = 1_000_000 + 50_000
    batch_size = 50_000
    #batch_size = total_samples
    num_batches = int(total_samples / batch_size)
    sample_sizes = [batch_size] * num_batches
    total_time_start = time.time() 
    
    sample_batches = []
    for i, size in enumerate(sample_sizes):
        print(i, size)
        sample_batch, duration = generate_rejection_samples_nd(
                size, num_dims=num_dims, max_pdf=max_pdf, g_val=g_val, epsilon_val=epsilon_val, alpha_val=alpha_val, seed=i)
        print(sample_batch.shape)
        print(f"{size:,} samples generated in {duration:.2f} seconds") 
        sample_batches.append(sample_batch)
    print(f"1M samples generated in {time.time() - total_time_start:.2f} seconds")
    samples_generated = torch.cat(sample_batches, dim=0)
    print(samples_generated.shape)


    # SUBSAMPLE ONE proportional rejection sampling subset, and separate it out as the ground truth
    samples_np = samples_generated.cpu().detach().numpy()
    subsample_size = 50_000
    indices = np.arange(len(samples_np))
    np.random.shuffle(indices)

    # Split into 50K and 1M and take the corresponding subsets
    subsample_indices = indices[:subsample_size]
    remaining_indices = indices[subsample_size:]
    subsample_50k = samples_np[subsample_indices]
    remainder_1m = samples_np[remaining_indices]


    # SAVE BOTH FILES SEPARATELY
    dir_current = os.path.dirname(__file__)
    dir_gt_data = "gt_data"
    dir_g_value = f"g{g_val}"
    dir_epsilon_value = f"epsilon{epsilon_val}"
    dir_alpha_value = f"alpha{alpha_val}"
    #dir_file_full = dir_g_value
    dir_file_full = dir_g_value + "_" + dir_epsilon_value + "_" + dir_alpha_value
    
    # Store 1M main training set
    filename_gt_data_main = f"d{num_dims}_sample_set_rej.npy"
    filepath_gt_data_main = os.path.join(dir_current, dir_gt_data, dir_file_full, filename_gt_data_main)
    print(filename_gt_data_main)
    np.save(filepath_gt_data_main, remainder_1m)
    samples_main_set = np.load(filepath_gt_data_main)
    print(samples_main_set.shape)

    # Store 50K representative subsample as Ground Truth
    filename_gt_data_gt = f"d{num_dims}_sample_set_rej_gt.npy"
    filepath_gt_data_gt = os.path.join(dir_current, dir_gt_data, dir_file_full, filename_gt_data_gt)
    print(filepath_gt_data_gt)
    np.save(filepath_gt_data_gt, subsample_50k)
    samples_gt_set = np.load(filepath_gt_data_gt)
    print(samples_gt_set.shape)



