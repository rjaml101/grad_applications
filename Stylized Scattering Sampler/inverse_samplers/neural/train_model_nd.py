import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from phase_functions.phase_functions import *
from utils.file_info import print_file_size_info
from inverse_samplers.neural.models import *
#from inverse_samplers.mlp_fourier.gen_nn_data import *


# === DEVICE ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# === DATA GENERATION ===
def generate_training_data(num_dims, n_samples=100_000, g_val=0.4, epsilon_val=1, alpha_val=1):
    u = np.random.rand(n_samples, num_dims) 

    dir_gt_data = "/home/ubuntu/Importance_Sampling/nn_FINAL_gpu/inverse_samplers/ground_truth/gt_data"
    dir_g_value = f"g{g_val}"
    dir_epsilon_value = f"epsilon{epsilon_val}"
    dir_alpha_value = f"alpha{alpha_val}"
    dir_full_value = dir_g_value + "_" + dir_epsilon_value + "_" + dir_alpha_value
    filename_gt_data = f"d{num_dims}_sample_set_rej.npy"
    filepath_gt_data = os.path.join(dir_gt_data, dir_full_value, filename_gt_data)
    print(filepath_gt_data)
    samples_full = np.load(filepath_gt_data)
    print(samples_full.shape)

    indices = np.random.choice(len(samples_full), size=n_samples, replace=False)
    samples = samples_full[indices]
    print(samples.shape)

    return u.astype(np.float32), samples.astype(np.float32)

# === MMD LOSS FUNCTION ===
def compute_pairwise_distances(x, y):
    x_norm = (x ** 2).sum(dim=1).view(-1, 1)
    y_norm = (y ** 2).sum(dim=1).view(1, -1)
    dist = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
    return torch.clamp(dist, 0.0, np.inf)

def mmd_loss(x, y, bandwidths=[0.01, 0.1, 1.0]):
    loss = 0
    for bw in bandwidths:
        Kxx = torch.exp(-compute_pairwise_distances(x, x) / (2 * bw))
        Kyy = torch.exp(-compute_pairwise_distances(y, y) / (2 * bw))
        Kxy = torch.exp(-compute_pairwise_distances(x, y) / (2 * bw))
        loss += Kxx.mean() + Kyy.mean() - 2 * Kxy.mean()
    return loss / len(bandwidths)

# === TRAINING ===
def train_inverse_sampler(model, u_train, targets, epochs=50, batch_size=512, loss_type="mmd"):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = mmd_loss if loss_type == "mmd" else nn.MSELoss()

    dataset = torch.utils.data.TensorDataset(
        torch.tensor(u_train).to(device),
        torch.tensor(targets).to(device)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    total_loss_over_epochs = []
    for epoch in range(epochs):
        total_loss = 0
        for u_batch, y_batch in loader:
            optimizer.zero_grad()
            pred = model(u_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(u_batch)
        avg_loss = total_loss / len(u_train)
        total_loss_over_epochs.append(avg_loss)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.6f}")
    return total_loss_over_epochs

# === EVALUATION ===
def eval_true_pdf(cos_theta_grid, phi_grid):
    PDF = hg_polar_azimuth_2d_nonsep(cos_theta_grid, phi_grid)
    PDF /= np.sum(PDF)
    return PDF

def sample_from_model(model, n_samples=100_000):
    model.eval()
    model.to(device)
    with torch.no_grad():
        u = torch.rand(n_samples, 2, device=device)
        samples = model(u).cpu().numpy()
    return samples[:, 0], samples[:, 1]

def compare_model_to_true_pdf(model):
    cos_theta_s, phi_s = sample_from_model(model)
    num_bins = 27
    H, xedges, yedges = np.histogram2d(
        cos_theta_s, phi_s, bins=num_bins,
        range=[[-1, 1], [0, 2*np.pi]], density=True
    )

    x_centers = 0.5 * (xedges[:-1] + xedges[1:])
    y_centers = 0.5 * (yedges[:-1] + yedges[1:])
    xx, yy = np.meshgrid(x_centers, y_centers, indexing='ij')
    true_pdf = eval_true_pdf(xx, yy)

    H /= np.sum(H)
    true_pdf /= np.sum(true_pdf)

    vmin = 0.0
    vmax = true_pdf.max()

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    im1 = plt.imshow(true_pdf, extent=[0, 2*np.pi, -1, 1], aspect='auto', origin='lower',
                     vmin=vmin, vmax=vmax)
    plt.title("True Phase Function PDF")
    plt.xlabel("φ")
    plt.ylabel("cos(θ)")
    plt.colorbar(im1)

    plt.subplot(1, 2, 2)
    im2 = plt.imshow(H, extent=[0, 2*np.pi, -1, 1], aspect='auto', origin='lower',
                     vmin=vmin, vmax=vmax)
    plt.title("Learned Sample Density")
    plt.xlabel("φ")
    plt.ylabel("cos(θ)")
    plt.colorbar(im2)

    plt.tight_layout()
    plt.show()

# === MAIN ===
if __name__ == "__main__":

    # Number of dimensions
    num_dims = 4

    # Specify conditioning parameters
    g_val = 0.99
    epsilon_val = 1
    alpha_val = 1
    print(f"{g_val=}")
    print(f"{epsilon_val=}")
    print(f"{alpha_val=}")

    num_frequencies = 10 # default is 10
    num_train_samples = 50_000 # 100_000, 500_000, 1_000_000
    print(f"{num_frequencies=}")

    u_train, train_targets = generate_training_data(
        num_dims=num_dims, n_samples=num_train_samples, 
        g_val=g_val, epsilon_val=epsilon_val, alpha_val=alpha_val)

    model = InverseSamplerNet(num_dims=num_dims, num_frequencies=num_frequencies, 
        g_val=g_val, epsilon_val=epsilon_val, alpha_val=alpha_val).to(device)

    #num_epochs = 150 (best so far, even if loss hasn't changed)
    num_epochs = 150
    batch_size = 512
    steps_per_epoch = int(num_train_samples / batch_size)
    num_training_steps = num_epochs * steps_per_epoch
    loss_type = "mmd"

    print(f"{loss_type=}")
    print(f"{num_training_steps=}")
    training_time_start = time.time()
    total_loss = train_inverse_sampler(
        model, u_train, train_targets,
        epochs=num_epochs,
        loss_type=loss_type,
        batch_size=batch_size)
    training_time_end = time.time()
    print("Training Time Elapsed:", training_time_end - training_time_start)

    
    model_type = "mlp"
    dir_current = os.path.dirname(__file__)
    dir_models = "models_trained"
    dir_g_value = f"g{g_val}"
    dir_epsilon_value = f"epsilon{epsilon_val}"
    dir_alpha_value = f"alpha{alpha_val}"
    dir_full_value = dir_g_value + "_" + dir_epsilon_value + "_" + dir_alpha_value
    neural_filename = f"d{num_dims}_{model_type}_{loss_type}.pth"
    neural_model_filepath = os.path.join(dir_current, dir_models, dir_full_value, neural_filename)
    print(neural_model_filepath)
    torch.save(model.state_dict(), neural_model_filepath)

    file_size_bytes = print_file_size_info(neural_model_filepath)

