import numpy as np
import torch
import torch.nn as nn
import os, sys
import matplotlib.pyplot as plt




class FourierFeatureEncoding(nn.Module):
    def __init__(self, in_features, num_frequencies=10):
        super().__init__()
        #self.B = 2 ** torch.arange(num_frequencies) * np.pi
        self.B = 2 ** torch.arange(num_frequencies) * torch.pi
        self.in_features = in_features
        # scale = 1
        # self.B = torch.randn(num_frequencies) * scale
        # self.B = torch.randn((in_features, num_frequencies)) * scale

    def forward(self, x):
        xb = x.unsqueeze(-1) * self.B.to(x.device)
        x_sin = torch.sin(xb)
        x_cos = torch.cos(xb)
        return torch.cat([x_sin, x_cos], dim=-1).view(x.shape[0], -1)


class InverseSamplerNet(nn.Module):
    def __init__(self, num_dims, num_frequencies=10, g_val=0.4, epsilon_val=1, alpha_val=1):
        super().__init__()
       
        self.g_val = g_val
        self.epsilon_val = epsilon_val
        self.alpha_val = alpha_val
        self.num_dims = num_dims
        self.input_dim = self.num_dims
        self.output_dim = self.num_dims

        encoded_dim = 2 * num_frequencies * self.input_dim
        self.encoding = FourierFeatureEncoding(in_features=self.input_dim, num_frequencies=num_frequencies)
        hidden_dim = 256
        layers = [nn.Linear(encoded_dim, hidden_dim)]

        for _ in range(5):
            #layers.append(nn.GELU())
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        #layers.append(nn.GELU())
        layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, self.output_dim))
        self.net = nn.Sequential(*layers)
        
    def forward(self, u):
        u_encoded = self.encoding(u)
        out = self.net(u_encoded)

        # Map outputs to correct ranges
        sampling_vars = []
        if self.num_dims >= 1:  # ∈ [-1, 1]
            cos_theta = torch.tanh(out[:, 0])  
            sampling_vars.append(cos_theta)
        if self.num_dims >= 2:  # ∈ [0, 2π]
            phi = torch.sigmoid(out[:, 1])
            phi = phi * (2*torch.pi)
            sampling_vars.append(phi)
        if self.num_dims >= 3:  # ∈ [0, 1]
            g = torch.sigmoid(out[:, 2]) 
            sampling_vars.append(g)
        if self.num_dims >= 4:  # ∈ [-1, 1]
            alpha = torch.tanh(out[:, 3])
            sampling_vars.append(alpha)
        return torch.stack(sampling_vars, dim=-1)



if __name__ == "__main__":

    # Setup current directory path


    # Load model path
    num_dims = 2
    num_frequencies = 10
    model = InverseSamplerNet(num_dims=num_dims, num_frequencies=num_frequencies)

    dir_current = os.path.dirname(__file__)   
    dir_neural = "mlp_fourier" 
    neural_filepath = os.path.join(dir_current, dir_neural, "d2_mmd.pth")
    model.load_state_dict(torch.load(neural_filepath))
    model.eval()

    # g = 0.99
    # alpha = 1.0
    compare_model_to_true_pdf(model)
