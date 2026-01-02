import numpy as np
import mitsuba as mi
import drjit as dr
import torch 



######################################################################
# 1 DIMENSION (Already normalized distribution over sphere): 
    # Henyey-Greenstein (HG) Phase Function as the Default
    # Sampling Variables:
        # "cos_theta" [-1,1]: the cosine of the scattering angle (input -> output)
    # Conditioning Parameters:
        # "g" [-1,1]: polar asymmetry term

    # Note: Azimuth angle used to define output direction is sampled uniformly
######################################################################

def henyey_greenstein_torch(cos_theta, g_const):
    temp = 1.0 + g_const**2 - 2.0*g_const*cos_theta
    return (1/(4*torch.pi)) * (1.0 - g_const**2) / (temp * torch.sqrt(temp))

def henyey_greenstein_drjit(cos_theta, g_const):
    temp = 1.0 + g_const**2 - 2.0*g_const*cos_theta
    return dr.inv_four_pi * (1.0 - g_const**2) / (temp * dr.sqrt(temp))

def henyey_greenstein(cos_theta, g_const):
    temp = 1.0 + g_const**2 - 2.0*g_const*cos_theta
    return (1/(4*np.pi)) * (1.0 - g_const**2) / (temp * np.sqrt(temp))



######################################################################
# 4 DIMENSIONS: 
 #   In Mitsuba 3, any function with more than 2 variables samples only
#  the first 2 variables "cos_theta", "phi". The rest are pseudo-sampling
# variables that are learned in the PDF, but conditioned during the render
# Replicating the sampling/memory cost, but not the actual sample used...

# When evaluating, take the average of the PDF slices

######################################################################

def hg_polar_azimuth_lambda_pol_4d_torch(cos_theta, phi, lmbd, pol, g_const, epsilon_const, alpha_const):
    hg = henyey_greenstein_torch(cos_theta, g_const)
    modulation = (1.0 + epsilon_const * lmbd**2 * torch.cos(phi + alpha_const*(cos_theta*torch.pi)) * pol)
    phase_function_val = hg * modulation
    return phase_function_val


def hg_polar_azimuth_lambda_p_4d(cos_theta, phi, lmbd, pol, g_const, epsilon_const, alpha_const):
    hg = henyey_greenstein(cos_theta, g_const)
    modulation = (1.0 + epsilon_const * lmbd**2 * np.cos(phi + alpha_const*(cos_theta*np.pi)) * pol)
    phase_function_val = hg * modulation
    return phase_function_val


























