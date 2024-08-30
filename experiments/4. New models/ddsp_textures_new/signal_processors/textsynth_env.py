import torch
import numpy as np
import ddsp_textures_new.auxiliar.filterbanks as fb
from ddsp_textures_new.auxiliar.auxiliar import *

def textsynth_env_param_extractor(signal, fs, N_filter_bank, percentage_use):
    percentage_use = percentage_use * (1 / N_filter_bank)
    low_lim, high_lim = 20, fs / 2  # Low limit of filter
    size = signal.size(0)
    
    # Assuming fb.EqualRectangularBandwidth works with torch tensors
    erb_bank = fb.EqualRectangularBandwidth(size, fs, N_filter_bank, low_lim, high_lim)
    erb_bank.generate_subbands(signal)  # generate subbands for signal y
    
    erb_subbands = torch.tensor(erb_bank.subbands[:, 1:-1], dtype=torch.float32)
    erb_envs = torch.abs(hilbert(erb_subbands.transpose(0, 1)).transpose(0, 1))
    
    print(f"The signal has a size of {size} samples.")
    print(f"Each envelope of the {N_filter_bank} filters will be approximated by {int(percentage_use * size * 0.5)} complex parameters.")
    print(f"This gives a total of {int(percentage_use * size * 0.5) * N_filter_bank} parameters in total, and it corresponds to {2 * int(percentage_use * size * 0.5) * N_filter_bank / size} of the entire signal size.")
    
    real_param = []
    imag_param = []
    
    for i in range(N_filter_bank):
        erb_env_local = erb_envs[:, i]
        fft_coeffs = torch.fft.rfft(erb_env_local)[:int(percentage_use * size * 0.5)] ###########################3
        real_param.append(fft_coeffs.real)
        imag_param.append(fft_coeffs.imag)
    
    real_param = torch.cat(real_param)
    imag_param = torch.cat(imag_param)
    
    return real_param, imag_param

def textsynth_env(parameters_real, parameters_imag, seed, N_filter_bank, size, target_loudness=1):
    N = parameters_real.size(0)
    parameters_size = N // N_filter_bank
    signal_final = torch.zeros(size, dtype=torch.float32)
    
    for i in range(N_filter_bank):
        # Construct the local parameters as a complex array
        parameters_local = parameters_real[i * parameters_size : (i + 1) * parameters_size] + 1j * parameters_imag[i * parameters_size : (i + 1) * parameters_size]
        
        # Initialize FFT coefficients array
        fftcoeff_local = torch.zeros(int(size/2)+1, dtype=torch.complex64)
        fftcoeff_local[:parameters_size] = parameters_local ###########################################3
        
        # Compute the inverse FFT to get the local envelope
        env_local = torch.fft.irfft(fftcoeff_local)
        
        # Extract the local noise
        noise_local = seed[:, i]
        
        # Generate the texture sound by multiplying the envelope and noise
        texture_sound_local = env_local * noise_local
        
        # Accumulate the result
        signal_final += texture_sound_local
    
    loudness = torch.sqrt(torch.mean(signal_final ** 2))
    signal_final = signal_final / loudness

    signal_final = target_loudness * signal_final

    return signal_final


def textsynth_env_batches(parameters_real, parameters_imag, seed, N_filter_bank, size):
    # Get the batch size
    batch_size = parameters_real.size(0)
    parameters_size = parameters_real.size(1) // N_filter_bank

    # Initialize the final signal tensor for the entire batch
    signal_final = torch.zeros((batch_size, size), dtype=torch.float32, device=parameters_real.device)
    
    for i in range(N_filter_bank):
        # Construct the local parameters as a complex array for each filter in the batch
        parameters_local = (parameters_real[:, i * parameters_size : (i + 1) * parameters_size] 
                            + 1j * parameters_imag[:, i * parameters_size : (i + 1) * parameters_size])
        
        # Initialize FFT coefficients array for the entire batch
        fftcoeff_local = torch.zeros((batch_size, int(size / 2) + 1), dtype=torch.complex64, device=parameters_real.device)
        fftcoeff_local[:, :parameters_size] = parameters_local

        # Compute the inverse FFT to get the local envelope for each batch item
        env_local = torch.fft.irfft(fftcoeff_local).real

        # Extract the local noise for each batch item
        noise_local = seed[:, i]

        # Generate the texture sound by multiplying the envelope and noise for each batch item
        texture_sound_local = env_local * noise_local

        # Accumulate the result for each batch item
        signal_final += texture_sound_local
    
    return signal_final

# def textsynth_env_batches(parameters_real, parameters_imag, seed, N_filter_bank, size):
#     # Get the device of the parameters_real tensor
#     device = parameters_real.device

#     # Get the batch size
#     batch_size = parameters_real.size(0)
#     parameters_size = parameters_real.size(1) // N_filter_bank

#     # Initialize the final signal tensor for the entire batch on the same device
#     signal_final = torch.zeros((batch_size, size), dtype=torch.float32, device=device)
    
#     # Move seed to the same device as parameters_real
#     seed = seed.to(device)

#     for i in range(N_filter_bank):
#         # Construct the local parameters as a complex array for each filter in the batch
#         parameters_local = (parameters_real[:, i * parameters_size : (i + 1) * parameters_size] 
#                             + 1j * parameters_imag[:, i * parameters_size : (i + 1) * parameters_size])
        
#         # Initialize FFT coefficients array for the entire batch
#         fftcoeff_local = torch.zeros((batch_size, int(size / 2) + 1), dtype=torch.complex64, device=device)
#         fftcoeff_local[:, :parameters_size] = parameters_local

#         # Compute the inverse FFT to get the local envelope for each batch item
#         env_local = torch.fft.irfft(fftcoeff_local).real

#         # Generate the texture sound by multiplying the envelope and noise for each batch item
#         texture_sound_local = env_local * seed[:, i].unsqueeze(1)

#         # Accumulate the result for each batch item
#         signal_final += texture_sound_local
    
#     return signal_final

def textsynth_env_stems(parameters_real, parameters_imag, seed, N_filter_bank, size, target_loudness=1):
    N = parameters_real.size(0)
    parameters_size = N // N_filter_bank
    signal_final = torch.zeros(size, dtype=torch.float32)
    
    envelopes = []

    for i in range(N_filter_bank):
        # Construct the local parameters as a complex array
        parameters_local = parameters_real[i * parameters_size : (i + 1) * parameters_size] + 1j * parameters_imag[i * parameters_size : (i + 1) * parameters_size]
        
        # Initialize FFT coefficients array
        fftcoeff_local = torch.zeros(int(size/2)+1, dtype=torch.complex64)
        fftcoeff_local[:parameters_size] = parameters_local ###########################################3
        
        # Compute the inverse FFT to get the local envelope
        env_local = torch.fft.irfft(fftcoeff_local)
        
        # Save local envelope
        envelopes.append(env_local)

        # # Extract the local noise
        # noise_local = seed[:, i]
        
        # # Generate the texture sound by multiplying the envelope and noise
        # texture_sound_local = env_local * noise_local
        
        # # Accumulate the result
        # signal_final += texture_sound_local
    return envelopes


def textsynth_env_stems_batches(parameters_real, parameters_imag, seed, N_filter_bank, size):
    # Get the batch size
    batch_size = parameters_real.size(0)
    parameters_size = parameters_real.size(1) // N_filter_bank

    envelopes = []

    # Initialize the final signal tensor for the entire batch
    signal_final = torch.zeros((batch_size, size), dtype=torch.float32, device=parameters_real.device)
    
    for i in range(N_filter_bank):
        # Construct the local parameters as a complex array for each filter in the batch
        parameters_local = (parameters_real[:, i * parameters_size : (i + 1) * parameters_size] 
                            + 1j * parameters_imag[:, i * parameters_size : (i + 1) * parameters_size])
        
        # Initialize FFT coefficients array for the entire batch
        fftcoeff_local = torch.zeros((batch_size, int(size / 2) + 1), dtype=torch.complex64, device=parameters_real.device)
        fftcoeff_local[:, :parameters_size] = parameters_local

        # Compute the inverse FFT to get the local envelope for each batch item
        env_local = torch.fft.irfft(fftcoeff_local).real

        envelopes.append(env_local)
    
    return envelopes


