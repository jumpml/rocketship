#  JumpML Rocketship - Neural Network Inference with Audio Processing
#  
#  Copyright 2020-2024 JUMPML LLC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  run_prediction.py
# 
import argparse
import numpy as np
import torch
import scipy.io.wavfile as wave
import librosa
from utils.utils import load_model_and_config, initialize_config

device = 'cpu'

# UTILITY FUNCTIONS
def get_window(N):
    TWO_PI_BY_N = 2*np.pi/N
    n = np.arange(0, N)
    win = np.sqrt(0.5 - 0.5 * np.cos(TWO_PI_BY_N * n)) 
    return win.astype('f')

# PROCESSING FUNCTIONS
class preproc:
    def __init__(self, n_fft, hop_length, logmag_epsilon):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.logmag_epsilon = logmag_epsilon

    def process(self, y):
    # Get STFT Magnitude and Phase, add dummy batch_size dimension at index0,
        noisy_mag, noisy_phase = librosa.magphase(librosa.stft(y, n_fft=self.n_fft, 
        hop_length=self.hop_length, window=get_window(self.n_fft)))
        noisy_mag_tensor = torch.tensor(noisy_mag[None, ...], device=device).permute(0, 2, 1)  # (batch_size, T, F)
        assert noisy_mag_tensor.dim() == 3
        nn_features_tensor = 10.0*torch.log10(noisy_mag_tensor**2 + self.logmag_epsilon)
        return (noisy_mag, noisy_phase, nn_features_tensor)

class postproc:
    def __init__(self, n_fft, hop_length, min_gain=0.001, naturalness=0.5):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.min_gain = min_gain
        self.naturalness = naturalness
    
    def process(self, noisy_mag, noisy_phase, gain_tensor):
        pi = torch.tensor(np.pi, device=device)
        gain_mod_tensor = torch.maximum(gain_tensor * torch.sin(pi/2 * gain_tensor), torch.tensor(self.min_gain, device=device))
        gain_mod_tensor = torch.maximum(self.naturalness*gain_tensor, gain_mod_tensor)
        # Now convert everything to numpy to use numpy/librosa
        speech_mask = (gain_mod_tensor.squeeze(0).permute(1,0)).detach().cpu().numpy()
        # Apply masks to Noisy magnitude and use noisy phase to perform ISTFT
        pred_clean_mag = noisy_mag * speech_mask
        pred_clean_y = librosa.istft(pred_clean_mag * noisy_phase, n_fft=self.n_fft, hop_length=self.hop_length, 
                                window=get_window(self.n_fft), center=False)
        return pred_clean_y


def inference(model, nn_features_tensor, io_size=160):
    # Make mask/gain predictions
    speech_mask_tensor = torch.zeros_like(nn_features_tensor)
    speech_mask_tensor[:,:,:io_size], _ = model(nn_features_tensor[:,:,:io_size])
    return speech_mask_tensor

def run_pytorch_prediction(model_file_path, y, naturalness, min_gain):
    # LOAD MODEL & CONFIG
    model, config = load_model_and_config(model_file_path)

    # CONFIGURE & RUN PREPROCESSING
    preproc = initialize_config(config["preprocessing"])
    (noisy_mag, noisy_phase, nn_features_tensor) = preproc.process(y)

    # RUN NN INFERENCE 
    io_size = config["model"]["args"]["io_size"]
    speech_mask_tensor = inference(model, nn_features_tensor, io_size=io_size)

    # CONFIGURE & RUN POSTPROCESSING
    postproc = initialize_config(config["postprocessing"])
    postproc.min_gain = min_gain       # command-line override
    postproc.naturalness = naturalness # command-line override
    y = postproc.process(noisy_mag, noisy_phase, speech_mask_tensor)

    return y

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Speech enhancement using JUMPML NR")
    parser.add_argument("-i", "--input_file", required=True, default='data/outdoor_mix.wav',
                        type=str, help="Input wav file")
    parser.add_argument("-o", "--output_file", required=False,
                        type=str, help="Output wav file", default='data/jumpmlnr_output.wav')
    parser.add_argument("-m", "--model_file", required=False,
                        type=str, help="Pytorch JumpML model (.ptj) file", default='pretrained_models/jumpmlnr_pro.pth')
    parser.add_argument("-g", "--minGain", required=False,
                        type=float, help="Min. Gain in dB in range [-60, 0.0] dB", default=-30)
    parser.add_argument("-n", "--naturalness", required=False,
                        type=float, help="Naturalness in range [0,1]", default=0.5)
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)

    min_gain = np.maximum(np.minimum(0.0, args.minGain),-60)
    if args.minGain != min_gain:
        print(f"minGain is out of range. Setting to {min_gain}")
    min_gain = 10**(min_gain / 10)
    
    nat = np.maximum(np.minimum(1.0, args.naturalness),0.0)
    if args.naturalness != nat:
        print(f"naturalness is out of range. Setting to {nat}")

    y, _ = librosa.load(args.input_file, sr=16000)
    y_clean = run_pytorch_prediction(args.model_file, y, naturalness=nat, min_gain=min_gain)
    
    wave.write(args.output_file, 16000, y_clean)
    