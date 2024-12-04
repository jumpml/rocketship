#  JumpML Rocketship - Neural Network Inference with Audio Processing
#
#  Copyright 2020-2024 JUMPML
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
#  compare_wave_files.py
#

import soundfile as sf
import numpy as np
import sys
import matplotlib.pyplot as plt
from pypesq import pesq
from pystoi.stoi import stoi
from utils import calc_mae


def compare_wav_files(filepath1, filepath2, plot_enable=False):
    # Read the audio data from the WAV files
    data1, rate1 = sf.read(filepath1)
    data2, rate2 = sf.read(filepath2)

    # Check if the sample rates are the same
    if rate1 != rate2:
        print("Sample rates do not match!")
        return False

    # Check if the lengths of the audio data are the same
    if len(data1) != len(data2):
        print(f"Lengths of audio data {len(data1)} !=  {len(data2)} do not match! ")
        L = min(len(data1), len(data2))
        data1 = data1[:L]
        data2 = data2[:L]

    # Compare the audio data
    if np.array_equal(data1, data2):
        print("Waveforms are the same.")
        return True
    else:
        print("Waveforms are different (or) are they really?")
        STOI = stoi(data1, data2, rate1, extended=False)
        # PESQ = pesq(data1, data2, 16000)
        # PESQ = 0
        # STOI = 0

        mae, mad, mse = calc_mae(data1, data2)
        print(f"MAE = {mae}\t MAD = {mad}\t MSE = {mse} \t STOI = {STOI:0.4f}")
        if plot_enable:
            plt.plot(data1)
            plt.plot(data2)
            plt.show()

        return False


# Get the file paths from the command line
filepath1 = sys.argv[1]
filepath2 = sys.argv[2]

# Compare the WAV files
compare_wav_files(filepath1, filepath2)
