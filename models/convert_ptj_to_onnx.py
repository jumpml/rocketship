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
#  convert_ptj_to_onnx.py
#  Helps to convert Pytorch/JumpML (ptj) model file format to ONNX format
#  
import torch
import torch.onnx
import onnxruntime
import argparse
from utils.utils import replace_extension, load_model_and_config, compare_tensors
import os
import numpy as np

def convert_to_onnx(model_file_path):
    device = "cpu"

    directory_path = os.path.dirname(model_file_path)
    
    # Relative path to model weights file (.pth)
    onnx_path, model_fname = replace_extension(model_file_path, "onnx")
    
    # Get model configuration settings
    model, config = load_model_and_config(model_file_path, print_config=True)
    hidden_size = config["model"]["args"]["hidden_size"][0]
    io_size = config["model"]["args"]["io_size"]
    
    # Create inputs and predict outputs using torch
    # First compute STFT magnitude, followed by 20*log10(epsilon + Mag)
    ipt = 20*torch.log10(1e-4 + torch.tensor([10.869651, 16.671688, 26.74527, 33.615658, 33.473988, 27.30435, 20.0569, 15.559971, 12.970183, 11.602915, 11.486061, 11.619814, 11.750951, 11.694953, 10.745853, 9.114418, 7.9566035, 7.086113, 6.0092897, 5.1131635, 4.458161, 4.0274725, 3.4854794, 2.757676, 2.4103115, 1.9966626, 1.1067314, 0.24194857, 0.27826765, 0.643473, 0.73862195, 0.5553532, 0.9257371, 1.2276006, 1.2931978, 1.4213613, 1.597105, 1.807468, 1.7115953, 1.3121465, 1.3970919, 1.5908898, 1.3832546, 1.3054293, 1.439988, 1.380313, 1.320168, 1.3674905, 1.3300865, 1.2359328, 1.2235013, 1.2872403, 1.2971729, 1.2790587, 1.2544004, 1.2064505, 1.1639596, 1.0956968, 0.9661235, 0.8403754, 0.8049353, 0.7664929, 0.6860457, 0.6247753, 0.51401734, 0.4327611, 0.3806892, 0.24829106, 0.4782541, 0.6006703, 0.53517514, 0.58825415, 0.6124979, 0.56295013, 0.6441737, 0.72400266, 0.79937303, 0.8965566, 0.8969054, 0.7727565, 0.61489075, 0.6346722, 0.7403714, 0.7460339, 0.7132762, 0.69163364, 0.64030623, 0.56419665, 0.48843664, 0.5366717, 0.6023088, 0.60178584, 0.5906987, 0.52865714, 0.51420754, 0.5628825, 0.5432606, 0.49792212, 0.5045553, 0.5234665, 0.50975037, 0.46311402, 0.41829136, 0.4093598, 0.40069383, 0.36333385, 0.3279804, 0.33399597, 0.36369646, 0.37688333, 0.35912573, 0.35330653, 0.3711064, 0.39472657, 0.4445981, 0.4651349, 0.4835688, 0.50805837, 0.4781384, 0.48001722, 0.5014009, 0.4818455, 0.48933104, 0.51101243, 0.4945479, 0.47264197, 0.4481584, 0.39673552, 0.4021186, 0.43152967, 0.38316366, 0.35522723, 0.37549475, 0.3545854, 0.3426673, 0.3324708, 0.31249693, 0.331516, 0.33506835, 0.32602382, 0.33439767, 0.324918, 0.32124168, 0.34077123, 0.3352492, 0.31267986, 0.32700986, 0.3490128, 0.35510468, 0.35690668, 0.34015277, 0.30949694, 0.28459585, 0.25389484, 0.2139383, 0.17479344, 0.14366016, 0.12711209, 0.11935056, 0.11536923, 0.11424923]))
    
    ipt = ipt[:io_size]
    ipt = torch.unsqueeze(torch.unsqueeze(ipt,0),0)
    h0 =  torch.rand(1, 1, hidden_size, dtype=torch.float32)
    state_in =torch.stack([h0, h0, h0])
    with torch.no_grad():
        out, state_out = model(ipt, state_in)

    #np.save(f'{directory_path}/X.npy', ipt.numpy() )
    #np.save(f'{directory_path}/Y.npy', (out.numpy(), state_out.numpy()))
    
    # Export the model to ONNX
    torch.onnx.export(model, (ipt, state_in), onnx_path, input_names=["input", "state_in"], output_names=["output", "state_out"])

    # compute ONNX Runtime output prediction
    ort_session = onnxruntime.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    ort_outs = ort_session.run(["output", "state_out"], {"input": ipt.numpy(), "state_in": state_in.numpy()})

    ort_out = ort_outs[0]
    ort_state_out = ort_outs[1]

    if (compare_tensors(ort_out, out.numpy()) == 0) & (compare_tensors(ort_state_out, state_out.numpy()) == 0):
       print("Outputs from Torch and ONNX Match!")
       return 0 # success
    else:
       print("We have a mismatch!")
       return 1 # not success


if __name__ == '__main__':
   parser = argparse.ArgumentParser(description="Convert .ptj format to ONNX format")
   parser.add_argument("-m", "--model_file", required=False,
                        type=str, help="Pytorch JumpML model (.ptj) file", default='models/pretrained_models/jumpmlnr_pro.ptj')
   args = parser.parse_args()

   convert_to_onnx(args.model_file)
    
        

