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
#  simple_gru.py
# 
import torch
import torch.nn as nn
import torch.nn.functional as F


class GRUlayer(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUlayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.input_size,self.hidden_size, num_layers=1, bias=bias, batch_first=True)

    def forward(self, input, h0=None):
        out, h = self.gru(input, h0)
        return out, h

class LinearLayer(nn.Module):
    def __init__(self, input_size, output_size, bias=True, linear_nl = nn.ReLU()):
        super(LinearLayer, self).__init__()
        self.input_size = input_size
        self.hidden_size = output_size
        self.linear = nn.Linear(in_features=input_size, out_features=output_size, bias=bias)
        self.linear_activation = linear_nl

    def forward(self, input):
        out = self.linear_activation(self.linear(input))
        return out

class GRU3(nn.Module):
    def __init__(self, io_size=161, hidden_size=[512,512,512], bias=True, bias_lin=True):
        super(GRU3, self).__init__()
        self.input_size = io_size
        self.output_size = io_size
        assert len(hidden_size) == 3
        self.hidden_size = hidden_size
        self.gru1 = GRUlayer(self.input_size, hidden_size[0], bias=bias)
        self.gru2 = GRUlayer(hidden_size[0], hidden_size[1], bias=bias)
        self.gru3 = GRUlayer(hidden_size[1], hidden_size[2], bias=bias)
        self.linear1 = LinearLayer(hidden_size[2], self.output_size, bias=bias_lin, 
                            linear_nl=nn.Sigmoid())

    def forward(self, input, h=[None,None,None]):
        # input = F.pad(input, (0,0,0,self.look_ahead))  # Pad the look ahead # (batch_size, T, F)
        out1, _ = self.gru1(input, h[0])
        out2, _ = self.gru2(out1, h[1])
        out3, _ = self.gru3(out2, h[2])
        out = self.linear1(out3)
        return out, torch.stack([out1, out2, out3])


class GRU2(nn.Module):
    def __init__(self, io_size=161, hidden_size=[512,512], bias=True, bias_lin=True):
        super(GRU2, self).__init__()
        self.input_size = io_size
        self.output_size = io_size
        assert len(hidden_size) == 2
        self.hidden_size = hidden_size
        self.gru1 = GRUlayer(self.input_size, hidden_size[0], bias=bias)
        self.gru2 = GRUlayer(hidden_size[0], hidden_size[1], bias=bias)
        self.linear1 = LinearLayer(hidden_size[1], self.output_size, bias=bias_lin, 
                            linear_nl=None)

    def forward(self, input, h=[None,None,None]):
        out1, _ = self.gru1(input, h[0])
        out2, _ = self.gru2(out1, h[1])
        out = self.linear1(out2)
        return out, torch.stack([out1, out2])