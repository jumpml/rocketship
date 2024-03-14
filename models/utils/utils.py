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
#  utils.py
# 

import re
import os
import sys

import numpy as np
import subprocess
import json5
import torch
import importlib

def initialize_config(module_cfg):
  module = importlib.import_module(module_cfg["module"])
  return getattr(module, module_cfg["main"])(**module_cfg["args"])

def load_model_and_config(file_path, print_config=False, device="cpu"):
  config = torch.load(file_path)['config']
  model_state_dict = torch.load(file_path)['model']
  # LOAD MODEL ON DEVICE
  model = initialize_config(config["model"])
  model = model.to(device)
  model.load_state_dict(model_state_dict)
  model.eval()
  if print_config:
    print(json5.dumps(config,indent=2))
  return model, config

def save_model_and_config(config_file, model_file, file_path):
    torch.save({'model':torch.load(check_path(model_file)),'config':json5.load(open(check_path(config_file)))}, file_path)


def check_path(file_path):
  if getattr(sys, 'frozen', False):
    file_path = os.path.join(sys._MEIPASS, file_path)  # Path inside the executable
  
  return file_path

def run_subprocess_cmd(command):
    # Execute the command using subprocess
    try:
      result = subprocess.run(command, check=True, capture_output=True, text=True)
      print("Command output:SUCCESS\n", result.stdout)
    except subprocess.CalledProcessError as error:
      print("Command failed:\n", error.output)

def calc_mae(data1, data2):
    mae = np.mean(np.abs(data1 - data2))
    mad = np.max(np.abs(data1 - data2))
    mse = np.mean((data1-data2)**2)
    return (mae, mad, mse)

def replace_extension(file_path, ext):
    """Replaces the file extension of a given file path with the specified extension.

    Args:
        file_path (str): The path to the file.
        ext (str): The new file extension to replace the existing one.

    Returns:
        str: The new file path with the updated extension.
    """

    base_filename, _ = os.path.splitext(os.path.basename(file_path))
    new_file_path = os.path.join(os.path.dirname(file_path), base_filename + "." +ext)
    return new_file_path, base_filename


def extract_jumpmlnr_params(filename):
  """Extracts the number of hidden units, input vector length, and output vector length from a filename in the naming convention `jumpmlnr_H{hidden_units}_IO{input_output_vector_length}.pth`.

  Args:
    filename: The filename to extract the parameters from.

  Returns:
    A tuple containing the number of hidden units, Input/Output vector length respectively.
  """

  match = re.search(r'dummy_H(?P<hidden_units>\d+)_IO(?P<input_output_vector_length>\d+)\.pth', filename)
  if match is None:
    raise ValueError('Invalid filename format: {}'.format(filename))

  hidden_units = int(match.group('hidden_units'))
  input_output_vector_length = int(match.group('input_output_vector_length'))

  return hidden_units, input_output_vector_length

def parse_model_name(prefix):
  """
  Parses a model name with the format "prefix_H{hidden_units}_IO{input_units}" and returns the number of hidden units and inputs.
  Args:
  prefix (str): The model name prefix.
  Returns:
  tuple(int, int): A tuple containing the number of hidden units and inputs, or None if the format is invalid.
  """
  pattern = r"(?P<prefix>\w+)_H(?P<hidden_units>\d+)_IO(?P<input_units>\d+)"
  match = re.match(pattern, prefix)
  if match:
    return int(match.group("hidden_units")), int(match.group("input_units"))
  else:
    return None
 

def compare_tensors(tensor1, tensor2, delta_tolerance=1e-3):
  """Compares two numpy floating point tensors with a given delta tolerance level and reports the number of disagreements.

  Args:
    tensor1: The first numpy floating point tensor.
    tensor2: The second numpy floating point tensor.
    delta_tolerance: The delta tolerance level.

  Returns:
    The number of disagreements between the two tensors.
  """

  # Check if the tensors have the same shape.
  if tensor1.shape != tensor2.shape:
    raise ValueError("Tensors must have the same shape.")

  # Calculate the absolute difference between the two tensors.
  diff = np.abs(tensor1 - tensor2)

  # Count the number of disagreements.
  num_disagreements = np.count_nonzero(diff > delta_tolerance)

  return num_disagreements
