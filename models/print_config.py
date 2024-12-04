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
#  print_config.py
#

import torch
import json5
import sys


def print_config_from_ptj(ptj_file_path):
    try:
        # Load the .ptj file
        data = torch.load(ptj_file_path)

        # Extract the config
        config = data["config"]

        # Print the config in a neat format
        print(json5.dumps(config, indent=2))

    except Exception as e:
        print(f"Error loading or parsing the .ptj file: {str(e)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python print_config.py <ptj_file_path>")
        sys.exit(1)

    ptj_file_path = sys.argv[1]
    print_config_from_ptj(ptj_file_path)
