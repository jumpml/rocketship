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
#  create_ptj.py
#

from utils.utils import check_path, initialize_config
import torch
import json5
import sys


def create_ptj_and_check_compatibility(model_path, json5_path, output_path):
    # Load model weights
    model_state_dict = torch.load(check_path(model_path))

    # Load JSON5 config
    with open(check_path(json5_path), "r") as f:
        config = json5.load(f)

    try:
        # Initialize model from config
        model = initialize_config(config["model"])

        # Check if the model can load the weights
        model.load_state_dict(model_state_dict)

        # If we reach here, the model is compatible
        print("Model and config are compatible.")

        # Save the .ptj file
        torch.save({"model": model_state_dict, "config": config}, output_path)
        print(f"Successfully created .ptj file: {output_path}")

    except Exception as e:
        print("Error: The provided config and weights are incompatible.")
        print(f"Specific error: {str(e)}")
        print(
            "This could be due to mismatched layer names, shapes, or missing/extra parameters."
        )


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <model_path> <json5_path> <output_path>")
        sys.exit(1)

    model_path = sys.argv[1]
    json5_path = sys.argv[2]
    output_path = sys.argv[3]

    create_ptj_and_check_compatibility(model_path, json5_path, output_path)
