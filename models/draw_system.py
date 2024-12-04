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
#  draw_system.py
#

import torch
import sys
from graphviz import Digraph
from utils.utils import load_model_and_config

import importlib
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image,
)
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import os


def create_model_info_sheet(config, model, model_name="jumpmlnr_700k.pth", sr=16000):
    doc = SimpleDocTemplate("docs/model_info_sheet.pdf", pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("Model Information Sheet", styles["Title"]))
    elements.append(Spacer(1, 12))

    # Model Summary
    elements.append(
        Paragraph(
            f"Model Summary for {os.path.basename(model_name)}", styles["Heading2"]
        )
    )
    num_params = count_parameters(model)
    memory_req = num_params * 1 / 1024 / 1024  # 1 byte per parameter, converted to MB
    hop_length = config["preprocessing"]["args"]["hop_length"]
    sample_rate = sr
    n_fft = config["preprocessing"]["args"]["n_fft"]
    freq_resolution = sample_rate / n_fft

    latency = hop_length / (sample_rate / 1000)

    data = [
        ["Model Attribute", "Value"],
        ["Number of Parameters", f"{num_params:,}"],
        ["Memory Requirement (8-bit weights)", f"{memory_req:.2f} MB"],
        ["Latency (hop length)", f"{latency} ms"],
        ["Frequency Resolution", f"{freq_resolution:.2f} Hz"],
    ]

    t = Table(data)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.grey),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 12),
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
                ("BACKGROUND", (0, 1), (-1, -1), colors.beige),
                ("TEXTCOLOR", (0, 1), (-1, -1), colors.black),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 1), (-1, -1), 11),
                ("TOPPADDING", (0, 1), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 1), (-1, -1), 6),
                ("GRID", (0, 0), (-1, -1), 1, colors.black),
            ]
        )
    )
    elements.append(t)
    elements.append(Spacer(1, 12))

    # System Flow Diagram
    elements.append(Paragraph("System Flow Diagram", styles["Heading2"]))
    elements.append(Image("docs/system_flow_diagram.png", width=550, height=60))
    elements.append(Spacer(1, 12))

    # Neural Network Architecture
    elements.append(Paragraph("Neural Network Architecture", styles["Heading2"]))
    elements.append(Image("docs/nn_architecture_diagram.png", width=550, height=60))

    doc.build(elements)
    print("Model info sheet saved as 'model_info_sheet.pdf'")


def create_system_flow_diagram(config, block_names, param_names):
    dot = Digraph(comment="System Processing Flow")
    dot.attr(rankdir="LR")

    for i, block_name in enumerate(block_names):
        params = [
            f"{name[2]}={config[name[0]][name[1]][name[2]]}" for name in param_names[i]
        ]
        label = f"{block_name}\n" + "\n".join(params)
        dot.node(f"block_{i}", label, shape="rectangle")

        if i > 0:
            dot.edge(f"block_{i-1}", f"block_{i}")

    dot.attr(dpi="300")  # Set DPI to 300 for better quality
    dot.render("docs/system_flow_diagram", format="png", cleanup=True)
    print("System flow diagram saved as 'system_flow_diagram.png'")


def get_model_class(config):
    module = importlib.import_module(config["model"]["module"])
    model_class = getattr(module, config["model"]["main"])
    return model_class


def create_nn_architecture_diagram(config):
    model_class = get_model_class(config)
    model = model_class(**config["model"]["args"])

    dot = Digraph(comment="Neural Network Architecture")
    dot.attr(rankdir="LR")

    layers = []

    def collect_layers(module):
        for layer in module.children():
            if isinstance(
                layer, (torch.nn.Linear, torch.nn.GRU, torch.nn.ReLU, torch.nn.Sigmoid)
            ):
                layers.append(layer)
            else:
                collect_layers(layer)

    collect_layers(model)

    # Add input node
    dot.node("input", "Input", shape="rectangle")
    prev_node = "input"

    # Add layers
    for i, layer in enumerate(layers):
        if isinstance(layer, torch.nn.Linear):
            label = f"Linear\n{layer.in_features} x {layer.out_features}"
            shape = "rectangle"
        elif isinstance(layer, torch.nn.GRU):
            label = f"GRU\n{layer.input_size} x {layer.hidden_size}"
            shape = "rectangle"
        elif isinstance(layer, torch.nn.ReLU):
            label = "ReLU"
            shape = "ellipse"
        else:
            label = f"{layer.__class__.__name__}"
            shape = "rectangle"

        node_name = f"layer_{i}"
        dot.node(node_name, label, shape=shape)
        dot.edge(prev_node, node_name)
        prev_node = node_name

    # Add output node
    dot.node("output", "Output", shape="rectangle")
    dot.edge(prev_node, "output")
    dot.attr(dpi="300")  # Set DPI to 300 for better quality

    dot.render("docs/nn_architecture_diagram", format="png", cleanup=True)
    print("Neural network architecture diagram saved as 'nn_architecture_diagram.png'")


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(ptj_file_path):
    _, config = load_model_and_config(ptj_file_path)

    # Example usage for system flow diagram
    block_names = ["Input", "STFT", "LogMag^2", "NN", "Apply Mask", "ISTFT", "Output"]
    param_names = [
        [("preprocessing", "args", "hop_length")],
        [("preprocessing", "args", "n_fft")],
        [("preprocessing", "args", "logmag_epsilon")],
        [("model", "args", "io_size")],
        [
            ("postprocessing", "args", "min_gain"),
            ("postprocessing", "args", "naturalness"),
        ],
        [("postprocessing", "args", "n_fft")],
        [("postprocessing", "args", "hop_length")],
    ]
    create_system_flow_diagram(config, block_names, param_names)
    model_class = get_model_class(config)
    model = model_class(**config["model"]["args"])
    create_nn_architecture_diagram(config)

    create_model_info_sheet(config, model, model_name=ptj_file_path)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python draw_system.py <ptj_file_path>")
        sys.exit(1)

    ptj_file_path = sys.argv[1]
    main(ptj_file_path)
