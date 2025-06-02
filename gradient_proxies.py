# File: gradient_proxies.py

import torch
from nas_201_api import get_cell_based_tiny_net
from xautodl.datasets import get_datasets

def compute_synflow(arch_str):
    model = get_cell_based_tiny_net({'name': arch_str})
    model.eval()

    dummy_input = torch.randn(1, 3, 32, 32, requires_grad=True)
    model.zero_grad()

    output = model(dummy_input)
    score = output.sum().backward()

    return sum(p.grad.abs().sum().item() for p in model.parameters() if p.grad is not None)
