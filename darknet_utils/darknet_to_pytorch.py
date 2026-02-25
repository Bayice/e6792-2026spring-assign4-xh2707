"""
Functions for converting from Darknet  to PyTorch.

E6792 Spring 2026

YOU DO NOT NEED TO MODIFY THIS FUNCTION TO COMPLETE THE ASSIGNMENT
"""

import sys
from .darknet2pytorch import Darknet
import torch

def load_pytorch(cfg_file, weights_file):
    loaded_model = Darknet(cfg_file, inference=True)
    loaded_model.load_state_dict(torch.load(weights_file))
    loaded_model.eval()
    return loaded_model

def load_darknet_as_pytorch(model_cfg, darknet_weights):
    pytorch_darknet_model = Darknet(model_cfg, inference=True)
    pytorch_darknet_model.load_weights(darknet_weights)
    return pytorch_darknet_model

def save_pytorch_model(model, path):
    torch.save(model.state_dict(), path)
