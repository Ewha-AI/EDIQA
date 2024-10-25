from .ediqa import EDIQA
import torch.nn as nn


def build_model(model_type, data_type):
    if model_type == 'ediqa':
        model = nn.Sequential(EDIQA(data_type))
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model