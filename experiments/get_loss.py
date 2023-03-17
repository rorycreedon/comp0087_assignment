"""
import packages
"""

import torch.nn as nn

"""
functions
"""

def get_loss_func(task):
    if task == "binary_cls":
        loss_func = nn.BCELoss()
    elif task == "multi_cls":
        loss_func = nn.CrossEntropyLoss()
    elif task == "regression":
        loss_func = nn.L1Loss()

    return loss_func

    