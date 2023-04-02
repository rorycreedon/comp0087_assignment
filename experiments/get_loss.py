"""
import packages
"""

import torch.nn as nn

"""
functions
"""

def get_loss_func(task):
    """
    Get the loss function based on the task.

    Params:
    `task` (str): the task to be performed
    """
    if task == "binary_cls" or task == "multi_cls":
        loss_func = nn.BCELoss()
    elif task == "regression":
        loss_func = nn.L1Loss()

    return loss_func

    