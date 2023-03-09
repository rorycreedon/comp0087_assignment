import torch

# use gpu
device = "mps" if torch.backends.mps.is_available() else "cpu"