import torch

# Native Python sum
def python_add(a, b):
    return a + b

# PyTorch tensor sum
def torch_add(a, b):
    return torch.add(a, b)

# Native Python element-wise list addition
def python_add_lists(a, b):
    return [x + y for x, y in zip(a, b)]

# PyTorch tensor element-wise addition
def torch_add_tensors(a, b):
    return a + b

