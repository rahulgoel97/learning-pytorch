### Pytorch Experiments 1

'''
 Understanding torch.nn using Linear

'''
import torch
from torch import nn


### Linear
print("=== Linear ===")


m = nn.Linear(20, 30) # in 20 -> out 30

print(f"\nm is {m}")

in_tensor = torch.randn(10, 20)

print(f"input tensor is {in_tensor}")
print(f"Weights {m.weight} | Bias {m.bias}")

print(f"output tensor is {m(in_tensor)}")

print("\nPossible commands on nn.Linear object:\n")
print(dir(m))
