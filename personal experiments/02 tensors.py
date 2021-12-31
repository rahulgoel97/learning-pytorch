### Pytorch experiments 2

'''

Tensors

'''

### Define the network
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

var_1 = torch.randn(4) # 1x4
print(f"torch.randn(4): {var_1}")

var_2 = torch.randn(4,2) # 4x2
print(f"torch.randn(4,2): {var_2}")

var_3 = torch.randn(4,2,3) # 8x3
print(f"torch.randn(4,2,3): {var_3}")

var_4 = torch.randn(4,2,3,1) # 24x1, with 4 tensors of 2 rows
print(f"torch.randn(4,2,3,1): {var_4}")

var_5 = torch.randn(3,2,1,5) # 6x5, with 3 tensors of 2 rows 
print(f"torch.randn(3,2,1,5): {var_5}")

# Self Q: Build a random tensor with 12 rows and 5 columns, with 4 tensors of 3 rows
reqd_var = torch.randn(4,3,1,5)
print(f"torch.randn(4,3,1,5): {reqd_var}")


'''Learned:
Columns = Last index
Total rows = Product of all prior indices

(a,b,c,d) -> d cols with abc rows

where there are a sub-tensors of b rows
'''

var_6 = torch.randn(1,1,32,32)
print(f"torch.randn(1,1,32,32): {var_6}") # 32 rows x 32 cols, where there is 1 sub-tensor with 1 row