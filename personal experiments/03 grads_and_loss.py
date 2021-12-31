### Pytorch experiments 3

'''

Understanding loss 

http://seba1511.net/tutorials/beginner/blitz/neural_networks_tutorial.html

A typical training procedure for a neural network is as follows:

- Define the neural network that has some learnable parameters (or weights)
- Iterate over a dataset of inputs
- Process input through the network
- Compute the loss (how far is the output from being correct)
- Propagate gradients back into the network’s parameters
- Update the weights of the network, typically using a simple update rule: weight = weight - learning_rate * gradient

'''

### Define the network
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 6, 5)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x): # Backward is already defined using autograd...
		x = F.max_pool2d(F.relu(self.conv1(x)), (2,2))
		x = F.max_pool2d(F.relu(self.conv2(x)), 2)
		x = x.view(-1, self.num_flat_features(x))
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *=s
		return num_features

net = Net()
print(net)


print("== Experiments ==")
params = list(net.parameters())
print(f"Params are {net.parameters()} | Param0 {params[0]} | Total {len(params)}")

print("\n\n Fitting random values")
var_in = Variable(torch.randn(1,1,32,32))
out = net(var_in)
print(out)

print("\n\n== Loss function ==")
