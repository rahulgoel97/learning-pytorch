"""
Rahul Goel
12/23/2021
Exploring stuff in pytorch...

Thanks to liaoxingyu: https://github.com/L1aoXingyu/pytorch-beginner/blob/master/01-Linear%20Regression/Linear_Regression.py
"""

import matplotlib.pyplot as plt 
import numpy as np 
import torch
from torch import nn
from torch.autograd import Variable

# Define some data...

x_data = np.array([[3.3], [4.4], [5.5], [6.7], [6.9], [4.2], [9.779], [6.2]],
					dtype=np.float32)

y_data = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], [3.366], [2.596]],
					dtype=np.float32)

# Convert to pytorch Tensors using .from_numpy (neat!)
x_train = torch.from_numpy(x_data)
y_train = torch.from_numpy(y_data)

# Linear Regression model
class LinearRegression(nn.Module):

	# Constructor
	def __init__(self):
		super(LinearRegression, self).__init__()


		# nn.Linear applies a linear transformation to the data
		# y = xA^T + b 
		# Params are in_features and out_features, bias default True
		self.linear = nn.Linear(1, 1) # Input and output dims are 1


	# Feed forward nn 
	def forward(self, x):
		out = self.linear(x)
		return out 

model = LinearRegression()
criterion = nn.MSELoss()

# Use stochastic gradient descent with step size/delta = 0.0001
optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)

# Epochs & training
num_epochs = 1000
for epoch in range(num_epochs):
	X = x_train
	y = y_train

	# Move forward
	out = model(X)

	# Calculate loss
	loss = criterion(out, y)

	# Backward

	# Set to zero / initialize the optimizer
	optimizer.zero_grad()

	# This calculates the gradident using the criterion (nn.MSELoss() here)
	loss.backward()

	# All optimizers implement a step() method that updates params
	optimizer.step()

	# Print results
	if((epoch+1) % 10 == 0):
		print(f"Epoch {epoch+1} | Out {out} | Loss {loss.item()} | Optimizer {optimizer}")



# Now evaluate
model.eval() # Set to evaluation mode

with torch.no_grad():
	predict = model(X)

predict = predict.data.numpy()

fig = plt.figure(figsize=(10, 5))
plt.plot(X.numpy(), y.numpy(), 'ro', label = "Original data")
plt.plot(X.numpy(), predict, label = "Result")

# Show
plt.legend()
plt.show()

# Save state
torch.save(model.state_dict(), './01linear.pth')






