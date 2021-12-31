"""

Thanks to liaoxingyu

Source: https://github.com/L1aoXingyu/pytorch-beginner/blob/master/02-Logistic%20Regression/Logistic_Regression.py
"""

from torchvision import datasets
from torch.utils.data import DataLoader

# Model hyperparams
batch_size = 64
learning_rate = 1e-3
num_epochs = 100

######## 1 Obtain the data

# Obtain data
train_dataset = datasets.FashionMNIST(
    root='../datasets', train=True, transform=transforms.ToTensor(), download=True)

test_dataset = datasets.FashionMNIST(
    root='../datasets', train=False, transform=transforms.ToTensor())

# Use dataloader for training and testing

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

######## 2 Build the model

class LogisticRegression(nn.Module):
	def __ini__(self, in_dim, n_class):
		super(LogisticRegression, self)__init__()
		self.logistic = nn.Linear(in_dim, n_class)

	def forward(self, x):
		# Forward pass 
		out = self.logistic(x)
		return out


criterion = nn.CrossEntropyLoss() # For logistic...
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) # First param = iterable of params to optimize
model = LogisticRegression(28*28, 10)

####### 3 Use the model

for epoch in range(num_epochs):
	print(f'epoch {epoch+1}')

	# Other variables
	running_loss = 0.0
	running_acc = 0.0

	# Start training
	model.train()
	for i, data in enumerate(train_loader, 1):

		# Unpack data
		img, label = data

		# Resize
		img = img.view(img.size(0), -1)

		# Model & loss
		out = model(img)
		loss = criterion(out, label)
		running_loss += loss
		_, pred = torch.max(out, 1)
		running_acc += (pred==label).float().mean()

		# Reset optimizer
		optimizer.zero_grad()

		# Calculate gradients
		loss.backward()

		# Update gradients y' = y + delta; where delta = -grad * step_size
		optimizer.step()
