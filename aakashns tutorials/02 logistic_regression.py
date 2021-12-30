"""
Rahul Goel
12/23/2021
Exploring stuff in pytorch...

Thanks to liaoxingyu: https://github.com/L1aoXingyu/pytorch-beginner/blob/master/02-Logistic%20Regression/Logistic_Regression.py
"""
# Imports
import torch
import torchvision
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
import torch.nn as nn

print("== Sec 1: Getting & Loading data ==")
# Download training dataset
dataset = MNIST(root='data/', download=True)

# Checking the length of the dataset
#print(f"Length of the dataset is {len(dataset)}")

# Get testing dataset, ~10k images
test_dataset = MNIST(root='data/', train=False)
#print(f"Length of test dataset is {len(test_dataset)}")

# Look at the first index in the large dataset
#print(f"Display image details...")
#print(dataset[0])

# Unpack and actually display the image
image_index = 5
image, label = dataset[image_index]
#plt.imshow(image, cmap='gray')
#plt.title(f"The digit is {label}")
#plt.show()


print("== Sec 2: Converting to tensors, creating dataloaders ==")
# Convert these PIL (Pillow) images into tensors
dataset = MNIST(root="data/",
				train=True,
				transform=transforms.ToTensor())

# Print out the new tensored data
img_tensor, label = dataset[0]
#print(img_tensor.shape, label)

#print(img_tensor[0, 10:15, 10:15])
#print(torch.max(img_tensor), torch.min(img_tensor))

#plt.imshow(img_tensor[0,10:15,10:15], cmap='gray')
#plt.show()

# Next steps... Training
# random_split will split into 2 sets randomly, scaled at the 
# relative numbers in the list provided
train_ds, val_ds = random_split(dataset, [50000, 10000])
print(f"After split, train: {train_ds}, val: {val_ds}")

# Next steps, using DataLoader
# Shuffle needed to train, but not to validate
batch_size = 128
train_loader = DataLoader(train_ds, batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size)

#===== Building the model! ========
print("== Sec 3: Builing model == ")
input_size = 28*28
num_classes = 10 # 10 digits...

# Logistic
model = nn.Linear(input_size, num_classes)

'''
print(model.weight.shape)
#print("==")
print(model.weight)

#print(f"\n == Print model biases ==")
print(model.bias.shape)
print(model.bias)
'''

'''
Taking first batch of 100 images from ds
to pass them to model
'''

'''
---

This is just one step
Note to self: this is basically like a linear reg
or in R, lm~factors...

Need to run this many times over large data
to create a model with appropriate weights
----

for images, labels in train_loader:
	print(" == Labels shape == ")
	print(labels.shape)
	print(" == Images shape ==")
	print(images.shape) # torch.Size([128, 1, 28, 28])
	outputs = model(images.reshape(128, 784)) # 1, 28, 28 ==> 128, 748 (to match model.weight.shape)
	print(outputs)
	break
	'''


# Create class for nn
class MnistModel(nn.Module):
	def __init__(self):
		super().__init__()
		self.linear = nn.Linear(input_size, num_classes)

	def forward(self, xb):

		# Note the -1 - let PyTorch get from original tensor
		# 784 is 28x28 (size out the inputs)
		
		xb = xb.reshape(-1, 784)

		out = self.linear(xb) 
		return out

	def training_step(self, batch):
		images, labels = batch
		out = self(images) # MnistModel(images)
		loss = F.cross_entropy(out, labels)
		return loss

	def validation_step(self, batch):
		images, labels = batch
		out = self(images)
		loss = F.cross_entropy(out, labels)
		acc = accuracy(out, labels)
		return {'val_loss': loss, 'val_acc': acc}

	def validation_epoch_end(self, outputs):
		batch_losses = [x['val_loss'] for x in outputs]
		epoch_loss = torch.stack(batch_losses).mean()
		batch_accs = [x['val_acc'] for x in outputs]
		epoch_acc = torch.stack(batch_accs).mean()

		return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

	def epoch_end(self, epoch, result):
		print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

model = MnistModel()


# Using some functions for evaluation...
def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
	# Select optimizer
	optimizer = opt_func(model.parameters(), lr)
	history = []


	for epoch in range(epochs):

		# Train
		for batch in train_loader:

			# Train and calculate loss using training_step:
			'''
			1. Unpacks the features and labels from batch
			-- a train_loader

			2. Uses the self(images) call to train the model

			3. Calculates cross-entropy loss, since this is probabilistic

			4. Returns loss

			'''
			loss = model.training_step(batch)

			# Compute gradients
			loss.backward()

			# Update weights
			optimizer.step()

			# Reset gradients
			optimizer.zero_grad()

		# Validation
		result = evaluate(model, val_loader)
		model.epoch_end(epoch, result)
		history.append(result)
	
	return history


result0 = evaluate(model, val_loader)
print(f"Results: {result0}")


# First training cycle
history1 = fit(5, 0.001, model, train_loader, val_loader)

'''
# Second training cycle
history2 = fit(15, 0.001, model, train_loader, val_loader)

# Third training cycle
history2 = fit(5, 0.0001, model, train_loader, val_loader)
'''


# Testing with individual images

# Get an image
# Pass to model to get the label/prediction
# Compare to actual

test_dataset = MNIST(root='data/', 
                     train=False,
                     transform=transforms.ToTensor())



def predict_image(model, img):
	x = img.unsqueeze(0) # 1x28x28 -> 1x1x28x28 batch with single image
	y = model(x)

	_, preds = torch.max(y, dim=1)

	return preds[0].item()

img, label = test_dataset[0]

plt.imshow(img[0])
plt.title(f"Number is {label} | Predicted as {predict_image(model, img)}")
plt.show()

