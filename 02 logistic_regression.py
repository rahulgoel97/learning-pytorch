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
import torch.nn as nn

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


print("==== Section 2 ====")
# Convert these PIL (Pillow) images into tensors
dataset = MNIST(root="data/",
				train=True,
				transform=transforms.ToTensor())

# Print out the new tensored data
img_tensor, label = dataset[0]
print(img_tensor.shape, label)

print(img_tensor[0, 10:15, 10:15])
print(torch.max(img_tensor), torch.min(img_tensor))

plt.imshow(img_tensor[0,10:15,10:15], cmap='gray')
plt.show()

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
input_size = 28*28
num_classes = 10 # 10 digits...

# Logistic
model = nn.Linear(input_size, num_classes)

print(model.weight.shape)
print("==")
print(model.weight)
