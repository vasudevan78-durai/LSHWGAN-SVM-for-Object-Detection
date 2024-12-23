# LSHWGAN-SVM-for-Object-Detection
-Dynamic data selection strategy of the LSHWGAN system.	
-Hierarchical Walk operation in LSGAN
-Weight parameter updating process and model synchronization in LSHWGAN

import numpy as np

def update_global_weights(W_local, Q):
    """
    Weight parameter updating process and model synchronization in LSHWGAN.

    Parameters:
    W_local: list of lists, where each inner list W_j contains the weights of a local model.
    Q: list of batch sizes for each local model (or edge node).

    Returns:
    W_global: list representing the new version of the global weight set.
    """
    # Step 1: Initialize the global weights W^(i) to zero with the same structure as W_local
    num_weights = len(W_local[0])
    W_global = np.zeros(num_weights)

    # Step 2-4: Update global weight parameters by iterating through each local weight set
    for j, W_j in enumerate(W_local):  # Loop through each local weight set W_j^(i)
        for k, w_jk in enumerate(W_j):  # Loop through each weight parameter w_(j,k)
            W_global[k] += w_jk * Q[j]  # Update global weight with weighted sum

    # Step 5: Return the global weight set W^(i)
    return W_global

# Example usage
# List of local weights from 3 edge nodes (each has a list of weights)
W_local = [
    [0.1, 0.2, 0.3],  # Weights from node 1
    [0.4, 0.5, 0.6],  # Weights from node 2
    [0.7, 0.8, 0.9]   # Weights from node 3
]

# Batch sizes for each node
Q = [10, 20, 30]

# Compute the updated global weights
W_global = update_global_weights(W_local, Q)
print("Updated Global Weights:", W_global)

# Preprocessing 

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os

# Define transformations
transform = transforms.Compose([
    transforms.Resize((64, 64)),  # Resize images to 64x64
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])  # Normalize to [-1, 1] range for GANs
])

# Custom Dataset to load images from a directory
class SimpleImageDataset(Dataset):
    def _init_(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.transform = transform
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder) if img.endswith(('.png', '.jpg', '.jpeg'))]

    def _len_(self):
        return len(self.image_paths)

    def _getitem_(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, 0  # We return 0 as a dummy label for compatibility

# Upload images to Colab
from google.colab import files
uploaded = files.upload()

# Create a directory for the images
os.makedirs('uploaded_images', exist_ok=True)

# Save uploaded files to directory
for filename in uploaded.keys():
    with open(f'uploaded_images/{filename}', 'wb') as f:
        f.write(uploaded[filename])

# Initialize the dataset and dataloader
dataset = SimpleImageDataset(image_folder='uploaded_images', transform=transform)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# Example: Display the number of images
print(f"Total images loaded: {len(dataset)}")

# Check one batch
for batch in dataloader:
    images, _ = batch
    print(f"Batch size: {images.shape}")
    break  # Remove this break if you want to loop through all batches

# Accuracy over Epochs using LSHWGAN

import matplotlib.pyplot as plt

# Sample data for accuracy and epochs
epochs = list(range(1, 21))  # Example epoch numbers (1 to 20)
accuracy = [0.5, 0.55, 0.6, 0.62, 0.65, 0.68, 0.7, 0.72, 0.75, 0.77,
            0.78, 0.8, 0.82, 0.83, 0.85, 0.86, 0.88, 0.89, 0.9, 0.95]
 # Example accuracy values

# Plotting the accuracy vs epochs graph
plt.figure(figsize=(10, 6))
plt.plot(epochs, accuracy, marker='o', color='b', label='Accuracy')
plt.title('Model Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

# Save the plot as an image
plt.savefig('accuracy_vs_epochs.png')
plt.show()




#Accuracy Comparison with various models using LSHWGAN
import matplotlib.pyplot as plt
 # Example data for three variables 
modules = ['UCSD PED-1', 'Shanghai Tech', 'Custom Data Set'] 
LSHWGAN = [0, 89, 95.67]
LSGAN = [0, 80, 92.46]
DCGAN = [0, 70, 85.67]
SGAN = [0, 70, 84.45]
 # Plotting plt.figure(figsize=(10, 6)) 
plt.plot(modules, LSHWGAN, marker='o', color='green', linestyle='-', label='LSHWGAN') 
plt.plot(modules, LSGAN, marker='o', color='blue', linestyle='-', label='LSGAN')
plt.plot(modules, DCGAN, marker='o', color='red', linestyle='-', label='DCGAN') 
plt.plot(modules, SGAN, marker='o', color='red', linestyle='-', label='SGAN') 
 # Adding labels and title 
plt.xlabel('Data Sets') 
plt.ylabel('Accuracy (%)') 
plt.title('Accuracy of Different Models on Various Datasets')
 # Set efficiency range from 0 to 100
plt.ylim(0, 100) 
# Adding a legend 
plt.legend()
 # Display the plot 
plt.show()


#Accuracy Comparison with various models (LSGAN, DCGAN, SGAN, HDCNN, DCNN) using LSHWGAN

#Accuracy Comparison with various models using LSHWGAN
import matplotlib.pyplot as plt
 # Example data for three variables 
modules = ['UCSD PED-1', 'Shanghai Tech', 'Custom Data Set'] 
LSHWGAN = [0, 89, 95.67]
LSGAN = [0, 80, 92.46]
DCGAN = [0, 70, 85.67]
SGAN = [0, 70, 84.45]
HDCNN = [0, 68, 80.02]
DCNN = [0, 66, 78.12]
 # Plotting plt.figure(figsize=(10, 6)) 
plt.plot(modules, LSHWGAN, marker='o', color='green', linestyle='-', label='LSHWGAN') 
plt.plot(modules, LSGAN, marker='o', color='blue', linestyle='-', label='LSGAN')
plt.plot(modules, DCGAN, marker='o', color='red', linestyle='-', label='DCGAN') 
plt.plot(modules, SGAN, marker='o', color='red', linestyle='-', label='SGAN') 
plt.plot(modules, HDCNN, marker='o', color='Yellow', linestyle='-', label='HDCNN') 
plt.plot(modules, DCNN, marker='o', color='magenta', linestyle='-', label='DCNN') 
 # Adding labels and title 
plt.xlabel('Data Sets') 
plt.ylabel('Accuracy (%)') 
plt.title('Accuracy of Different Models on Various Datasets')
 # Set efficiency range from 0 to 100
plt.ylim(0, 100) 
# Adding a legend 
plt.legend()
 # Display the plot 
plt.show(


