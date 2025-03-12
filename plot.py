import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Initialize wandb
wandb.init(project="DA6401 Assignments", name="fashion-mnist sample images")
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Define class labels
class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]
sample_images = {}
for img, label in zip(X_train, y_train):
    if label not in sample_images:
        sample_images[label] = img
    if len(sample_images) == 10:  # Stop when one sample image for eah class is found
        break

# Create a figure for visualization
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
fig.suptitle("Fashion-MNIST Sample Images", fontsize=14)

wandb_images = []  # List to store images for wandb logging

for ax, (label, img) in zip(axes.flatten(), sample_images.items()):
    ax.imshow(img, cmap="gray")
    ax.set_title(class_labels[label])
    ax.axis("off")

    # Add to wandb images list
    wandb_images.append(wandb.Image(img, caption=class_labels[label]))

# Log images to wandb
wandb.log({"fashion_mnist_samples": wandb_images})
plt.show()
wandb.finish()
