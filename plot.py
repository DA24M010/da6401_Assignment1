import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

wandb.init(project="DA6401 Assignments", name="sample_images")

(train_images, train_labels), (_, _) = fashion_mnist.load_data()

# Class labels in Fashion-MNIST
class_names = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

# Select one image per class
selected_images = []
selected_labels = []
for class_id in range(10):
    idx = np.where(train_labels == class_id)[0][0]  # Get first occurrence
    selected_images.append(train_images[idx])
    selected_labels.append(class_names[class_id])

# Log images to wandb
wandb_table = wandb.Table(columns=["Class", "Image"])
for label, img in zip(selected_labels, selected_images):
    wandb_table.add_data(label, wandb.Image(img))

# Log table to wandb
wandb.log({"Fashion-MNIST Samples": wandb_table})

# Finish wandb run
wandb.finish()
