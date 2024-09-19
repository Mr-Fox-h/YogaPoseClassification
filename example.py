from icecream import ic
import tensorflow as tf
from tensorflow.keras import models
import matplotlib.pyplot as plt
import datasets as dt
import numpy as np

# Icecream configuration
ic.configureOutput(prefix='Status|> ')

# Load the test dataset
example_data = dt.load_dataset("imagefolder", data_dir = "./examples", split = "train")
row_images = example_data['image']

# Transform
converted_images = [img.convert("L").resize((124, 124)) for img in row_images]

# Converting into tensor
images_tensor = []
show = []

for index in converted_images:
    images_tensor.append(tf.convert_to_tensor(index) / 255)
    show.append(np.array(index) / 255)

images_tensor = tf.convert_to_tensor(images_tensor)
show = np.array(images_tensor)

# Class names of datasets
class_names = ['Downdog', "Goddess", "Plank", "Tree", "Warrior2"]
ic(class_names)

# Load the model
ic("Loading the model")

model = models.load_model('YogaPoseClassification.h5')

# Prediction
ic("Predicting")
prd = model.predict(images_tensor)

# Show img the data
ic("Rendering the images")
fig, axes = plt.subplots(1, len(class_names), figsize=(15, 5))
for index, ax in enumerate(axes):
    prd_guess = class_names[np.argmax(prd[index])]
    ax.grid(False)
    ax.imshow(show[index], cmap=plt.cm.binary)
    ax.set_title(f"Prediction: {prd_guess}")
    ax.axis('off')
plt.tight_layout()
plt.show()
