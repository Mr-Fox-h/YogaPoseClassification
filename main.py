from icecream import ic
import tensorflow as tf
from tensorflow.keras import layers, models
import datasets as dt
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Icecream configuration
ic.configureOutput(prefix='Status|> ')

# Load the dataset
ic("Loading dataset")

whole_data = dt.load_dataset("imagefolder", data_dir = "./datasets")
image_data = whole_data['train'].select_columns('image')
label_data = whole_data['train'].select_columns('label')

# Train/Test split
image_train, image_test, label_train, label_test = train_test_split(image_data['image'], label_data["label"], test_size = 0.2)
train_label = tf.convert_to_tensor(label_train)
test_label = tf.convert_to_tensor(label_test)

# Transform
ic("Transforming data")

train_img = [img.convert("L").resize((124, 124)) for img in image_train]
test_img = [img.convert("L").resize((124, 124)) for img in image_test]

# Converting into tensor
train_tensor = []
test_tensor = []

ic("Converting/Normalizing data into tensor")

for index in train_img:
    train_tensor.append(tf.convert_to_tensor(index) / 255)

for index in test_img:
    test_tensor.append(tf.convert_to_tensor(index) / 255)

train_tensor = tf.convert_to_tensor(train_tensor)
test_tensor = tf.convert_to_tensor(test_tensor)

# CNN Model
ic("The CNN model")

model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation = 'relu', input_shape = (124, 124, 1), ))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(24, activation='sigmoid'))
model.add(layers.Dense(5, activation='softmax'))

ic("Optimizer/Loss/metrics settup")
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
ic("Feeding the train data data")
history = model.fit(train_tensor, train_label, epochs = 15)

# Loss/Accuracy calculatation
ic("Loss/Accuracy calculatation")
loss, accuracy = model.evaluate(test_tensor, test_label)
ic(loss, accuracy)

# Saving the model
ic("Saving the model")
model.save("YogaPoseClassification.h5")

# Plot accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(loc='upper left')

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(loc='upper left')
plt.show()
