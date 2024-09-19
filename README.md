# Yoga Pose Classification

## Overview
This project implements a Convolutional Neural Network (CNN) for Yoga Pose Classification. The model is trained to recognize and classify five different yoga poses:
- Downdog
- Goddess
- Plank
- Tree
- Warrior2

## Setup
1. Clone the repository:
  ```bash
  $ git clone https://github.com/your-username/yoga-pose-classification.git
  ```
2.Navigate to the project directory:
  ```bash
  $ cd yoga-pose-classification
  ```
3. Install the required dependencies.
  ```bash
  $ pip install -r requirements.txt
  ```

## Usage
1. Prepare your dataset: Place your images in the `./datasets` directory.
2. The [dataset](https://www.kaggle.com/datasets/ujjwalchowdhury/yoga-pose-classification) I used.
3. Run the script:
  ```bash
  $ python main.py
  ```
  This will train the model, evaluate its performance, and save the trained model as `YogaPoseClassification.h5`.

## Model
- The CNN model consists of the following layers:
- Conv2D (128 filters)
- MaxPooling2D
- Conv2D (64 filters)
- MaxPooling2D
- Conv2D (64 filters)
- Flatten
- Dense (32 units)
- Dense (24 units)
- Dense (5 units with softmax activation)

## Result
The model's performance will be printed to the console and visualized in plots showing accuracy and loss over epochs.

## License
This project is licensed under the **MIT** License. See the LICENSE file for details.
