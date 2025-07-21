# 🌿 Plant Classifier Model

A simple deep learning model built with **TensorFlow** to classify plant species based on input images. This project demonstrates the use of convolutional neural networks (CNNs) for image classification tasks and is trained on a dataset sourced from **Kaggle**.

This is a simple project in helping me understand how to use Tensorflow and such. I am also planning on creating a web app using this model I created.

**NOTE: The GitHub does not have the model as it is too large to be saved on to the repo which is why the training file and link to the dataset has been given**

## 📂 Dataset

The dataset was obtained from [Kaggle]([https://www.kaggle.com/](https://www.kaggle.com/datasets/omrathod2003/140-most-popular-crops-image-dataset?resource=download-directory&select=140_crops_list.txt)) and includes labeled images of various plant species. Each image is categorized under a specific plant class.
https://www.kaggle.com/datasets/omrathod2003/140-most-popular-crops-image-dataset?resource=download-directory&select=140_crops_list.txt

## 🏗️ Model Architecture

The plant classifier model is a Convolutional Neural Network (CNN) built using TensorFlow's Keras `Sequential` API. It consists of the following layers:

1. **Conv2D Layer**  
   - 32 filters  
   - Kernel size: 3x3  
   - Activation: ReLU  
   - Input shape: `(224, 224, 3)` (RGB images resized to 224x224)

2. **MaxPooling2D Layer**  
   - Pool size: 2x2 (reduces spatial dimensions)

3. **Conv2D Layer**  
   - 64 filters  
   - Kernel size: 3x3  
   - Activation: ReLU

4. **MaxPooling2D Layer**  
   - Pool size: 2x2

5. **Flatten Layer**  
   - Flattens 2D feature maps into 1D vector

6. **Dense Layer**  
   - 128 units  
   - Activation: ReLU

7. **Output Dense Layer**  
   - Number of units equals the number of plant classes (`train_data.num_classes`)  
   - Activation: Softmax (for multi-class classification)

### Model Compilation

- **Optimizer:** Adam  
- **Loss function:** Categorical Crossentropy  
- **Metrics:** Accuracy
