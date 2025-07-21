# 🌿 Plant Classifier Model

A simple deep learning model built with **TensorFlow** to classify plant species based on input images. This project demonstrates the use of convolutional neural networks (CNNs) for image classification tasks and is trained on a dataset sourced from **Kaggle**.


## 📂 Dataset

The dataset was obtained from [Kaggle]([https://www.kaggle.com/](https://www.kaggle.com/datasets/omrathod2003/140-most-popular-crops-image-dataset?resource=download-directory&select=140_crops_list.txt)) and includes labeled images of various plant species. Each image is categorized under a specific plant class.
https://www.kaggle.com/datasets/omrathod2003/140-most-popular-crops-image-dataset?resource=download-directory&select=140_crops_list.txt

## 🏗️ Model Architecture

The model is built using TensorFlow and Keras with the following layers:

- **Input Layer**: Image preprocessing and resizing  
- **Convolutional Layers**: Feature extraction  
- **Pooling Layers**: Dimensionality reduction  
- **Dense Layers**: Classification  
- **Output Layer**: Softmax activation for multi-class prediction
