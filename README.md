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


## ⚙️ How I Built It
Preprocessed images using TensorFlow's ImageDataGenerator

Built and compiled the CNN model with Adam optimizer and categorical crossentropy loss

Trained the model on the RGB images dataset

Visualized training progress using accuracy and loss plots

Evaluated the model using a classification report on test data



## The Problem I Wanted to Solve
Many beginner machine learning projects stop at MNIST or CIFAR datasets. I wanted to push myself further by working with a real-world, multi-class image classification problem.

Specifically, I built a model that can classify 139 plant species from images. This kind of classifier could be useful in agriculture, education, or even web apps for plant identification.


## 🛠️ Challenges & Learnings

### 🔧 Challenges
- **Hardware limitations:** Training on CPU took a long time, which limited experimentation speed. To address this, I plan to run the model on my NVIDIA laptop GPU or use cloud services like Google Colab for faster training in future iterations.  
- **Overfitting:** The model showed high training accuracy (~97%) but low validation accuracy (~30%), indicating overfitting due to insufficient or imbalanced data.

### 📚 Learnings
- Implemented custom preprocessing with `ImageDataGenerator` to handle dynamic image transformations during training.
- Gained hands-on experience designing CNN architectures and tuning hyperparameters.
- Learned to interpret classification reports for model evaluation beyond simple accuracy metrics.
- Realized the importance of using more diverse and balanced datasets to improve model generalization.




## 📊 Results
The model achieved a training accuracy of ~97%, but the validation accuracy was around 30%, indicating overfitting — it performed well on training data but struggled to generalize to unseen images.

Despite the low overall validation accuracy, the classification report showed strong performance on certain plant classes, suggesting the model was able to learn meaningful patterns for some categories.

This highlights the need for more diverse and balanced data across all classes to improve the model’s generalization.

