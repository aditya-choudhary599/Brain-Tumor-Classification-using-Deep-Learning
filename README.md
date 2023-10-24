# Brain Tumor Classification using Deep Learning

## Table of Contents
- [Introduction](#introduction)
- [Data Preparation](#data-preparation)
  - [Data Source](#data-source)
  - [Data Preprocessing](#data-preprocessing)
- [Data Exploration](#data-exploration)
  - [Data Visualization](#data-visualization)
  - [Class Distribution](#class-distribution)
  - [Image Dimensions](#image-dimensions)
- [Model Architectures](#model-architectures)
  - [Model 1: Linear Layer](#model-1-linear-layer)
  - [Model 2: Convolutional Neural Network (CNN)](#model-2-convolutional-neural-network-cnn)
  - [Model 3: Custom ResNet-18](#model-3-custom-resnet-18)
  - [Model 4: Pretrained ResNet-34](#model-4-pretrained-resnet-34)
- [Model Training](#model-training)
  - [Hyperparameters](#hyperparameters)
- [Model Evaluation](#model-evaluation)
  - [Evaluation Metrics](#evaluation-metrics)
  - [Confusion Matrices](#confusion-matrices)
- [Conclusion and Results](#conclusion-and-results)
- [Future Directions](#future-directions)

## Introduction
Welcome to the Brain Tumor Classification project! This project focuses on using deep learning to classify Brain Tumor MRI images into four categories: Glioma Tumor, Meningioma Tumor, No Tumor, and Pituitary Tumor. The primary objective is to develop models that can assist medical professionals in diagnosing brain tumors accurately and efficiently.

## Data Preparation
### Data Source
- The dataset used in this project was sourced from Kaggle using the `opendatasets` library. It consists of MRI images of patients with and without brain tumors.
- The dataset was split into a training set and a testing set, each containing images of the four tumor types.

### Data Preprocessing
- Data preprocessing is a critical step in preparing the dataset for model training.
- The following preprocessing steps were applied:
  - **Image Resizing**: Images were resized to a common size to ensure uniformity.
  - **Center Cropping**: Center cropping was performed to focus on the central part of the image.
  - **Data Augmentation (Training Set)**: Data augmentation techniques, including random horizontal flips and random rotations, were applied to the training set to increase data diversity.

## Data Exploration
### Data Visualization
- Visualization of the data is crucial for understanding the nature of the images and the challenges associated with brain tumor classification.
- Sample images from the training dataset were visualized to gain insights into the data.

### Class Distribution
- Analyzing the class distribution in both the training and testing datasets is important. This distribution can significantly impact the performance of machine learning models.
- The distribution of the four tumor types was visualized to understand class balance.

### Image Dimensions
- Examining the dimensions of the images in the dataset is essential to ensure uniformity in size. This information is crucial for setting the input size for model training.

## Model Architectures
In this project, multiple model architectures were explored and implemented for brain tumor classification. Here are the key models used:

### Model 1: Linear Layer
- A simple model with fully connected linear layers.
- The architecture included flattening the input image, two linear layers with ReLU activation, and a final linear layer for classification.

### Model 2: Convolutional Neural Network (CNN)
- A convolutional neural network (CNN) architecture was used, consisting of convolutional layers with max-pooling, followed by fully connected layers.
- CNNs are known for their ability to capture spatial features in images.

### Model 3: Custom ResNet-18
- A custom ResNet-18 architecture was constructed from scratch. This architecture included residual blocks to facilitate the training of deep networks and improve accuracy.

### Model 4: Pretrained ResNet-34
- A pre-trained ResNet-34 model from the torchvision library was fine-tuned for brain tumor classification.
- Transfer learning was applied to leverage knowledge learned from a large dataset for improved classification performance.

## Model Training
- Model training is a critical phase in developing effective deep learning models. In this project, the models were trained using the one-cycle learning rate policy.
- The one-cycle policy involves gradually increasing the learning rate and then annealing it, resulting in improved training and convergence.

### Hyperparameters
- Hyperparameters such as learning rate, weight decay, and gradient clipping were fine-tuned for optimal model performance.
- The models were trained for a fixed number of epochs (12 epochs) to ensure convergence.

## Model Evaluation
- Model evaluation is essential to assess the performance of each model. The evaluation phase included the following steps:

### Evaluation Metrics
- Evaluation metrics such as accuracy, precision, recall, and F1-score were calculated to measure the effectiveness of the models.
- These metrics provide insights into the model's ability to correctly classify brain tumor images.

### Confusion Matrices
- Confusion matrices were plotted to visualize the model's performance in classifying each tumor type.
- These matrices help identify any class-specific issues in classification.

## Conclusion and Results
- After extensive model training and evaluation, it was determined that Model 4, which utilized a pre-trained ResNet-34 architecture, achieved the highest accuracy and outperformed the other models.
- This project demonstrates the effectiveness of deep learning in medical image classification, specifically in the context of brain tumor diagnosis.
- The best-performing model can be saved and used for real-world tumor classification tasks, potentially assisting medical professionals in diagnosing brain tumors more accurately and efficiently.

## Future Directions
- The project's success opens the door to further improvements and future work. Here are some possible directions:

### Advanced Architectures and Techniques
- Explore more advanced architectures and techniques, including ensembling multiple models to improve classification accuracy.

### Dataset Expansion
- Expand the dataset by collecting more brain tumor MRI images. A larger dataset can lead to more robust models.

### Deployment
- Develop a user-friendly interface for medical professionals to use the model in a clinical setting for real-time brain tumor diagnosis.

This project is a significant step forward in the field of medical image analysis and has the potential to make a meaningful impact on the early and accurate diagnosis of brain tumors.
