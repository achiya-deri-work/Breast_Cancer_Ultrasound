# Breast Ultrasound Image Classification

## Overview

This project focuses on training models to classify breast ultrasound images into three categories: benign tumor, malignant tumor, and normal. We experimented with various neural network architectures, including custom CNN models as well as both pre-trained and non-pre-trained ResNet models, to determine the best approach for this classification task.

## Dataset

The dataset used for this project is described in the following paper:

- **Citation**: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. *Data in Brief*. 2020 Feb;28:104863. DOI: [10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863).

This dataset is relatively small, containing only 780 images. Given the limited size, training models from scratch can lead to overfitting and suboptimal performance. To overcome this, we leveraged both custom architectures and pre-trained models.

## Model Architectures

We evaluated several models for this classification task:

### 1. Custom Convolutional Neural Network (CNN)
This basic CNN model consists of three convolutional layers followed by fully connected layers. It serves as a baseline model for comparison with more complex architectures.

- **Performance**: Achieved 82% accuracy.

### 2. Non-Pre-trained ResNet Models
These models were trained from scratch on the breast cancer dataset without using any pre-trained weights.

- **ResNet18 (Non-pre-trained)**:
  - **Architecture**: 18 layers deep residual network.
  - **Performance**: Achieved 87% accuracy.

- **ResNet34 (Non-pre-trained)**:
  - **Architecture**: 34 layers deep residual network.
  - **Performance**: Achieved 84% accuracy.

### 3. Pre-trained ResNet Models
These models were initialized with pre-trained weights from ImageNet and fine-tuned for our classification task by replacing the final fully connected (FC) layer with a 3-class output layer (benign, malignant, normal).

- **ResNet34 (Pre-trained on ImageNet)**:
  - **Architecture**: 34 layers deep residual network.
  - **Performance**: Achieved 93% accuracy.

- **ResNet50 (Pre-trained on ImageNet)**:
  - **Architecture**: 50 layers deep residual network.
  - **Performance**: Achieved 95% accuracy.

## Why Use Pre-trained Models?

Using pre-trained models is particularly beneficial when dealing with small datasets like ours (780 images). Pre-trained models have already learned useful features from large datasets, such as ImageNet, which contains over a million images and 1000 classes. By fine-tuning these models on our specific task, we leverage their ability to extract meaningful patterns from images, even when the new dataset is small.

The pre-trained models, ResNet34 and ResNet50, significantly outperformed the non-pre-trained ResNet18 and ResNet34 models due to the use of pre-learned feature representations.

## Optimizers

- **Adam Optimizer**: For the pre-trained models (ResNet34 and ResNet50), we used the **Adam optimizer**. We found that using the **AdamW optimizer** caused a loss of the pre-learned feature representations, which negatively impacted the model's ability to generalize.
- **AdamW Optimizer**: This optimizer was used for non-pre-trained models and custom architectures, where regularization through weight decay was important to prevent overfitting.

## Results

The performance of the models is summarized as follows:

### Basic CNN Model
- **Custom 3-layer CNN** achieved 82% accuracy.

### Non-Pre-trained Models
- **ResNet18** achieved 87% accuracy.
- **ResNet34** achieved 84% accuracy.

### Pre-trained Models
- **ResNet34** achieved 93% accuracy.
- **ResNet50** achieved the highest accuracy of 95%.

The pre-trained ResNet34 and ResNet50 models showed significant improvements over the models trained from scratch due to their ability to leverage pre-learned feature extraction.

## Regularization Methods

To further improve model performance and prevent overfitting, we used the following regularization techniques:

- **Data Augmentation**: Applied various transformations to the training data to improve generalization.
- **Dropout Layers**: Reduced overfitting by randomly dropping a fraction of the input units during training.
- **AdamW Optimizer**: For non-pre-trained models, we used **AdamW** to decouple weight decay from the optimization step, which provided better regularization and prevented overfitting, particularly with the smaller dataset.

These techniques contributed to boosting the model accuracy to as high as 95%.

## Conclusion

The custom 3-layer CNN model serves as a solid baseline but was outperformed by both non-pre-trained and pre-trained ResNet models. The use of pre-trained ResNet models, particularly ResNet34 and ResNet50, led to significant improvements in the classification of breast ultrasound images. By using the Adam optimizer, we preserved the pre-learned understandings of these models. This project demonstrates the power of transfer learning in medical image classification tasks where data is limited.

For more information and to contribute to the project, please feel free to reach out or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).

