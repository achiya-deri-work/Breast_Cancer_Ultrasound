# Breast Ultrasound Image Classification

## Overview

This project focuses on training models to classify breast ultrasound images into three categories: benign tumor, malignant tumor, and normal. We experimented with various neural network architectures to determine the best approach for this classification task.

## Dataset

The dataset used for this project is described in the following paper:

- **Citation**: Al-Dhabyani W, Gomaa M, Khaled H, Fahmy A. Dataset of breast ultrasound images. *Data in Brief*. 2020 Feb;28:104863. DOI: [10.1016/j.dib.2019.104863](https://doi.org/10.1016/j.dib.2019.104863).

## Model Architectures

I evaluated several models for this classification task:

1. **Convolutional Neural Network (CNN)**
   - **Architecture**: Three convolutional layers followed by multi-layer perceptron (MLP) layers.
   - **Performance**: Achieved 82% accuracy.

2. **ResNet Models**
   - **ResNet18**
     - **Architecture**: 18 layers deep residual network.
     - **Performance**: Achieved 87% accuracy.
   - **ResNet34**
     - **Architecture**: 34 layers deep residual network.
     - **Performance**: Achieved 84% accuracy.

## Results

The ResNet models consistently outperformed the simpler CNN model. Specifically:

- **ResNet18** achieved the highest accuracy of 87%.
- **ResNet34** achieved 84% accuracy.
- The simpler CNN model achieved 82% accuracy.

Despite the higher performance of ResNet models compared to the CNN, I observed that increasing the depth of the ResNet model did not always result in better performance. This is likely due to the relatively small size of the dataset, which can lead to overfitting with more complex models like ResNet34. 

## Regularization Methods

To improve model performance and prevent overfitting, I employed several regularization techniques:

- **Data Augmentation**: Enhanced the diversity of the training data by applying various transformations, helping the model generalize better to new, unseen data.
- **Dropout Layers**: Introduced dropout in the neural network to reduce overfitting by randomly setting a fraction of the input units to zero during training.
- **AdamW Optimizer**: Used AdamW (Adam with Weight Decay) optimizer instead of the standard Adam optimizer. AdamW improves regularization and helps in better generalization by decoupling weight decay from the optimization steps.

These regularization methods contributed to a significant improvement in model accuracy from an initial 75% to 82-87% across the tested models.

## Conclusion

ResNet models generally performed better than the CNN model for this classification task. The observed performance gains were largely due to effective use of regularization techniques. However, the limited dataset size suggests that expanding the dataset could lead to further improvements in model accuracy.

For more information and to contribute to the project, please feel free to reach out or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
