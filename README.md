Project: Breast Ultrasound Image Classification

Objective:

This project aims to develop a machine learning model capable of accurately classifying breast ultrasound images into three categories: benign tumor, malignant tumor, and normal. The goal is to assist in the early detection and diagnosis of breast cancer.

Model Architectures and Performance:

Several deep learning models were explored, including Convolutional Neural Networks (CNNs) and Residual Networks (ResNets). The key findings were:

ResNet Models Outperform CNN: ResNet18 achieved the highest accuracy of 87%, surpassing the CNN model's 82% accuracy. ResNets' ability to handle deeper networks without vanishing gradients is crucial for learning complex patterns in medical images.
Dataset Size and Overfitting: While ResNet34 has a deeper architecture than ResNet18, it did not lead to improved performance. This is likely due to the relatively small size of the breast ultrasound dataset, which can cause larger models to overfit the training data.
Regularization Techniques and Impact:

Regularization techniques were employed to enhance model generalization and prevent overfitting. These methods included:

Data Augmentation: Applying transformations like rotations, flips, and scaling to the training data increases its diversity and helps the model learn more robust features.
Dropout Layers: Randomly dropping neurons during training introduces noise and prevents the model from relying too heavily on any particular feature.
AdamW Optimizer: Using AdamW instead of the standard Adam optimizer often leads to improved convergence and stability, particularly in deep learning models.
These regularization techniques contributed to a significant improvement in model performance, raising the accuracy from around 75% to 82-87-84%.
