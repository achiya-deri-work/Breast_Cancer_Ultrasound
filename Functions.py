import torch

"""
Contains various functions for data processing.

Functions:
    one_hot: Generates a one-hot encoded tensor from the given tensor `y`.
    randint_distinct: Returns a tensor of `n_samples` distinct random integers in the range [a, b].
    accuracy: Calculates the accuracy between predicted labels and true labels.
    augmant_data: Data augmentation for the input data given 'transforms'.
"""


def one_hot(y, num_classes=3) -> torch.Tensor:
    """
    Generates a one-hot encoded tensor from the given tensor `y`.

    Parameters:

        y (torch.Tensor): The input tensor.
        num_classes (int, optional): The number of classes. Defaults to 3.

    Returns:
        torch.Tensor: The one-hot encoded tensor.

    Note: This function assumes that the input tensor `y` contains integer values
    between 0 and `num_classes - 1`.
    """
    # Get the shape of the input tensor
    batch_size = y.shape[0]

    # Create a tensor filled with zeros, with shape (batch_size, num_classes)
    y_one_hot = torch.zeros(batch_size, num_classes)

    # Set the corresponding element in each row of y_one_hot to 1 based on the
    # value of y
    y_one_hot[torch.arange(batch_size), [i for i in y]] = 1
    return y_one_hot


def accuracy(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    Calculates the accuracy between predicted labels and true labels.

    Parameters:
        y_pred (torch.Tensor): The predicted labels.
        y_true (torch.Tensor): The true labels.

    Returns:
        torch.Tensor: The accuracy value between y_pred and y_true.

    Note: Assumes that y_pred and y_true are one-hot encoded tensors of shape
    (batch_size, num_classes), where num_classes is the number of classes.
    """
    # Get the index of the maximum value in each row of y_pred and y_true
    # This is equivalent to getting the class label with the highest probability
    y_pred = torch.argmax(y_pred, dim=1)
    y_true = torch.argmax(y_true, dim=1)

    # Calculate the accuracy by comparing y_pred and y_true
    # The accuracy is the proportion of correctly classified samples
    acc = torch.tensor(torch.sum(y_pred == y_true).item() / len(y_pred))

    return acc


def randint_distinct(a: int, b: int, n_samples: int) -> torch.Tensor:
    """
    Returns a tensor of `n_samples` distinct random integers in the range [a, b].

    Parameters:
        a (int): The start of the range (inclusive).
        b (int): The end of the range (inclusive).
        n_samples (int): The number of distinct random integers to generate.

    Returns:
        torch.Tensor: A tensor of `n_samples` distinct random integers in the range [a, b].
    """
    # Generate all possible values in the range [a, b] and shuffle them
    all_values = torch.arange(a, b + 1)
    shuffled_values = all_values[torch.randperm(b - a + 1)]
    # Return the first n_samples values from the shuffled tensor
    return shuffled_values[:n_samples]


def augmant_data(images: torch.Tensor, labels: torch.Tensor, transforms, times: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Augments the given images and labels with the given transforms and returns the augmented data.

    Parameters:
        images (torch.Tensor): The images to be augmented. Should be a torch tensor of shape (N, C, H, W) where N is the number of samples, C is the number of channels, H is the height of the image, and W is the width of the image.
        labels (torch.Tensor): The labels corresponding to the images. Should be a torch tensor of shape (N, ) where N is the number of samples.
        transforms: The transforms to apply to the images. Should be a callable that takes an image and returns the transformed image.
        times (int, optional): The number of times to apply the transforms to each image. Defaults to 2.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: The augmented images and labels. The shape of the augmented images is (N*times, C, H, W), and the shape of the augmented labels is (N*times, ).
    """

    # Get device of images
    device = images.device
    images, labels = images.to("cpu"), labels.to("cpu")

    # Initialize empty tensors to store the augmented data
    X = torch.tensor([], dtype=torch.float32, device=images.device)
    y = torch.tensor([], dtype=torch.int16, device=labels.device)

    # Loop over the images and labels, and apply the transforms
    for i in range(len(images)*times):
        # Apply the transform to the current image
        X = torch.cat((X, transforms(images[int(i%len(images))].squeeze(0)).unsqueeze(0)), dim=0)
        # Append the label to the labels tensor
        y = torch.cat((y, labels[int(i%len(labels))].unsqueeze(0)), dim=0)

    # Return the augmented data
    return X.to(device), y.to(device)

