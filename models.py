import torch
from torch import nn
import torch.nn.functional as F
import torch.nn.init as init

class BaseCNNClassifier(nn.Module):
    """
    A simple CNN classifier for breast cancer ultrasound images.

    It has 3 convolutional layers and 3 fully connected layers.
    The output size of the convolutional layers is calculated and used to define the input size of the first fully connected layer.
    """
    def __init__(self):
        """
        Initializes a BaseCNNClassifier instance.

        Parameters:
            None

        Returns:
            nn.Module: The initialized BaseCNNClassifier class instance.
        """
        super().__init__()
        self.best_acc = 0

        # Setting conv layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=6, stride=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(64, 64, kernel_size=3, stride=1)
        )

        # Get the output size of the convolutional layers
        self.conv_output_size = len(torch.flatten(self.conv_layers(torch.rand(1, 1, 224, 224))))

        # Setting fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.conv_output_size, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, 3)
        )


    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Parameters:
            x (torch.Tensor): The input tensor to the neural network.

        Returns:
            torch.Tensor: The output of the neural network after passing through all the layers.
        """
        # Pass the input through the convolutional layers
        x = self.conv_layers(x)

        # Flatten the output
        x = torch.flatten(x, 1) 

        # Pass the output through the fully connected layers
        x = self.fc_layers(x)
        return x
    

class BasicBlock(nn.Module):
    expansion=1
    
    def __init__(self, in_planes, planes, stride):
        """
        Initializes a BasicBlock instance.

        Args:
            in_planes (int): The number of input planes.
            planes (int): The number of output planes.
            stride (int): The stride of the convolutional layers.

        Returns:
            BasicBlock (nn.Module): The initialized BasicBlock class instance.
        """
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        self.shortcut = nn.Sequential()
        if ((stride != 1) or (in_planes != self.expansion*planes)):
            self.shortcut = nn.Sequential(
                    nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(self.expansion*planes)
                    )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of a BasicBlock instance.

        Args:
            x (torch.Tensor): The input tensor to be processed.

        Returns:
            torch.Tensor: The output tensor after passing through the convolutional and batch normalization layers.
        """
        # First convolutional layer with batch normalization and ReLU activation
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # Second convolutional layer with batch normalization and ReLU activation
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Add the shortcut connection and apply ReLU activation
        out = out + self.shortcut(x)
        out = F.relu(out)
        
        return out
    

class ResNet(nn.Module):
    """
    The ResNet class is a PyTorch nn.Module class that represents a ResNet model.

    Args:
        block: A PyTorch nn.Module class that represents the building block of the ResNet model.
        num_blocks (list of int): A list of the number of blocks in each layer of the ResNet model.
        num_classes (int, optional): The number of classes in the classification task. Defaults to 3.
    """
    def __init__(self, block, num_blocks, num_classes=3):
        super(ResNet, self).__init__()
        
        # Set the best accuracy to 0 before training
        self.best_acc = 0
        self.in_planes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # The first convolutional layer is used to extract features from the input image.
        self.dropout1 = nn.Dropout(0.35)
        self.dropout2 = nn.Dropout(0.75)
        # Adding dropout layers for regularization
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # The max pooling layer is used to reduce the spatial dimensions of the feature maps.
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # The four layers of the ResNet model are defined here. Each layer is a sequence of BasicBlock instances.
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(512*block.expansion, num_classes)
        # The average pooling layer is used to reduce the spatial dimensions of the feature maps to a single value.
        # The fully connected layer is used to output the final probabilities of the classes.
        
        for m in self.modules():
            if(isinstance(m, nn.Conv2d)):
                # Initialize the weights of the convolutional layers using the Kaiming normal initialization method.
                init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                
            elif(isinstance(m, nn.BatchNorm2d)):
                # Initialize the weights of the batch normalization layers to 1 and the biases to 0.
                init.constant_(m.weight,1)
                init.constant_(m.bias,0)
                
                
    def _make_layer(self, block, planes, num_blocks, stride):
        """
        Makes a layer of the ResNet model.

        Args:
            block (nn.Module): The type of block to use in the layer.
            planes (int): The number of output planes of the layer.
            num_blocks (int): The number of blocks in the layer.
            stride (int): The stride of the first convolutional layer in the layer.

        Returns:
            nn.Sequential: The layer of the ResNet model.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        
        for stride in strides:
            # Create a new block and append it to the list of layers
            layers.append(block(self.in_planes, planes, stride))
            # Update the number of input planes for the next block
            self.in_planes = planes * block.expansion
        # Return the layer as a nn.Sequential module
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Defines the forward pass of the ResNet model.

        Args:
            x (torch.Tensor): The input tensor to be processed.

        Returns:
            torch.Tensor: The output tensor after passing through all the layers.
        """
        # First convolutional layer with batch normalization and ReLU activation
        # Followed by a max pooling layer to reduce the spatial dimensions of the feature maps
        x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = self.maxpool(x)
        
        # Dropout layer to randomly zero out 35% of the neurons during training
        x = self.dropout1(x)
        
        # Four residual layers with different number of output planes
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Dropout layer to randomly zero out 75% of the neurons during training
        x = self.dropout2(x)
        
        # Adaptive average pooling layer to reduce the spatial dimensions of the feature maps to a single value
        x = self.avgpool(x)
        
        # Flatten the output tensor to a 1D tensor
        x = torch.flatten(x, 1)
        
        # Fully connected layer to output the final probabilities of the classes
        x = self.fc1(x)
        return x

def ResNet18() -> ResNet:
    """
    Returns a ResNet18 model.

    The ResNet18 model is a ResNet model with 18 layers, and is commonly used as a baseline model for image classification tasks.

    Args:
        None

    Returns:
        ResNet (nn.Module): A ResNet18 model.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2])

def ResNet34() -> ResNet:
    """
    Returns a ResNet34 model.

    The ResNet34 model is a ResNet model with 34 layers, and is commonly used as a baseline model for image classification tasks.

    Args:
        None

    Returns:
        ResNet (nn.Module): A ResNet34 model.
    """
    return ResNet(BasicBlock, [3, 4, 6, 3])
