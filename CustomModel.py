import torch
import torch.nn as nn
import torch.nn.functional as F
from RotEquivConv2d import RotEquivConv2d
from ScaleEquivConv2d import ScaleEquivConv2d

# --- Custom neural network model combining rotational and scale equivariant layers ---
# Input: num_classes - number of output classes, device - device for computation
# Output: Neural network model
class CustomModel(nn.Module):
    def __init__(self, num_classes=100, device='cuda'):
        super(CustomModel, self).__init__()
        self.device = device

        # Standard Convolutional Layer with increased filters
        self.conv1 = nn.Conv2d(3, 128, kernel_size=5, padding=2).to(device)  # Increased filters and larger kernel
        self.bn1 = nn.BatchNorm2d(128).to(device)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Rotation Equivariant Convolutional Layer with increased filters
        self.rot_equiv_conv1 = RotEquivConv2d(128, 256, kernel_size=3).to(device)  # Increased filters
        self.bn2 = nn.BatchNorm2d(256).to(device)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Scale Equivariant Convolutional Layer with increased filters
        self.scale_equiv_conv = ScaleEquivConv2d(256, 512, kernel_size=3, padding=1).to(device)  # Increased filters
        self.bn3 = nn.BatchNorm2d(512).to(device)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Additional Rotation Equivariant Convolutional Layer with increased filters
        self.rot_equiv_conv2 = RotEquivConv2d(512, 1024, kernel_size=3).to(device)  # Increased filters
        self.bn4 = nn.BatchNorm2d(1024).to(device)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully Connected Layer
        self.fc_input_features = 1024 * 2 * 2  # Adjust as per the final feature map size
        self.fc = nn.Linear(self.fc_input_features, num_classes).to(device)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.rot_equiv_conv1(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.scale_equiv_conv(x)))
        x = self.pool3(x)
        x = F.relu(self.bn4(self.rot_equiv_conv2(x)))
        x = self.pool4(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x
