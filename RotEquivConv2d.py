import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Custom Rotate Filter Function ---
# Function to rotate a filter tensor by a specified angle and it uses affine transformation with grid sampling
# Input: filter - input filter tensor, angle - rotation angle, device - device for computation
# Output: rotated_filter - rotated filter tensor
def rotate_filter(filter, angle, device):
    # Calculate rotation matrix
    theta = torch.tensor([
        [np.cos(angle), -np.sin(angle), 0],
        [np.sin(angle), np.cos(angle), 0]
    ], dtype=torch.float).to(device)

    # Get the number of filters and the size of each filter
    N, C, H, W = filter.size()

    # Repeat and reshape theta to match the batch size of the filter tensor
    # The batch size after reshaping the filter is N * C
    theta = theta.repeat(N * C, 1, 1)

    # Adjust the shape of the filter for 4D input (combining batch and channel dimensions)
    reshaped_filter = filter.view(N * C, 1, H, W)

    # Create affine grid
    grid_size = reshaped_filter.size()
    grid = F.affine_grid(theta, grid_size, align_corners=False)

    # Apply grid sampling and reshape back to original
    rotated_filter = F.grid_sample(reshaped_filter, grid, align_corners=False)
    return rotated_filter.view(N, C, H, W)


# --- Custom convolutional layer with rotational equivariance ---
# Input: in_channels - input channels, out_channels - output channels,
#        kernel_size - size of the convolutional kernel, num_rotations - number of rotations
# Output: Convolutional layer with rotational equivariance
class RotEquivConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, num_rotations=8):
        super(RotEquivConv2d, self).__init__()
        self.num_rotations = num_rotations
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False)

    def forward(self, x):
        weight = self.conv.weight
        rotated_outputs = []
        for i in range(self.num_rotations):
            angle = 2 * np.pi * i / self.num_rotations
            rotated_weight = rotate_filter(weight, angle, weight.device)
            rotated_output = F.conv2d(x, rotated_weight, padding=self.conv.padding)
            rotated_outputs.append(rotated_output.unsqueeze(1))
        output = torch.cat(rotated_outputs, dim=1)
        return output.mean(dim=1)