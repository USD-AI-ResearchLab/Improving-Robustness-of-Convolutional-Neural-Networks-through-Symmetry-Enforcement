import torch
import torch.nn as nn
import torch.nn.functional as F


# ---    Custom convolutional layer with scale equivariance ---
    # Input: in_channels - input channels, out_channels - output channels,
    #        kernel_size - size of the convolutional kernel, scales - list of scales
    # Output: Convolutional layer with scale equivariance
class ScaleEquivConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scales=[1.0, 0.8, 0.6], padding=0):
        super(ScaleEquivConv2d, self).__init__()
        self.scales = scales
        self.out_channels = out_channels
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.scaled_filters = self.create_scaled_filters()

    def create_scaled_filters(self):
        with torch.no_grad():
            weight = self.conv.weight
            original_size = weight.size()[2:]  # Spatial dimensions (H, W)
            scaled_filters = []

            for scale in self.scales:
                scaled_filter = F.interpolate(weight, scale_factor=scale, mode='bilinear', align_corners=False)
                resized_filter = F.interpolate(scaled_filter, size=original_size, mode='bilinear', align_corners=False)
                scaled_filters.append(resized_filter)

            # Move scaled filters to the same device as the weight
            return torch.cat(scaled_filters, dim=0).to(weight.device)

    def forward(self, x):
        device = x.device  # Get the device of the input tensor
        outputs = []
        split_size = self.out_channels
        for i in range(len(self.scales)):
            # Ensure scaled filters are on the same device as the input tensor
            scaled_filters_on_device = self.scaled_filters[i*split_size:(i+1)*split_size, :, :, :].to(device)
            output = F.conv2d(x, scaled_filters_on_device, padding=self.conv.padding)
            outputs.append(output)
        return sum(outputs)