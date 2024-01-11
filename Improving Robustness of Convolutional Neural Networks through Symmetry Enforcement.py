import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

# Custom Rotate Filter Function, RotEquivConv2d, ScaleEquivConv2d, MyModel definition...


# --- Custom Rotate Filter Function ---
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


# --- RotEquivConv2d Definition ---
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

# --- ScaleEquivConv2d Definition ---
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
        
        
        
# --- MyModel Definition ---


class MyModel(nn.Module):
    def __init__(self, num_classes=100, device='cuda'):
        super(MyModel, self).__init__()
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






def train(model, trainloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    return running_loss / len(trainloader)

def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total
    
    
    
    
    
    

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    
    torch.cuda.manual_seed_all(66)
    
    

    # Data Preparation with Random Rotation
    


    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])



    trainset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

    testset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_train)
    testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)

    # Model, Loss Function, Optimizer
    model_custom = MyModel(num_classes=100, device=device).to(device)
   

    criterion = nn.CrossEntropyLoss()
    optimizer_custom = torch.optim.Adam(model_custom.parameters(), lr=0.001)
    optimizer_standard = torch.optim.Adam(model_standard.parameters(), lr=0.001)

    # Training and Testing
    epochs = 200
    train_losses_custom, test_accuracies_custom = [], []
    train_losses_standard, test_accuracies_standard = [], []

    with open("original_data.txt", "w") as file:
        file.write("Epoch,Custom Model Loss,Custom Model Accuracy,Standard Model Loss,Standard Model Accuracy\n")

        for epoch in range(epochs):
            loss_custom = train(model_custom, trainloader, criterion, optimizer_custom, device)
            accuracy_custom = test(model_custom, testloader, device)
            train_losses_custom.append(loss_custom)
            test_accuracies_custom.append(accuracy_custom)

           

            # Save the data to file
            file.write(f"{epoch+1},{loss_custom},{accuracy_custom},{loss_standard},{accuracy_standard}\n")

            print(f"Epoch {epoch+1}/{epochs} - Custom Model Loss: {loss_custom:.4f}, Accuracy: {accuracy_custom:.2f}% ")

   



if __name__ == "__main__":
    main()



