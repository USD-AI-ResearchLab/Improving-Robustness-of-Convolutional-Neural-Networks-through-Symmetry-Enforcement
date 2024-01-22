import torch 
import torch.nn as nn 
import torch.nn.functional as F 
import torchvision 
import torchvision.transforms as transforms 
import numpy as np 
import matplotlib.pyplot as plt 
from CustomModel import CustomModel
from TrainTest import train, test

# Main function to set up and run the training process
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
    model_custom = CustomModel(num_classes=100, device=device).to(device) 
    criterion = nn.CrossEntropyLoss() 
    optimizer_custom = torch.optim.Adam(model_custom.parameters(), lr=0.001) 

    # Training and Testing 
    epochs = 200
    train_losses_custom, test_accuracies_custom = [], [] 

    with open("original_data.txt", "w") as file: 
        file.write("Epoch,Custom Model Loss,Custom Model Accuracy\n") 

        for epoch in range(epochs): 
            loss_custom = train(model_custom, trainloader, criterion, optimizer_custom, device) 
            accuracy_custom = test(model_custom, testloader, device) 
            train_losses_custom.append(loss_custom) 
            test_accuracies_custom.append(accuracy_custom) 

            # Save the data to file 
            file.write(f"{epoch+1},{loss_custom},{accuracy_custom}\n") 
            print(f"Epoch {epoch+1}/{epochs} - Custom Model Loss: {loss_custom:.4f}, Accuracy: {accuracy_custom:.2f}% ") 

    # Plotting 
    plt.figure(figsize=(12, 5)) 
    plt.subplot(1, 2, 1) 
    plt.plot(train_losses_custom, label='Custom Model Loss') 
    plt.title('Training Loss') 
    plt.legend() 

    plt.subplot(1, 2, 2) 
    plt.plot(test_accuracies_custom, label='Custom Model Accuracy') 
    plt.title('Test Accuracy') 
    plt.legend() 
    plt.savefig('CustomModelPlot.png')
    plt.show() 


if __name__ == "__main__": 
    main() 