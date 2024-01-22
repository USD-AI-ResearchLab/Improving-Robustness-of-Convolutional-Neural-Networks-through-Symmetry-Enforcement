import torch

# Function to train the model on the training dataset
# Input: model - neural network model, trainloader - training data loader,
#                criterion - loss function, optimizer - optimization algorithm, device - device for computation
# Output: Average training loss
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


# Function to test the model on the test dataset
# Input: model - neural network model, testloader - test data loader, device - device for computation
# Output: Model accuracy on the test dataset
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
    