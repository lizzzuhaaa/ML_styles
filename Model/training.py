import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from model import *
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

# Optimizer
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0

        # for each batch
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward and backward
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")


# train
train_model(model, train_loader, criterion, optimizer, device, num_epochs=10)


def evaluate_model(model, test_loader, device):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Predictions
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            # True values and Predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Metrics
    print("Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=classes))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)
    return cm


conf_matrix = evaluate_model(model, test_loader, device)
