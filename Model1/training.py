import torch.optim as optim
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from model import *
import os
from testing import evaluate_model, save_log

os.makedirs("saved_models", exist_ok=True)

# Optimizer
criterion = CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)


def train_model(my_model=model, start_epoch=0, num_epochs=10):
    my_model.train()
    save_log("Training Started")

    for epoch in range(start_epoch, (num_epochs+1)):
        running_loss = 0.0
        correct = 0
        total = 0

        # for each batch
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward and backward
            outputs = my_model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        accuracy = 100 * correct / total
        log_message = f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader):.4f}, Accuracy: {accuracy:.2f}%"
        print(log_message)
        save_log(log_message)

        # Save model after each epoch
        model_path = f"saved_models/model_epoch_{epoch + 1}.pth"
        torch.save(my_model.state_dict(), model_path)
        print(f"Model saved: {model_path}")


def get_last_epoch(epoch):
    last_saved_model_path = f"saved_models/model_epoch_{epoch}.pth"
    model.load_state_dict(torch.load(last_saved_model_path))
    model.to(device)
    return model


last_epoch_run = 10
get_last_epoch(last_epoch_run)
train_model(my_model=model, start_epoch=last_epoch_run, num_epochs=10)
conf_matrix = evaluate_model(my_model=get_last_epoch(last_epoch_run), output_dir="reports")
