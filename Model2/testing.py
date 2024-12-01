from model import *
from sklearn.metrics import classification_report, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns
results_file = "training_logs.txt"


def save_log(message, file=results_file):
    with open(file, "a") as f:
        f.write(message + "\n")


def evaluate_model(my_model=model, output_dir="reports"):
    my_model.eval()
    all_labels = []
    all_predictions = []

    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Predictions
            outputs = my_model(inputs)
            _, predicted = torch.max(outputs, 1)

            # True values and Predictions
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Metrics
    print("Classification Report:")
    save_log("Classification Report:")
    print(classification_report(all_labels, all_predictions, target_names=classes))
    save_log(classification_report(all_labels, all_predictions, target_names=classes))

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")

    # Save the confusion matrix plot
    cm_plot_path = os.path.join(output_dir, "confusion_matrix.png")
    plt.savefig(cm_plot_path)
    plt.close()
    return cm
