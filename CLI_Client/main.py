import os
import click
import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import transforms
from datetime import datetime

classes = ['Abstract_Expressionism', 'Analytical_Cubism', 'Baroque', 'Cubism', 'Early_Renaissance', 'Expressionism',
           'Fauvism', 'High_Renaissance', 'Impressionism', 'Mannerism_Late_Renaissance', 'Minimalism', 'Modern',
           'Northern_Renaissance', 'Pointillism', 'Pop_Art', 'Post_Impressionism', 'Primitivism', 'Realism', 'Rococo',
           'Romanticism', 'Symbolism', 'Ukiyo_e']


def load_model(model_path, device):
    """
    Loads the trained model.
    """
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model


def predict_image(image_path, model, device):
    """
    Predicts the class of the given image.
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
    return classes[predicted.item()], image


def visualize_prediction(image, true_label, predicted_label, model_name):
    """
    Displays the image along with its true and predicted labels, and adds the model name as the title.
    """
    output_dir = "reports"
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Model: {model_name}\nTrue Label: {true_label}\nPredicted Label: {predicted_label}", fontsize=12)

    current_dateTime = datetime.now()

    cm_plot_path = os.path.join(output_dir, f"{str(current_dateTime).replace('.', ' ').replace(':', ' ')}.png")
    plt.savefig(cm_plot_path)
    plt.show()
    plt.close()


@click.command()
@click.option('--model-name', required=True, help='Name of model')
@click.option('--style-name', required=True, help='True style of the painting')
@click.option('--image-path', required=True, help='Path to the image to classify.')
def cli(model_name, style_name, image_path):
    """
    CLI for classification styles.
    """

    if model_name == 'resnet18':
        model_path = r'D:\lizuwka\dz 4\ML_styles\Model1\full_model.pth'
    else:
        model_path = r'D:\lizuwka\dz 4\ML_styles\Model2\full_model.pth'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_model(model_path, device)

    result, image = predict_image(image_path, model, device)
    click.echo(f"The predicted style is: {result}")

    visualize_prediction(image, style_name, result, model_name)


if __name__ == '__main__':
    cli()

# python main.py --model-name 'resnet18' --style-name 'Expressionism' --image-path 'D:\data\train\Expressionism\alexander-calder_elephant-1928.jpg'
# python main.py --model-name 'resnet50' --style-name 'Expressionism' --image-path 'D:\data\train\Expressionism\alexander-calder_elephant-1928.jpg'
