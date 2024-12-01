from flask import Flask, request, jsonify, render_template
import torch
from main import load_model, predict_image
import os


def find_style(name):
    path = r'D:\data\test'

    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            style = os.path.join(root, name).split('\\')[3]
            result.append(style)
    return result


app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18 = load_model(r'D:\lizuwka\dz 4\ML_styles\Model1\full_model.pth', device)
resnet50 = load_model(r'D:\lizuwka\dz 4\ML_styles\Model2\full_model.pth', device)

# Path to save uploaded images temporarily
TEMPLATES_FOLDER = 'templates'
os.makedirs(TEMPLATES_FOLDER, exist_ok=True)
app.config['TEMPLATES_FOLDER'] = TEMPLATES_FOLDER


@app.route('/predict', methods=['POST'])
def predict_route():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    actual_class = find_style(file.filename)

    static_folder = 'static/images'
    os.makedirs(static_folder, exist_ok=True)  # Ensure the folder exists
    file_path = os.path.join(static_folder, 'temp_image.jpg')
    file.save(file_path)

    predicted_class_18, _ = predict_image(file_path, resnet18, device)
    predicted_class_50, _ = predict_image(file_path, resnet50, device)

    return render_template('res.html',
                           predicted_class_18=predicted_class_18,
                           predicted_class_50=predicted_class_50,
                           actual_class=actual_class,
                           image_url='/static/images/temp_image.jpg')

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
