from flask import Flask, request, jsonify, render_template
import torch
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the EfficientNet model architecture
def create_efficientnet_model(num_classes):
    model = models.efficientnet_b0(weights='DEFAULT')  # Load the model
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)  # Update the classifier layer
    return model

# Load the model
num_classes = 7  # Update based on your number of classes
model = create_efficientnet_model(num_classes)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))  # Load the state dict
model.eval()  # Set to evaluation mode

# Define class labels
classes = ['MEL', 'NV', 'BCC', 'AKIEC', 'BKL', 'DF', 'VASC']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])  # Ensure this matches your React code
def classify():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(file).convert('RGB')
    except Exception as e:
        print(f"Error opening image: {e}")
        return jsonify({'error': 'Invalid image file'}), 400

    # Apply transformations
    image = transform(image)
    image = image.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction = classes[predicted.item()]
        confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted.item()].item()

    return jsonify({'prediction': prediction, 'confidence': confidence})

if __name__ == '__main__':
    app.run(debug=True)
