from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import torch
from torchvision import models, transforms
from PIL import Image
import os

# -------------------------------
# Configuration
# -------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# -------------------------------
# Class Names
# -------------------------------
class_names_map = {
    'Liver Disease': ['Jaundiced Liver', 'Normal Liver'],
    'Skin Disease': ['Acne', 'Chickenpox', 'Measles', 'Monkeypox'],
}

# -------------------------------
# Model Cache
# -------------------------------
loaded_models = {}

def load_model_for_disease(disease):
    if disease in loaded_models:
        return loaded_models[disease]

    model_path = ''
    class_names = []

    if disease == 'Liver Disease':
        model_path = os.path.join('models', 'jAUNDICE_resnet34(1).pth')
        class_names = class_names_map[disease]
    elif disease == 'Skin Disease':
        model_path = os.path.join('models', 'skinm_resnet34.pth')
        class_names = class_names_map[disease]
    else:
        raise ValueError(f"Unknown disease: {disease}")

    checkpoint = torch.load(model_path, map_location=device)
    model = models.resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    loaded_models[disease] = (model, class_names)
    return model, class_names

# -------------------------------
# Preprocessing Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -------------------------------
# Predict Endpoint
# -------------------------------
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files or 'uid' not in request.form or 'disease' not in request.form:
        return jsonify({'error': 'Missing image, UID, or disease'}), 400

    image_file = request.files['image']
    uid = request.form['uid']
    disease = request.form['disease']

    try:
        model, class_names = load_model_for_disease(disease)

        image = Image.open(image_file.stream).convert("RGB")
        input_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(input_tensor)
            _, predicted = torch.max(output, 1)
            predicted_class = class_names[predicted.item()]

        # Save image
        save_dir = os.path.join(app.config['UPLOAD_FOLDER'], disease, uid)
        os.makedirs(save_dir, exist_ok=True)
        image_path = os.path.join(save_dir, image_file.filename)
        image.save(image_path)

        with open(os.path.join(save_dir, "result.txt"), 'w') as f:
            f.write(predicted_class)

        return jsonify({
            'uid': uid,
            'filename': image_file.filename,
            'prediction': predicted_class,
            'image_url': request.host_url.rstrip('/') + f"/uploads/{disease}/{uid}/{image_file.filename}"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# -------------------------------
# Image Serving Endpoint
# -------------------------------
@app.route('/uploads/<disease>/<uid>/<filename>')
def uploaded_file(disease, uid, filename):
    dir_path = os.path.join(app.config['UPLOAD_FOLDER'], disease, uid)
    file_path = os.path.join(dir_path, filename)

    if not os.path.isfile(file_path):
        return jsonify({'error': 'File not found'}), 404

    return send_from_directory(directory=dir_path, path=filename)

# -------------------------------
# Run Server
# -------------------------------
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 3000))
    app.run(host='0.0.0.0', port=port)
