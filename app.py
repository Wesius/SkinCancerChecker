import io
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from model import SkinLesionModel

app = Flask(__name__)

# Load the model
model = SkinLesionModel()
model.load_state_dict(torch.load('skin_lesion_model.pth', map_location=torch.device('cpu')))
model.eval()

# Define the transformation for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Define the class labels and their descriptions
label_info = {
    'MEL': {'name': 'Melanoma', 'description': 'A serious form of skin cancer that develops in the cells (melanocytes) that produce melanin.'},
    'NV': {'name': 'Melanocytic Nevus', 'description': 'A common, usually non-cancerous growth on the skin that develops when pigment cells (melanocytes) grow in clusters or clumps.'},
    'BCC': {'name': 'Basal Cell Carcinoma', 'description': 'A type of skin cancer that begins in the basal cells, usually caused by long-term exposure to ultraviolet (UV) radiation.'},
    'AKIEC': {'name': 'Actinic Keratosis / Intraepithelial Carcinoma', 'description': 'A precancerous skin growth that can develop into squamous cell carcinoma if left untreated.'},
    'BKL': {'name': 'Benign Keratosis', 'description': 'A non-cancerous skin growth that develops from keratinocytes, often appearing as a waxy, scaly, or rough growth on the skin.'},
    'DF': {'name': 'Dermatofibroma', 'description': 'A common, benign skin growth that often appears as a small, firm bump on the skin, usually on the legs.'},
    'VASC': {'name': 'Vascular Lesion', 'description': 'A general term for visible abnormalities of blood vessels on the skin, including hemangiomas and port-wine stains.'}
}

def analyze_image(image):
    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

        # Convert probabilities to percentages
        percentages = probabilities.mul(100).tolist()

        # Create a dictionary of class names and their probabilities
        predictions = {label: f"{prob:.2f}%" for label, prob in zip(label_info.keys(), percentages)}

        # Get the class with the highest probability
        max_prob_index = probabilities.argmax().item()
        predicted_class = list(label_info.keys())[max_prob_index]

    return {
        'predicted_class': predicted_class,
        'predicted_name': label_info[predicted_class]['name'],
        'description': label_info[predicted_class]['description'],
        'probabilities': predictions,
        'confidence': max(predictions.values())
    }

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    image_bytes = file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)

    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]

        # Convert probabilities to percentages
        percentages = probabilities.mul(100).tolist()

        # Create a dictionary of class names and their probabilities
        predictions = {label: f"{prob:.2f}%" for label, prob in zip(label_info.keys(), percentages)}

        # Get the class with the highest probability
        max_prob_index = probabilities.argmax().item()
        predicted_class = list(label_info.keys())[max_prob_index]

    return jsonify({
        'predicted_class': predicted_class,
        'predicted_name': label_info[predicted_class]['name'],
        'description': label_info[predicted_class]['description'],
        'probabilities': predictions,
        'confidence': max(predictions.values())
    })


@app.route('/', methods=['GET', 'POST'])
def upload_and_predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            # Read the image file
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))

            # Convert to RGB if it's not already
            if image.mode != 'RGB':
                image = image.convert('RGB')

            # Analyze the image
            result = analyze_image(image)

            # Return the results
            return jsonify(result)

    return '''
    <!doctype html>
    <title>Upload an image for skin lesion analysis</title>
    <h1>Upload an image for skin lesion analysis</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Analyze>
    </form>
    '''

if __name__ == '__main__':
    app.run(debug=True)