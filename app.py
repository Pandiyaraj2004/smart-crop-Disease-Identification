from flask import Flask, render_template, request
import os
import cv2
import numpy as np
import tensorflow as tf
from werkzeug.utils import secure_filename
import uuid

# ----------------------
# Flask Configuration
# ----------------------
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = os.path.join('static', 'uploads')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ----------------------
# Plant Configuration
# ----------------------
plants = {
    'rice': {
        'model_path': 'models/rice_model.h5',
        'classes': [
            'Bacterial Leaf Blast',
            'Brown Spot',
            'Healthy',
            'Leaf Blast',
            'Leaf Scald',
            'Narrow Brown Spot'
        ]
    },

    'corn': {
        'model_path': 'models/corn_model.h5',
        'classes': ['Common Rust', 'Gray Leaf Spot', 'Blight', 'Healthy']
    },

    'tomato': {
        'model_path': 'models/Tomato_model.h5',
        'classes': [
            'Early Blight',
            'Late Blight',
            'Leaf Mold',
            'Septoria Leaf Spot',
            'Target Spot',
            'Tomato Mosaic Virus',
            'Healthy'
        ]
    },

    'potato': {
        'model_path': 'models/potato_model.h5',
        'classes': ['Early Blight', 'Late Blight', 'Healthy']
    },

    'wheat': {
        'model_path': 'models/wheatDiseaseModel.h5',
        'classes': ['Leaf Rust', 'Stripe Rust', 'Healthy']
    }
}


# ----------------------
# Load Models Safely + Detect Input Size
# ----------------------
for plant in plants:
    model = tf.keras.models.load_model(plants[plant]['model_path'], compile=False)
    plants[plant]['loaded_model'] = model

    input_shape = model.input_shape

    # If model has multiple inputs
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    height = input_shape[1]
    width = input_shape[2]

    # If dynamic input shape (None)
    if height is None or width is None:
        print(f"[WARNING] {plant} model has dynamic input shape. Using default 224x224.")
        height = 224
        width = 224

    plants[plant]['img_height'] = int(height)
    plants[plant]['img_width'] = int(width)

    print(f"{plant.upper()} Model Loaded | Input Size: {height}x{width}")


# ----------------------
# Disease Solutions (Single 20-word sentence)
# ----------------------
disease_solutions = {
    'Bacterial Leaf Blast': "Apply recommended bactericides early, use resistant rice varieties, and maintain balanced nitrogen fertilization to reduce infection spread.",
    'Brown Spot': "Use appropriate fungicides during early stages, maintain proper irrigation practices, and remove infected plant debris to prevent further damage.",
    'Leaf Blast': "Spray suitable fungicides immediately after detection and ensure proper plant spacing to reduce humidity and fungal growth.",
    'Leaf Scald': "Use certified seeds, remove infected plants promptly, and improve field drainage systems to control disease spread.",
    'Narrow Brown Spot': "Maintain balanced nutrients, use quality seeds, and monitor crops regularly to detect and control disease early.",
    'Common Rust': "Apply fungicides at early stages and use resistant corn hybrids along with crop rotation practices.",
    'Gray Leaf Spot': "Improve air circulation, remove infected leaves, and apply recommended fungicide sprays to reduce severity.",
    'Blight': "Use certified seeds, remove affected plants quickly, and apply protective fungicides to prevent progression.",
    'Early Blight': "Apply suitable fungicides, remove infected leaves, and practice annual crop rotation to prevent recurring infections.",
    'Late Blight': "Use resistant varieties and apply protective fungicide sprays especially during humid weather conditions.",
    'Leaf Mold': "Improve ventilation, reduce humidity levels, and remove infected foliage to stop disease development.",
    'Septoria Leaf Spot': "Adopt drip irrigation methods and apply fungicides promptly after removing infected leaves.",
    'Target Spot': "Maintain adequate plant spacing and apply fungicides when early symptoms are visible.",
    'Tomato Mosaic Virus': "Use certified virus-free seeds and remove infected plants immediately to prevent transmission.",
    'Leaf Rust': "Use resistant wheat varieties and apply fungicides as soon as symptoms appear in the field.",
    'Stripe Rust': "Spray fungicides early and maintain proper crop spacing to reduce infection risk.",
    'Healthy': "The crop appears healthy; continue proper irrigation, balanced fertilization, and regular monitoring practices."
}


# ----------------------
# Helper Functions
# ----------------------
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess_image(filepath, height, width):
    img = cv2.imread(filepath)

    if img is None:
        raise ValueError("Image not read properly.")

    if height <= 0 or width <= 0:
        raise ValueError("Invalid model input size.")

    img = cv2.resize(img, (width, height))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# ----------------------
# Routes
# ----------------------

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/home')
def home():
    return render_template('home.html', plants=plants.keys())


@app.route('/plant/<plant_name>')
def plant_page(plant_name):
    if plant_name not in plants:
        return "Plant not found", 404
    return render_template('plant.html', plant=plant_name)


@app.route('/predict/<plant_name>', methods=['POST'])
def predict(plant_name):

    if plant_name not in plants:
        return "Plant not found", 404

    file = request.files.get('file')

    if not file or file.filename == '':
        return render_template('plant.html', plant=plant_name, error="Please upload an image.")

    if not allowed_file(file.filename):
        return render_template('plant.html', plant=plant_name, error="Invalid file type. Upload PNG/JPG/JPEG.")

    try:
        # Unique file name
        unique_name = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(filepath)

        model = plants[plant_name]['loaded_model']
        height = plants[plant_name]['img_height']
        width = plants[plant_name]['img_width']

        img = preprocess_image(filepath, height, width)

        prediction = model.predict(img)
        class_index = np.argmax(prediction)

        disease = plants[plant_name]['classes'][class_index]
        confidence = round(float(np.max(prediction)) * 100, 2)

        solution = disease_solutions.get(disease, "Follow recommended agricultural practices and consult experts if necessary.")

        return render_template(
            'plant.html',
            plant=plant_name,
            result=disease,
            confidence=confidence,
            solution=solution,
            img_path=f'uploads/{unique_name}'
        )

    except Exception as e:
        return render_template(
            'plant.html',
            plant=plant_name,
            error=f"Prediction Error: {str(e)}"
        )


# ----------------------
# Run App
# ----------------------
if __name__ == '__main__':
    app.run(debug=True)
