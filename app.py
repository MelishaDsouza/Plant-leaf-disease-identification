import os
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from tensorflow.keras.models import load_model
import numpy as np
import cv2

app = Flask(__name__)

# Load the trained model
model = load_model('plant_disease_model.h5')

# Categories
data_dir = '/Users/melisha/Desktop/6th sem project/dataset'
categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

# Upload folder
UPLOAD_FOLDER = '/Users/melisha/Desktop/6th sem project/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Disease information dictionary
disease_info = {
    "Pepper__bell___Bacterial_spot": {
        "description": "Bacterial spot is a disease that affects peppers causing small, water-soaked, circular spots on leaves.",
        "solution": "Use copper-based bactericides and ensure proper sanitation of tools and equipment."
    },
    "Pepper__bell___healthy": {
        "description": "Your pepper plant is healthy. No diseases detected.",
        "solution": "Maintain regular care with balanced fertilization and proper watering."
    },
    "Potato___Early_blight": {
        "description": "Early blight causes dark brown to black spots on leaves and can reduce yield.",
        "solution": "Apply fungicides containing chlorothalonil or mancozeb and practice crop rotation."
    },
    "Potato___healthy": {
        "description": "Your potato plant is healthy. No diseases detected.",
        "solution": "Continue regular care with appropriate fertilization and watering."
    },
    "Potato___Late_blight": {
        "description": "Late blight is a serious disease causing large, dark brown spots on leaves and stems.",
        "solution": "Use fungicides like metalaxyl and maintain good air circulation around plants."
    },
    "Tomato___Target_Spot": {
        "description": "Target spot causes circular spots with concentric rings on leaves and fruits.",
        "solution": "Use fungicides such as chlorothalonil and practice crop rotation."
    },
    "Tomato__Tomato_mosaic_virus": {
        "description": "Tomato mosaic virus causes mottling and distortion of leaves, reducing yield.",
        "solution": "Remove and destroy infected plants, and disinfect tools."
    },
    "Tomato_Tomato_YellowLeaf__Curl_Virus": {
        "description": "This virus causes yellowing and curling of leaves, stunting plant growth.",
        "solution": "Control whitefly populations and use virus-resistant tomato varieties."
    },
    "Tomato_Bacterial_spot": {
        "description": "Bacterial spot causes small, dark, water-soaked spots on leaves and fruits.",
        "solution": "Use copper-based bactericides and ensure proper sanitation."
    },
    "Tomato_Early_blight": {
        "description": "Early blight causes dark brown spots with concentric rings on leaves and stems.",
        "solution": "Apply fungicides containing chlorothalonil or mancozeb and rotate crops."
    },
    "Tomato_healthy": {
        "description": "Your tomato plant is healthy. No diseases detected.",
        "solution": "Continue regular care with balanced fertilization and proper watering."
    },
    "Tomato_Late_blight": {
        "description": "Late blight causes large, dark brown spots on leaves and stems.",
        "solution": "Use fungicides like metalaxyl and ensure good air circulation around plants."
    },
    "Tomato_Leaf_Mold": {
        "description": "Leaf mold causes yellow spots on leaves, which turn brown as they age.",
        "solution": "Improve air circulation and use fungicides such as copper-based sprays."
    },
    "Tomato_Septoria_leaf_spot": {
        "description": "Septoria leaf spot causes small, dark spots with gray centers on leaves.",
        "solution": "Use fungicides containing chlorothalonil or mancozeb and practice crop rotation."
    },
    "Tomato_Spider_mites_Two_spotted_spider_mite": {
        "description": "Spider mites cause stippling and yellowing of leaves, leading to defoliation.",
        "solution": "Use miticides like abamectin and maintain adequate moisture levels."
    }
}

def preprocess_image(image_path, img_size=128):
   img_array = cv2.imread(image_path)
   if img_array is None:
       raise ValueError(f"Image at path {image_path} could not be loaded.")
   resized_array = cv2.resize(img_array, (img_size, img_size))
   normalized_array = resized_array / 255.0
   return np.expand_dims(normalized_array, axis=0)


def predict_disease(image_path, model, categories):
   preprocessed_image = preprocess_image(image_path)
   prediction = model.predict(preprocessed_image)
   predicted_class = np.argmax(prediction, axis=1)[0]
   disease_name = categories[predicted_class]
   confidence = np.max(prediction)
   return disease_name, confidence


@app.route('/')
def index():
   return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        # Predict the disease
        disease_name, confidence = predict_disease(filepath, model, categories)
        # URL for the uploaded image
        image_url = url_for('uploaded_file', filename=file.filename)
        # Get disease information
        info = disease_info.get(disease_name, {"description": "No information available.", "solution": "No solution available."})
        return render_template('result.html', prediction=disease_name, confidence=confidence * 100, image_url=image_url, info=info)
    return redirect(url_for('index'))



@app.route('/uploads/<filename>')
def uploaded_file(filename):
   return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


if __name__ == "__main__":
   # Ensure the upload folder exists
   if not os.path.exists(UPLOAD_FOLDER):
       os.makedirs(UPLOAD_FOLDER)
  
   app.run(debug=True)
