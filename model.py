import os
import numpy as np
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the trained model
model = load_model("facemask_model.h5")

# Load class label mapping
with open("label_map.json") as f:
    label_map = json.load(f)
reverse_map = {v: k for k, v in label_map.items()}

# Preprocess function for image
def preprocess_image(img_path, target_size=(128, 128)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Predict on all images from folders
base_path = "dataset"  # Path where 3 folders are stored
for folder_name in os.listdir(base_path):
    folder_path = os.path.join(base_path, folder_name)
    if os.path.isdir(folder_path):
        print(f"\nChecking folder: {folder_name}")
        for file_name in os.listdir(folder_path):
            if file_name.endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(folder_path, file_name)
                processed = preprocess_image(img_path)
                prediction = model.predict(processed)
                predicted_class_index = np.argmax(prediction)
                predicted_class = reverse_map[predicted_class_index]

                print(f"File: {file_name} → Predicted: {predicted_class}")
