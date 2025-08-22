import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("face_mask_cnn_model.h5")

# Define class folder path
dataset_path = "face_mask_detection"
classes = os.listdir(dataset_path)  # ['with_mask', 'without_mask', 'improper_mask']

# Image settings
img_size = (150, 150)
num_images_per_class = 5  # Change this to select more/less images

# Get class names as per training order
class_names = sorted(classes)

# Loop through each class and pick random images
for class_name in class_names:
    class_path = os.path.join(dataset_path, class_name)
    image_files = os.listdir(class_path)
    selected_images = random.sample(image_files, min(num_images_per_class, len(image_files)))

    for img_name in selected_images:
        img_path = os.path.join(class_path, img_name)
        
        try:
            img = image.load_img(img_path, target_size=img_size)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

            prediction = model.predict(img_array)
            predicted_label = class_names[np.argmax(prediction)]

            # Display
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Actual: {class_name} | Predicted: {predicted_label}")
            plt.show()
        
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")