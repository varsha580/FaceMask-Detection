# Face Mask Detection Using Deep Learning (CNN)

## 📝 Project Overview
This project implements a **Face Mask Detection system** using **Convolutional Neural Networks (CNN)**. The system can classify whether a person is **wearing a mask** or **not wearing a mask** from images or video streams in real-time. It is useful for monitoring public spaces and ensuring safety during pandemics.

---

## 🧰 Technologies & Tools Used
- **Programming Language:** Python 3.x  
- **Libraries:** TensorFlow, Keras, OpenCV, NumPy, Matplotlib, scikit-learn  
- **IDE:** VS Code / Jupyter Notebook  
- **Framework:** CNN (Convolutional Neural Network)  

---

## 📂 Dataset
- The dataset contains images of people **with masks** and **without masks**.  
- Preprocessing includes:  
  - Resizing images to 128x128 pixels  
  - Normalization  
  - Train-test split  

**Example sources:**  
- [Kaggle Face Mask Dataset](https://www.kaggle.com/datasets)  

---

## 🏗 Methodology
1. **Data Preprocessing**  
   - Load dataset and resize images  
   - Normalize pixel values (0-1)  
   - Split into training, validation, and test sets  

2. **CNN Model Architecture**  
   - Input Layer  
   - 2-3 Convolutional + MaxPooling Layers  
   - Flatten Layer  
   - Dense Layers with ReLU activation  
   - Output Layer with Softmax activation (2 classes: Mask / No Mask)  

3. **Model Training**  
   - Optimizer: Adam  
   - Loss Function: Categorical Crossentropy  
   - Metrics: Accuracy  
   - Epochs: 20–50 (adjustable)  

4. **Evaluation**  
   - Accuracy, Precision, Recall, F1-Score  
   - Confusion Matrix visualization  

5. **Real-Time Detection**  
   - Use OpenCV to capture video from webcam  
   - Detect faces using Haar Cascade or DNN face detector  
   - Predict mask status for each detected face  

---

## 📈 Results
- Achieved **~95% accuracy** on test dataset (example, replace with your result)  
- Real-time detection works efficiently on webcam feed  

---

## ⚙ Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/face-mask-detection.git
   cd face-mask-detection
