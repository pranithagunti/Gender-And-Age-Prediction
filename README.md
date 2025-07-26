# Gender-And-Age-Prediction
A Complete Convolutional Neural Network(CNN) Based Case Study.

A Convolutional Neural Network (CNN)-based deep learning project for predicting **gender** and **age** from facial images using the **UTKFace dataset**.

---

### 📌 Project Overview

This project uses a CNN model to classify the **gender** (male or female) and estimate the **age** of a person from an image. The model is trained and evaluated on the **UTKFace dataset**, which contains over 20,000 labeled face images.

---

### 🎯 Objectives

* Predict **Gender**: Binary classification (Male/Female).
* Predict **Age**: Multi-class regression or classification problem.
* Use **CNN** for feature extraction and model training.

---

### 🧠 Model Architecture

* Input: 200x200 RGB face images
* Convolutional layers with ReLU activations
* MaxPooling layers
* Flattening layer
* Fully Connected (Dense) layers
* Two output heads:

  * One for **gender** (binary classification)
  * One for **age** (regression or multi-class classification)

---

### 🗂️ Dataset: UTKFace

* Over 20,000 face images
* Filename format: `age_gender_ethnicity_date&time.jpg`
* Used labels:

  * **Gender**: 0 = Male, 1 = Female
  * **Age**: Integer values from 0 to 116

---

### 🛠️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy, Pandas, Matplotlib
* Google Colab

---

### 🚀 How to Run

1. Open the Colab notebook below.
2. Mount Google Drive (if needed) and upload UTKFace dataset.
3. Run all cells to preprocess data, build the model, train, and evaluate.

---

### 📎 Colab Notebook

👉 [Open in Google Colab](https://colab.research.google.com/drive/1fc9TNUDV8_9tePr7CK-u_hSS_xzUqBQT?usp=sharing)
*(Replace the link with your actual Colab notebook URL)*

---

### 📈 Results

* Accuracy for Gender Classification: \~95% (depending on preprocessing and model tuning)
* Age Prediction Error: Varies (e.g., MAE around 4–6 years)

---

### 📌 Use Cases

* Surveillance and crowd analysis
* Age-based content recommendation
* Personalized marketing
* Security systems

---

### ✅ Conclusion

This case study shows how CNNs can effectively be used for **multi-task learning** — predicting both gender and age from images using a single neural network.

---

Let me know if you want me to generate the CNN model code or provide a working Colab notebook for it!
