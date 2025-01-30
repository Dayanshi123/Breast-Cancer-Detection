---

# Breast Cancer Classification with ResNet50 🧑‍⚕️💻

Welcome to the **Breast Cancer Classification** project! This deep learning model classifies breast cancer histopathological images as **benign** or **malignant** using the **ResNet50** architecture. In addition, we integrate **LIME (Local Interpretable Model-agnostic Explanations)** to make the model's predictions more transparent and interpretable for clinicians.

By combining cutting-edge deep learning with explainability, we aim to empower healthcare professionals to make more informed decisions with greater confidence. Let’s dive in! 🌟

## 🚀 Project Overview

Breast cancer is a leading cause of death globally. Early detection plays a crucial role in increasing survival rates. This project utilizes deep learning to automate the classification of breast cancer images, helping pathologists to better diagnose and treat patients. The dataset used is **BreakHis**, a collection of histopathological images, and the classification model is based on **ResNet50**, one of the most powerful architectures in the field.

### Features:
- **Classifies images**: Benign vs. Malignant
- **Incorporates LIME**: Visualizes and explains model predictions to improve interpretability
- **High accuracy**: Achieves 90%+ accuracy in classification

---

## 📦 Technologies Used

- **Python**: Main programming language for model building and data handling
- **Keras + TensorFlow**: Deep learning framework used for model creation and training
- **Matplotlib**: For visualization of training curves and results
- **OpenCV**: For preprocessing images
- **LIME**: For model interpretability
- **Scikit-learn**: For evaluation metrics (Accuracy, Precision, Recall, F1-Score)

---

## 🧠 Model Architecture

We use **ResNet50**, a powerful convolutional neural network (CNN) designed to avoid vanishing gradients using residual connections. Here's a breakdown of the architecture:

1. **Input Layer**: Accepts resized images from the BreakHis dataset.
2. **ResNet50 Backbone**: Uses a pre-trained ResNet50 model to extract features from the images.
3. **Fully Connected Layer**: Dense layers that process the extracted features.
4. **Output Layer**: A sigmoid activation function that outputs the classification result: **Benign** or **Malignant**.

---

## 📂 Dataset

### BreakHis Dataset
- **Link**: [BreakHis Dataset](https://drive.google.com/drive/folders/1eEKbE_wrpKkeqVXDzm6vHTFv8SnEqOuu?usp=drive_link)
- **Classes**:
  - **Benign**: Non-cancerous tissue
  - **Malignant**: Cancerous tissue
- **Magnifications**: Images come from different magnifications to cover a broad range of features.

---

## ⚙️ Setup Instructions

Let's get your environment ready! Follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/Dayanshi123/Breast-Cancer-Detection.git
    cd Breast-Cancer-Detection
    ```

2. **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

3. **Download the BreakHis dataset** from [here](https://www.kaggle.com/datasets) and place it in the `data/` folder or else you can **download the BreakHis dataset** from [here](https://drive.google.com/drive/folders/1eEKbE_wrpKkeqVXDzm6vHTFv8SnEqOuu?usp=drive_link)) and place it in the `data/` folder.

4. **Make sure you have access to a GPU** for faster training (optional but recommended).

---

## 🏋️ Training the Model

Training your model is simple! Run the script below to start the training process.

```bash
train_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/drive/MyDrive/Dataset/1 1 BreakHis_Data-200x/train',
    labels='inferred',
    label_mode = 'int',
    batch_size=32,
    image_size=(150,150)
)

validation_ds = keras.utils.image_dataset_from_directory(
    directory = '/content/drive/MyDrive/Dataset/1 1 BreakHis_Data-200x/val',
    labels='inferred',
    label_mode = 'int',
    batch_size=32,
    image_size=(150,150)
)

train_ds = train_ds.map(lambda x, y: (preprocess_input(x), y))
validation_ds = validation_ds.map(lambda x, y: (preprocess_input(x), y))

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

history = model.fit(train_ds, epochs=150, validation_data=validation_ds, callbacks=[early_stopping, reduce_lr])
```

- **Training Configuration**: The model will train on the BreakHis dataset, utilizing data augmentation and dropout for regularization.
- **Output**: The best model weights will be saved in `trainedModel.h5`.

---

## 🏆 Evaluation

After training, you can evaluate the model's performance with:

```bash
# Evaluate the model on the validation set
val_labels = []
val_preds = []

for images, labels in validation_ds:
    preds = model.predict(images)
    val_labels.extend(labels.numpy())
    val_preds.extend(np.argmax(preds, axis=1))

# Generate classification report
report = classification_report(val_labels, val_preds, target_names=cm_labels)
print(report)
TP = np.diag(cm)
FP = np.sum(cm, axis=0) - TP
FN = np.sum(cm, axis=1) - TP
TN = np.sum(cm) - (FP + FN + TP)

import pandas as pd
metrics_df = pd.DataFrame({
    'Class': cm_labels,
    'TP': TP,
    'TN': TN,
    'FP': FP,
    'FN': FN
})
```

This will display:
- **Accuracy**, **Precision**, **Recall**, **F1-Score**
- A **Confusion Matrix** to understand model performance
- **LIME** explanation to understand what the model is focusing on during predictions

---

## 🔍 LIME (Local Interpretable Model-agnostic Explanations)

We use **LIME** to make the model interpretable and transparent. LIME generates local surrogate models to explain individual predictions, helping us understand the reasons behind the model's decision.

### How LIME Works:
LIME perturbs the input image and uses a surrogate interpretable model (like a decision tree) to approximate the behavior of the complex model for a given instance. This allows us to highlight which parts of the image were influential in the model's prediction.

### Example Code for LIME:
```python
import tensorflow as tf
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array  # Make sure to import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras import models
import numpy as np
# ... (rest of the code)

# Create a LIME explainer
explainer = lime_image.LimeImageExplainer()

# Define a prediction function for LIME
def predict_fn(images):
    images = np.array([preprocess_input(img_to_array(image)) for image in images]) #img_to_array should be accessible here as well.
    return model.predict(images)

# Generate LIME explanation
explanation = explainer.explain_instance(
    img_to_array(load_img(image_path, target_size=(150, 150))), # img_to_array call
    predict_fn,
    top_labels=4,
    num_samples=1000,
    batch_size=10
)


temp, mask = explanation.get_image_and_mask(
    explanation.top_labels[0],
    positive_only=True,
    num_features=10,
    hide_rest=True
)
img_boundry = mark_boundaries(temp / 255.0, mask)
plt.figure(figsize=(10, 5))

# Original image
plt.subplot(1, 2, 1)
plt.imshow(load_img(image_path, target_size=(150, 150)))
plt.title(f"True Class: {predicted_class}")

# LIME explanation
plt.subplot(1, 2, 2)
plt.imshow(img_boundry)
plt.title(f"Predicted Class: {predicted_class}")

plt.show()
```

This code will visualize which regions of the image the model focused on for making predictions, helping clinicians interpret the model's decisions.

---

## 📊 Results

Here’s how the model performed on the test set:

-                  precision    recall  f1-score   support
-  Benign              0.98      0.95      0.97        60
-  Malignant           0.96      0.99      0.98        80
-  accuracy                                0.97       140
-  macro avg           0.97      0.97      0.97       140
-  weighted avg        0.97      0.97      0.97       140


These numbers show how well the model generalizes to new, unseen data. The **LIME** explanations confirm that the model is focusing on relevant areas of the tissue samples, ensuring trust and transparency.

---

## 💬 Contributing

We encourage you to contribute! If you have any improvements, bug fixes, or suggestions for future versions, feel free to fork the repo and submit a pull request.

---

## 📄 License

This project is licensed under the **MIT License**. For more details, check out the [LICENSE](LICENSE) file.

---

## 🙏 Acknowledgements

- **BreakHis Dataset**: For providing a rich set of histopathological images.
- **Keras & TensorFlow**: For the deep learning framework and its vast community.
- **LIME**: For offering an interpretable approach to understanding model predictions.

---

### 🙋‍♂️ Get in Touch!

Have any questions or feedback? Feel free to reach out:

- **Email**: jaindayanshi123@gmail.com

---
