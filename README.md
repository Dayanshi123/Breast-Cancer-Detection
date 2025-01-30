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
    git clone https://github.com/your-username/breast-cancer-classification.git
    cd breast-cancer-classification
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
python train_model.py
```

- **Training Configuration**: The model will train on the BreakHis dataset, utilizing data augmentation and dropout for regularization.
- **Output**: The best model weights will be saved in `model_weights.h5`.

---

## 🏆 Evaluation

After training, you can evaluate the model's performance with:

```bash
python evaluate_model.py
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
from lime import lime_image
from sklearn.linear_model import LogisticRegression
import numpy as np

# Create an explainer
explainer = lime_image.LimeImageExplainer()

# Choose an image for explanation
image = test_image[0]

# Explain the model's prediction on this image
explanation = explainer.explain_instance(image, model.predict, top_labels=5, hide_color=0, num_samples=1000)

# Visualize the explanation
explanation.show_in_browser()
```

This code will visualize which regions of the image the model focused on for making predictions, helping clinicians interpret the model's decisions.

---

## 📊 Results

Here’s how the model performed on the test set:

- **Accuracy**: 97%
- **Precision**: 88%
- **Recall**: 92%
- **F1-Score**: 90%

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
