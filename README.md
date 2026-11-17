# Diabetic Retinopathy Detection Using Deep Learning and MobileNetV2

![Python](https://img.shields.io/badge/Python-3.12-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-orange)
![Deep Learning](https://img.shields.io/badge/Deep%20Learning-CNN-green)
![Model](https://img.shields.io/badge/Model-MobileNetV2-red)
![Status](https://img.shields.io/badge/Status-Completed-success)

**Project Type:** College Project

## Overview

Diabetic Retinopathy (DR) is a diabetes-related eye disease that affects the retina and can lead to vision impairment if not identified and managed appropriately.

This project demonstrates a deep-learning-based image classification system that analyzes retinal fundus photographs and classifies them into five diabetic retinopathy severity grades.

The system uses **MobileNetV2 Transfer Learning and Fine-Tuning**, Convolutional Neural Networks (CNN), Grad-CAM visualization and a Streamlit web application.

The objective is to explore the application of deep learning in retinal image classification through a complete machine-learning workflow.

**Note:** This is an educational prototype and is not a clinically validated diagnostic system.

---

## Objectives

- Study diabetic retinopathy and its five severity grades.
- Explore and preprocess retinal fundus images.
- Develop a baseline CNN classification model.
- Implement MobileNetV2 transfer learning.
- Improve the model through fine-tuning.
- Handle class imbalance using class weights.
- Evaluate model performance using classification metrics.
- Visualize model predictions using Grad-CAM.
- Develop an interactive Streamlit web application.

---

## Dataset Information

**Dataset:** APTOS 2019 Blindness Detection

**Source:** [Kaggle – APTOS 2019](https://www.kaggle.com/c/aptos2019-blindness-detection/data)

Total images: **3,662**

### Class Distribution

| Label | DR Grade | Images |
|---|---|---:|
| 0 | No DR | 1,805 |
| 1 | Mild | 370 |
| 2 | Moderate | 999 |
| 3 | Severe | 193 |
| 4 | Proliferative DR | 295 |
| | **Total** | **3,662** |

### Dataset Splitting

| Dataset | Images | Percentage |
|---|---:|---:|
| Training | 2,562 | 70% |
| Validation | 550 | 15% |
| Testing | 550 | 15% |
| **Total** | **3,662** | **100%** |

---

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| TensorFlow | Deep learning framework |
| Keras | Neural network development |
| MobileNetV2 | Transfer learning architecture |
| NumPy | Numerical computation |
| Pandas | Dataset management |
| Scikit-learn | Model evaluation |
| Matplotlib | Visualization |
| Seaborn | Confusion matrix visualization |
| Streamlit | Web application |
| Jupyter Notebook | Model development |

---

## Model Architecture

Three main model configurations were explored:

### 1. Baseline CNN
A convolutional neural network developed from scratch to establish a baseline for retinal image classification.

### 2. MobileNetV2 Transfer Learning
An ImageNet-pretrained MobileNetV2 network with a custom five-class classification head.

### 3. Fine-Tuned MobileNetV2
The pretrained model is adapted to retinal image classification by selectively unfreezing the last 20 layers and training with a lower learning rate.

**Final selected model:**

`mobilenetv2_independent_finetuned_best.keras`

### Image Preprocessing

- RGB image conversion.
- Resizing to 224 × 224 pixels.
- Aspect-ratio preservation using resize-with-padding.
- MobileNetV2 preprocessing.
- Pixel normalization to approximately [-1, 1].
- Class weighting to address class imbalance.

---

## Final Model Performance

The final fine-tuned MobileNetV2 model was evaluated on 550 held-out test images.

### Overall Evaluation

| Metric | Score |
|---|---:|
| Test Accuracy | **77.64%** |
| Test Loss | 0.5597 |
| Macro Precision | 62.90% |
| Macro Recall | 63.29% |
| Macro F1-Score | 62.18% |
| Weighted F1-Score | 77.27% |
| Balanced Accuracy | 63.29% |
| Quadratic Weighted Kappa | 0.8048 |

### Class-Wise Performance

| DR Grade | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| No DR | 94.57% | 96.31% | 95.43% |
| Mild | 51.76% | 78.57% | 62.41% |
| Moderate | 71.76% | 62.67% | 66.90% |
| Severe | 46.43% | 44.83% | 45.61% |
| Proliferative DR | 50.00% | 34.09% | 40.54% |

**Observation:** The model performs best on No DR images. Performance is weaker for Severe and Proliferative DR, indicating that further improvement and validation are required.

---

## Grad-CAM Visualization

Gradient-weighted Class Activation Mapping (Grad-CAM) is implemented to visualize image regions contributing to model predictions.

The system generates:

- Original retinal image.
- Grad-CAM heatmap.
- Heatmap overlay on the original image.

Grad-CAM provides a coarse visualization of model behaviour. It does not provide precise lesion segmentation or establish clinical correctness.

---

## Streamlit Web Application

An interactive web application has been developed using Streamlit.

### Features

- Upload retinal fundus images.
- Automatically preprocess uploaded images.
- Predict one of five diabetic retinopathy grades.
- Display class-wise prediction scores.
- Generate Grad-CAM heatmaps.
- Display heatmap overlays.

---

## Project Structure

```text
Diabetic-Retinopathy-Detection/
│
├── app/
│   └── app.py
│
├── dataset/
│   └── README.md
│
├── models/
│   └── mobilenetv2_independent_finetuned_best.keras
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_image_preprocessing.ipynb
│   ├── 03_cnn_baseline.ipynb
│   ├── 04_mobilenetv2_training.ipynb
│   ├── 05_model_evaluation.ipynb
│   └── 06_independent_test_evaluation.ipynb
│
├── outputs/
│   ├── gradcam/
│   └── screenshots/
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR-USERNAME/diabetic-retinopathy-detection.git
```

### 2. Navigate to the Project Directory

```bash
cd diabetic-retinopathy-detection
```

### 3. Create a Virtual Environment

```bash
python -m venv .venv
```

### 4. Activate the Environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the Dataset

Download the APTOS 2019 dataset from Kaggle and place the retinal images inside:

```text
dataset/train_images/
```

The dataset is not included in this repository. Follow the dataset provider's terms when downloading or using it.

### 7. Run the Streamlit Application

```bash
streamlit run app/app.py
```

Open the application in your browser:

```text
http://localhost:8501
```

Ensure that the final trained model is available at:

```text
models/mobilenetv2_independent_finetuned_best.keras
```

---

## Limitations

- The model has been developed using the APTOS 2019 dataset.
- External validation on a separate hospital dataset has not been performed.
- Performance varies across the five DR severity classes.
- Severe and Proliferative DR classification requires further improvement.
- Softmax scores are model outputs, not calibrated medical probabilities.
- Grad-CAM visualizations do not establish clinical correctness.

**This system is intended for educational and research purposes only. It must not be used as a substitute for professional medical examination, diagnosis or treatment.**

---

## Future Scope

- Improve classification performance for minority DR classes.
- Experiment with EfficientNet and other architectures.
- Evaluate the model using external retinal image datasets.
- Explore advanced image enhancement techniques.
- Improve model interpretability.
- Conduct clinical validation with qualified ophthalmologists.

---

## References

1. APTOS 2019 Blindness Detection Dataset, Kaggle.
2. Sandler et al., "MobileNetV2: Inverted Residuals and Linear Bottlenecks," CVPR, 2018.
3. Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks via Gradient-Based Localization," ICCV, 2017.
4. [TensorFlow Documentation](https://www.tensorflow.org/)
5. [Streamlit Documentation](https://docs.streamlit.io/)

---

**College Project | 2026**