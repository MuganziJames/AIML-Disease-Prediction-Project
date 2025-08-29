# Disease Prediction Toolkit - Heart Disease Risk Assessment

Colab-first workflow to train and evaluate heart disease prediction models.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [What You'll Learn](#what-youll-learn)
- [Dataset](#dataset)
- [Features](#features)
- [Models](#models)
- [Results](#results)
- [Usage](#usage)
- [Installation](#installation)
- [Contributing](#contributing)

## Overview

This toolkit provides a comprehensive solution for heart disease prediction using machine learning. It includes:

- **Complete ML Pipeline**: From data preprocessing to model deployment
- **Multiple Models**: Logistic Regression and Random Forest with hyperparameter tuning
- **Rich Visualizations**: Confusion matrices, ROC curves, and feature analysis
- **Easy-to-Use Demo**: Ready-to-run prediction examples
- **Production Ready**: Saved models with metadata for deployment

## Project Structure

```
disease-prediction-toolkit/
├── data/
│   └── heart_dataset.csv
├── notebooks/
│   └── colab_training.ipynb
└── README.md
```

## Quick Start

### 1. Installation

```bash
# Clone or download the project
cd disease-prediction-toolkit

# Install dependencies
pip install -r requirements.txt
```

### 2. Run in Google Colab

1. Open `notebooks/colab_training.ipynb` in Google Colab
2. When prompted, upload `data/heart_dataset.csv`
3. Run all cells to preprocess, train, evaluate, and visualize models

### 3. Make Predictions

Use the notebook outputs; metrics and plots are displayed in Colab.

## Dataset

The heart disease dataset contains 17 clinical features:

### Numerical Features:

- **age**: Age in years
- **trestbps**: Resting blood pressure (mm Hg)
- **chol**: Serum cholesterol (mg/dl)
- **thalch**: Maximum heart rate achieved
- **oldpeak**: ST depression induced by exercise
- **ca**: Number of major vessels colored by fluoroscopy (0-3)

### Categorical Features:

- **sex**: Gender (Male/Female - one-hot encoded)
- **cp**: Chest pain type (4 types - one-hot encoded)
- **fbs**: Fasting blood sugar > 120 mg/dl (boolean)
- **restecg**: Resting electrocardiographic results
- **exang**: Exercise induced angina (boolean)
- **slope**: Slope of peak exercise ST segment
- **thal**: Thalassemia type

### Target Variable:

- **target**: Heart disease risk (0 = Low Risk, 1 = High Risk)
  - Created using medical risk factors and clinical guidelines

## Features

### Data Analysis

- **Comprehensive EDA**: Distribution analysis, correlation matrices, missing value assessment
- **Feature Engineering**: Risk score calculation, categorical encoding, data preprocessing
- **Visualization**: Box plots, histograms, heatmaps, and statistical summaries

### Machine Learning

- **Multiple Algorithms**: Logistic Regression, Random Forest
- **Hyperparameter Tuning**: GridSearchCV with cross-validation
- **Model Evaluation**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Model Persistence**: Automatic saving of best models with metadata

### Visualization & Reporting

- **Performance Metrics**: Detailed classification reports
- **Confusion Matrices**: Visual representation of model performance
- **ROC Curves**: Threshold analysis and AUC visualization
- **Model Comparison**: Side-by-side performance charts

## Models

### Logistic Regression

- **Type**: Linear classifier
- **Hyperparameters**: C (regularization), penalty, solver
- **Advantages**: Interpretable, fast, good baseline

### Random Forest

- **Type**: Ensemble method
- **Hyperparameters**: n_estimators, max_depth, min_samples_split
- **Advantages**: Handles non-linearity, feature importance, robust

### Model Selection

- **Primary Metric**: ROC-AUC score
- **Secondary Metric**: F1-Score
- **Validation**: Cross-validation with stratified splits

## Results

The models achieve competitive performance on heart disease prediction:

### Performance Metrics

| Model               | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| ------------------- | -------- | --------- | ------ | -------- | ------- |
| Logistic Regression | 0.85+    | 0.80+     | 0.75+  | 0.77+    | 0.85+   |
| Random Forest       | 0.80+    | 0.85+     | 0.70+  | 0.75+    | 0.80+   |

_Note: Actual results may vary based on the specific dataset and random seed._

### Key Insights

- Age and chest pain type are strong predictors
- Blood pressure and cholesterol show clear patterns
- Exercise-induced symptoms are significant indicators
- Gender differences in risk factors

## Usage

### For Data Scientists

1. **Training**: Use `notebooks/colab_training.ipynb` in Colab to train/evaluate

### For Developers

1. **Integration**: Load models using `joblib.load('models/best_model.pkl')`
2. **Deployment**: Models are ready for production deployment

### For Healthcare Professionals

1. **Interpretation**: Review feature importance and risk factors
2. **Validation**: Compare predictions with clinical expertise

## Installation

Not required for Colab. Open the notebook in Google Colab and upload the dataset when prompted.

### Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **scikit-learn**: Machine learning algorithms
- **matplotlib**: Basic plotting
- **seaborn**: Statistical visualization
- **plotly**: Interactive visualizations
- **jupyter**: Notebook environment
- **joblib**: Model serialization

## Model Deployment

Not applicable for Colab-only usage in this project.

## Contributing

We welcome contributions! Please see our guidelines:

1. **Issues**: Report bugs or request features via GitHub issues
2. **Pull Requests**: Submit improvements with clear descriptions
3. **Documentation**: Help improve docs and examples
4. **Testing**: Add tests for new functionality

### Development Setup

```bash
# Fork the repository
git clone https://github.com/your-username/disease-prediction-toolkit.git

# Create feature branch
git checkout -b feature/new-feature

# Make changes and test
jupyter notebook notebooks/

# Submit pull request
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This tool is for educational and research purposes only. It should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare providers for medical decisions.

## Contact

For questions, suggestions, or collaboration opportunities:

- **Issues**: GitHub Issues page
- **Discussions**: GitHub Discussions
- **Email**: [Your contact information]

---

**Made with ❤️ for the healthcare and ML community**

## What You'll Learn

- End-to-end ML workflow on a healthcare dataset
- Data preprocessing with imputers, encoding, and scaling
- Training and comparing Logistic Regression, Decision Tree, and Random Forest
- Evaluating with Accuracy, Precision, Recall, F1, and ROC-AUC
- Visualizing performance via confusion matrix and ROC curve

### Google Colab Usage

1. Open `notebooks/colab_training.ipynb` in Google Colab
2. When prompted, upload `data/heart_dataset.csv`
3. Run all cells to preprocess, train, evaluate, and visualize models
4. Review metrics in the displayed dataframe and plots
