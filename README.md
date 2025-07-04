# Millicent data analysis project

## Overview
Millicent is a data analysis project for academic performance prediction and explainability. It processes student and institutional data, applies various regression models, and provides interpretable results to help educators and administrators understand the factors influencing student grades.

## Installation
1. Install Python 3.9+ (recommended via Homebrew):
    ```bash
    brew update
    brew upgrade python
    python -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```
2. Install additional packages for explainability:
    ```bash
    pip install torch shap lime scikit-learn matplotlib pandas tqdm
    ```

## How to Run
1. Place your data file as `data/data.xlsx`.
2. Run the main analysis:
    ```bash
    python main.py
    ```
3. Results will be saved in the `results/` directory, including:
    - Descriptive statistics, grouped analysis, frequency counts, correlations, outlier analysis, and visualizations
    - Regression results and plots for linear, polynomial, random forest, and neural network models
    - Explainability outputs (feature importance, SHAP, LIME, surrogate formula, etc.)

## What This Project Does
- Reads and analyzes academic data from Excel
- Performs multiple regression analyses to predict student grades:
    - **Linear Regression:** Finds the best straight-line relationship between features and the target grade
    - **Polynomial (Cubic) Regression:** Fits a more flexible curve (degree 3) to capture non-linear relationships
    - **Random Forest Regression:** Uses an ensemble of decision trees to model complex, non-linear patterns and feature interactions
    - **Neural Network Regression:** Uses a feedforward neural network (MLP) to capture highly complex, non-linear relationships, including categorical variables
- Provides explainability for each model, including feature importance, sample predictions, and advanced tools (SHAP, LIME, surrogate models)

## Requirements
- Python 3.9+
- pandas, numpy, scikit-learn, matplotlib, tqdm
- torch (PyTorch)
- shap, lime (for explainability)

## How to Interpret the Results
- **Descriptive and grouped analysis:** Understand the distribution and grouping of your data
- **Regression results:**
    - **R² score:** How much variance in grades is explained by the model (closer to 1 is better)
    - **Mean Squared Error:** Average squared prediction error (lower is better)
    - **Feature importance:** Which features most influence the predictions
    - **Sample predictions:** See how close the model’s predictions are to actual grades
- **Explainability outputs:**
    - **SHAP/LIME:** Visual and tabular explanations of how features impact individual predictions
    - **Surrogate formula:** A simple linear model that approximates the neural network’s behavior for interpretability
    - **Partial dependence plots:** Show how changing one feature affects the predicted grade

## What Each Regression Model Means
- **Linear Regression:**
    - Models the relationship between features and the target as a straight line. Good for simple, linear relationships.
- **Polynomial (Cubic) Regression:**
    - Extends linear regression by fitting a curve (degree 3), allowing for more complex, non-linear relationships.
- **Random Forest Regression:**
    - An ensemble of decision trees. Captures non-linearities and feature interactions. Robust to outliers and overfitting.
- **Neural Network Regression:**
    - Uses layers of interconnected nodes to model highly complex, non-linear relationships. Can handle both numeric and categorical data. Less interpretable, but very powerful.

## Using the Trained Model
- The neural network model and its preprocessing metadata are saved in `results/nn_regression/`.
- You can use these files to make predictions on new data by applying the same preprocessing steps and loading the model.

## Contact
For questions or contributions, please contact the project maintainer.
