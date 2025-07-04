import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import pickle
import shap
import lime
import lime.lime_tabular

EXCLUDE_COLS = [
    "DICTAMEN",
    "Puntaje Obtenido",
    "Porcentaje ",
    "Porcentaje EXIL ",
    "IECONOMIA",
    "IADMINTRA",
    "ICONTAFIN",
    "IADERECHO",
    "IMERCADOC",
    "IMATEMEST"
]

def preprocess_features(df, target_col):
    # Exclude only the target column
    X = df.drop(columns=[target_col], errors='ignore')
    # Replace NaN in all columns with string 'NaN' for categorical encoding
    X = X.apply(lambda col: col.astype(str).where(col.notnull(), 'NaN') if col.dtype == 'object' or str(col.dtype).startswith('category') else col)
    # Drop datetime columns
    X = X.select_dtypes(exclude=["datetime", "datetime64[ns]"])
    # One-hot encode all categorical columns
    X = pd.get_dummies(X, drop_first=True)
    # Fill any remaining NaNs with 0 (for numeric columns)
    X = X.fillna(0)
    return X

def linear_regression(filepath="data/data.xlsx", target_col="Calificación 3 er parcial"):
    """
    Performs multiple linear regression to predict the target column using all other numeric columns.
    Saves results and plots to results/linear_regression/.
    """
    # Load data
    df = pd.read_excel(filepath)
    os.makedirs("results/linear_regression", exist_ok=True)
    # Drop rows with missing target
    df = df.dropna(subset=[target_col])
    X = preprocess_features(df, target_col)
    y = df[target_col]
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Fit model
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    # Metrics
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    # Save results
    with open("results/linear_regression/data.txt", "w") as f:
        f.write("Linear Regression Results\n")
        f.write("------------------------\n")
        f.write(f"R^2 score: {r2:.4f}\n")
        f.write(f"Mean Squared Error: {mse:.4f}\n\n")
        f.write("Coefficients:\n")
        for col, coef in zip(X.columns, model.coef_):
            f.write(f"{col}: {coef:.4f}\n")
        f.write(f"Intercept: {model.intercept_:.4f}\n")
    # Plot predicted vs actual
    plt.figure(figsize=(7,5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual: Calificación 3 er parcial")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.savefig("results/linear_regression/predicted_vs_actual.png")
    plt.close()
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(7,5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted")
    plt.savefig("results/linear_regression/residuals_vs_predicted.png")
    plt.close()
    print("Linear regression analysis complete. Results and plots saved in results/linear_regression/.")

def polynomial_regression(filepath="data/data.xlsx", target_col="Calificación 3 er parcial", degree=3):
    """
    Performs cubic polynomial regression to predict the target column using all other numeric columns.
    Saves results and plots to results/polynomial_regression/.
    """
    print("Starting cubic polynomial regression (degree 3)...")
    df = pd.read_excel(filepath)
    os.makedirs("results/polynomial_regression", exist_ok=True)
    df = df.dropna(subset=[target_col])
    X = preprocess_features(df, target_col)
    y = df[target_col]
    # Polynomial features
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    print(f"Transformed features to degree {degree} polynomial features. Shape: {X_poly.shape}")
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    print("Training polynomial regression model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    # Save results
    with open("results/polynomial_regression/data.txt", "w") as f:
        f.write(f"Polynomial Regression Results (degree {degree})\n")
        f.write("------------------------------------------\n")
        f.write(f"R^2 score: {r2:.4f}\n")
        f.write(f"Mean Squared Error: {mse:.4f}\n\n")
        f.write("First 10 coefficients (due to high dimensionality):\n")
        for i, coef in enumerate(model.coef_[:10]):
            f.write(f"coef_{i}: {coef:.4f}\n")
        f.write(f"...\nTotal coefficients: {len(model.coef_)}\n")
        f.write(f"Intercept: {model.intercept_:.4f}\n")
    # Plot predicted vs actual
    plt.figure(figsize=(7,5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(f"Predicted vs Actual: Calificación 3 er parcial (degree {degree})")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.savefig("results/polynomial_regression/predicted_vs_actual.png")
    plt.close()
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(7,5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title(f"Residuals vs Predicted (degree {degree})")
    plt.savefig("results/polynomial_regression/residuals_vs_predicted.png")
    plt.close()
    print("Polynomial regression analysis complete. Results and plots saved in results/polynomial_regression/.")

def random_forest_regression(filepath="data/data.xlsx", target_col="Calificación 3 er parcial"):
    """
    Performs random forest regression to predict the target column using all other numeric columns.
    Saves results and plots to results/random_forest_regression/.
    """
    print("Starting random forest regression...")
    df = pd.read_excel(filepath)
    os.makedirs("results/random_forest_regression", exist_ok=True)
    df = df.dropna(subset=[target_col])
    X = preprocess_features(df, target_col)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    print("Training random forest model...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    # Save results
    with open("results/random_forest_regression/data.txt", "w") as f:
        f.write("Random Forest Regression Results\n")
        f.write("-------------------------------\n")
        f.write(f"R^2 score: {r2:.4f}\n")
        f.write(f"Mean Squared Error: {mse:.4f}\n\n")
        f.write("Feature Importances:\n")
        importances = model.feature_importances_
        for col, imp in sorted(zip(X.columns, importances), key=lambda x: -x[1]):
            f.write(f"{col}: {imp:.4f}\n")
    # Plot predicted vs actual
    plt.figure(figsize=(7,5))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual: Calificación 3 er parcial (Random Forest)")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.savefig("results/random_forest_regression/predicted_vs_actual.png")
    plt.close()
    # Plot residuals
    residuals = y_test - y_pred
    plt.figure(figsize=(7,5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted (Random Forest)")
    plt.savefig("results/random_forest_regression/residuals_vs_predicted.png")
    plt.close()
    print("Random forest regression analysis complete. Results and plots saved in results/random_forest_regression/.")

def nn_regression(filepath="data/data.xlsx", target_col="Calificación 3 er parcial", epochs=100, batch_size=32):
    """
    Performs regression using a feedforward neural network (MLP) with PyTorch.
    Saves results and plots to results/nn_regression/.
    """
    print("Starting feedforward neural network regression (PyTorch)...")
    df = pd.read_excel(filepath)
    os.makedirs("results/nn_regression", exist_ok=True)
    df = df.dropna(subset=[target_col])
    X = preprocess_features(df, target_col)
    y = df[target_col]
    # Normalize features
    X_mean, X_std = X.mean(), X.std()
    X = (X - X_mean) / X_std
    # Fill any remaining NaNs with 0 (after normalization)
    X = X.fillna(0)
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X.values, y.values, test_size=0.2, random_state=42)
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    # DataLoader
    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    # Define model
    model = nn.Sequential(
        nn.Linear(X_train.shape[1], 64),
        nn.ReLU(),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Linear(32, 1)
    )
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    # Training loop with progress bar
    print("Training neural network...")
    for epoch in tqdm(range(epochs), desc="Epochs"):
        model.train()
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
    # Evaluation
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test).numpy().flatten()
        y_true = y_test.numpy().flatten()
    r2 = r2_score(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    # Explainability: statistics
    residuals = y_true - y_pred
    pred_stats = f"Predictions: mean={np.mean(y_pred):.4f}, std={np.std(y_pred):.4f}, min={np.min(y_pred):.4f}, max={np.max(y_pred):.4f}\n"
    resid_stats = f"Residuals: mean={np.mean(residuals):.4f}, std={np.std(residuals):.4f}, min={np.min(residuals):.4f}, max={np.max(residuals):.4f}\n"
    # Feature importance: abs(weight) of first layer
    first_layer_weights = model[0].weight.detach().cpu().numpy()
    feature_importance = np.abs(first_layer_weights).sum(axis=0)
    feature_names = X.columns if hasattr(X, 'columns') else [f'feat_{i}' for i in range(X.shape[1])]
    feat_imp = sorted(zip(feature_names, feature_importance), key=lambda x: -x[1])
    feat_imp_str = '\n'.join([f"{name}: {imp:.4f}" for name, imp in feat_imp[:20]])
    # Sample predictions
    sample_str = '\n'.join([f"Actual: {a:.3f}, Predicted: {p:.3f}, Residual: {r:.3f}" for a, p, r in zip(y_true[:10], y_pred[:10], residuals[:10])])
    # Save results
    with open("results/nn_regression/data.txt", "w") as f:
        f.write("Feedforward Neural Network Regression Results (PyTorch)\n")
        f.write("------------------------------------------------------\n")
        f.write(f"R^2 score: {r2:.4f}\n")
        f.write(f"Mean Squared Error: {mse:.4f}\n\n")
        f.write(pred_stats)
        f.write(resid_stats)
        f.write("\nTop 20 Most Influential Features (by abs(weight) in first layer):\n")
        f.write(feat_imp_str + "\n")
        f.write("\nSample Predictions (first 10):\n")
        f.write(sample_str + "\n")
    # Save model and metadata
    torch.save(model.state_dict(), "results/nn_regression/model.pth")
    metadata = {
        'feature_names': feature_names,
        'X_mean': X_mean,
        'X_std': X_std
    }
    with open("results/nn_regression/metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)
    # Plot predicted vs actual
    plt.figure(figsize=(7,5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Predicted vs Actual: Calificación 3 er parcial (NN)")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.savefig("results/nn_regression/predicted_vs_actual.png")
    plt.close()
    # Plot residuals
    residuals = y_true - y_pred
    plt.figure(figsize=(7,5))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted")
    plt.ylabel("Residuals")
    plt.title("Residuals vs Predicted (NN)")
    plt.savefig("results/nn_regression/residuals_vs_predicted.png")
    plt.close()
    # SHAP explanations using KernelExplainer for tabular data
    def model_predict(X_arr):
        X_tensor = torch.tensor(X_arr, dtype=torch.float32)
        with torch.no_grad():
            return model(X_tensor).numpy().flatten()
    background = X_train.numpy()[:100]
    explainer = shap.KernelExplainer(model_predict, background)
    shap_values = explainer.shap_values(X_test.numpy()[:10], nsamples=100)
    shap.summary_plot(shap_values, X_test.numpy()[:10], feature_names=feature_names, show=False)
    plt.savefig("results/nn_regression/shap_summary.png")
    plt.close()
    # SHAP force plot for first test sample
    shap.initjs()
    force_plot = shap.force_plot(explainer.expected_value, shap_values[0], X_test.numpy()[0], feature_names=feature_names, matplotlib=True, show=False)
    plt.savefig("results/nn_regression/shap_force_sample.png")
    plt.close()
    # LIME explanation for first test sample
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(X_train.numpy(), feature_names=feature_names, class_names=[target_col], mode='regression')
    lime_exp = lime_explainer.explain_instance(X_test.numpy()[0], lambda x: model(torch.tensor(x, dtype=torch.float32)).detach().numpy().flatten(), num_features=10)
    lime_exp.save_to_file("results/nn_regression/lime_sample.html")
    # Partial Dependence Plots for top 3 features
    from sklearn.inspection import partial_dependence, PartialDependenceDisplay
    from sklearn.linear_model import LinearRegression as SklearnLinearRegression
    # Fit surrogate model
    surrogate = SklearnLinearRegression()
    surrogate.fit(X_train.numpy(), model(X_train).detach().numpy().flatten())
    surrogate_coefs = sorted(zip(feature_names, surrogate.coef_), key=lambda x: -abs(x[1]))
    surrogate_str = '\n'.join([f"{name}: {coef:.4f}" for name, coef in surrogate_coefs[:20]])
    # Save surrogate formula
    with open("results/nn_regression/surrogate_formula.txt", "w") as f:
        f.write("Global Surrogate Linear Regression (approximates NN predictions)\n")
        f.write("--------------------------------------------------------------\n")
        f.write(surrogate_str + "\n")
    # Partial dependence for top 3 features
    feature_names_list = list(feature_names)
    top3 = [name for name, _ in surrogate_coefs[:3]]
    for feat in top3:
        feat_idx = feature_names_list.index(feat)
        disp = PartialDependenceDisplay.from_estimator(surrogate, X_train.numpy(), features=[feat_idx], feature_names=feature_names_list)
        plt.savefig(f"results/nn_regression/partial_dependence_{feat}.png")
        plt.close()
    print("Neural network regression analysis complete. Results and plots saved in results/nn_regression/.")