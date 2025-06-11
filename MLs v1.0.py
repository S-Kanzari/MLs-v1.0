import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import openpyxl

def nash_sutcliffe(y_true, y_pred):
    """Calculate Nash-Sutcliffe efficiency coefficient"""
    return 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))

def evaluate_model(y_true, y_pred, model_name):
    """Calculate all evaluation metrics"""
    metrics = {
        'Model': model_name,
        'R²': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'Absolute Error': mean_absolute_error(y_true, y_pred),
        'Nash Coefficient': nash_sutcliffe(y_true, y_pred),
        'Mean Error': np.mean(y_true - y_pred)
    }
    return metrics

def plot_results(y_true, y_pred, model_name, ax):
    """Plot measured vs predicted values"""
    ax.scatter(y_true, y_pred, alpha=0.5)
    ax.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'k--', lw=2)
    ax.set_xlabel('Measured')
    ax.set_ylabel('Predicted')
    ax.set_title(f'{model_name} (R² = {r2_score(y_true, y_pred):.3f})')
    
    # Add text box with metrics
    metrics_text = (
        f"R² = {r2_score(y_true, y_pred):.3f}\n"
        f"RMSE = {np.sqrt(mean_squared_error(y_true, y_pred)):.3f}\n"
        f"MAE = {mean_absolute_error(y_true, y_pred):.3f}\n"
        f"Nash = {nash_sutcliffe(y_true, y_pred):.3f}"
    )
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

# 1. Load data from Excel file
file_path = 'Input.xlsx'  # Change to your file path
data = pd.read_excel(file_path)

# Assume the last column is the target, others are predictors
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# 2. Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_scaled = scaler.transform(X)  # Scale the complete feature matrix

# 3. Initialize models
models = {
    'Regression Tree': DecisionTreeRegressor(random_state=42),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Support Vector Machine': SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1),
    'Artificial Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), 
                                            activation='relu', 
                                            solver='adam', 
                                            max_iter=1000, 
                                            random_state=42)
}

# 4. Train models, make predictions, and evaluate
results = []
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.ravel()

for (name, model), ax in zip(models.items(), axes):
    if name in ['Support Vector Machine', 'Artificial Neural Network']:
        # Use scaled data for SVM and ANN
        model.fit(X_train_scaled, y_train)
        y_pred_train = model.predict(X_train_scaled)
        y_pred_val = model.predict(X_val_scaled)
        y_pred_all = model.predict(X_scaled)  # Predictions for complete dataset
    else:
        # Use unscaled data for tree-based methods
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_all = model.predict(X)  # Predictions for complete dataset
    
    # Evaluate on validation set
    val_metrics = evaluate_model(y_val, y_pred_val, name)
    results.append(val_metrics)
    
    # Plot validation results
    plot_results(y_val, y_pred_val, name, ax)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()

# 5. Convert results to DataFrame and save to Excel
results_df = pd.DataFrame(results)
results_df.set_index('Model', inplace=True)

# Save to Excel
output_path = 'model_results.xlsx'
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    results_df.to_excel(writer, sheet_name='Model Metrics')
    
    # Add raw predictions for each model
    for name, model in models.items():
        if name in ['Support Vector Machine', 'Artificial Neural Network']:
            y_pred = model.predict(X_scaled)
        else:
            y_pred = model.predict(X)
            
        pred_df = pd.DataFrame({
            'Actual': y,
            'Predicted': y_pred,
            'Error': y - y_pred
        })
        pred_df.to_excel(writer, sheet_name=f'{name} Predictions')

print("Modeling completed. Results saved to:", output_path)
print("\nModel Performance Metrics:")
print(results_df)