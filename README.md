Predictive Modeling Tool
Machine Learning for Regression Analysis

MLs v1.0

1. Overview
This tool performs predictive modeling using:
•	Regression Trees (RT)
•	Random Forests (RF)
•	Support Vector Machines (SVM)
•	Artificial Neural Networks (ANN)
Key features:
✔ Automated data preprocessing
✔ Model comparison with visualizations
✔ Comprehensive statistical reporting
✔ Excel-based results export

2. System Requirements
Component	Requirement
OS	Windows 10/11, macOS 10.15+, Linux
Python	3.8 or newer
RAM	8GB minimum (16GB recommended)
Disk Space	500MB available
Required Python Packages:
text
pandas, numpy, matplotlib, scikit-learn, openpyxl

3. Installation
Install Python from python.org
Install required packages:
bash
•  pip install pandas numpy matplotlib scikit-learn openpyxl
•  Download the script file (predictive_modeling.py)

4. Input Data Format
File Format: .xlsx (Excel)
Structure:
•	Predictor variables in columns A to N
•	Target variable in the last column
Example:
Feature1	Feature2	...	Target
10.5	20.1	...	50.2

5. Model Descriptions

5.1 Regression Tree (RT)
•	Algorithm: CART (Classification and Regression Trees)
•	Parameters:
o	Max depth: Optimized automatically
o	Splitting criterion: MSE

5.2 Random Forest (RF)
•	Ensemble of 100 decision trees
•	Features: Bootstrap aggregation, feature importance

5.3 Support Vector Machine (SVM)
•	Kernel: Radial Basis Function (RBF)
•	Parameters:
o	C = 100 (Regularization)
o	gamma = 0.1

5.4 Artificial Neural Network (ANN)
•	Architecture: 2 hidden layers (100 → 50 neurons)
•	Activation: ReLU
•	Optimizer: Adam

6. Output Files
File	Description
model_comparison.png	Scatter plots of predicted vs actual values
model_results.xlsx	Contains:
- Model Metrics	R², RMSE, MAE, Nash-Sutcliffe
- Predictions	Actual vs predicted values per model

7. Performance Metrics
	
R²		
RMSE		
MAE				
Nash-Sutcliffe			




[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15638616.svg)](https://doi.org/10.5281/zenodo.15638616)


