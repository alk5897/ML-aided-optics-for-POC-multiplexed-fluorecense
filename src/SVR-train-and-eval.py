import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Construct the path to the dataset
dataset_path = os.path.join('datasets', 'three-fluorophore-calibration-data-for-model-eval.xlsx')

# Read and preprocess the data
data = pd.read_excel(dataset_path)

# Define targets and features
y = data[['FAM', 'ATTO550', 'Cy5']]
X = data.drop(columns=['FAM', 'ATTO550', 'Cy5'])

# Scaling the data
scaler_X = StandardScaler().fit(X)
X_normalized = scaler_X.transform(X)
scalers_y = {col: StandardScaler().fit(y[[col]]) for col in y.columns}
scaled_targets = [scalers_y[col].transform(y[[col]]) for col in y.columns]
y_normalized = np.hstack(scaled_targets)
X_train, X_test, y_train, y_test = train_test_split(X_normalized, y_normalized, test_size=0.20, random_state=39)# random_state=42, Splitting the dataset into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X_normalized, Y_normalized, test_size=0.20, random_state=42)

# Number of target variables
num_targets = y.shape[1]  # Assuming y is the original DataFrame with target variable

# Train a SVR model for each target using integer indices to access columns
svr_models = {}
for i in range(num_targets):
    svr_model = SVR(kernel='rbf', C=100, epsilon=0.001)
    svr_model.fit(X_train, y_train[:, i])
    svr_models[y.columns[i]] = svr_model

# Check for outputs directory and create excel writers to save data
outputs_dir = os.path.join('outputs')
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
predictions_path = os.path.join(outputs_dir, 'SVR-predictions.xlsx')
mae_mse_path = os.path.join(outputs_dir, 'SVR-concentrationWise-MAE&MSE.xlsx')
residuals_path = os.path.join(outputs_dir, 'SVR-histogramOfResiduals.xlsx')
writer_predictions = pd.ExcelWriter(predictions_path, engine='openpyxl')
writer_mae_mse = pd.ExcelWriter(mae_mse_path, engine='openpyxl')
writer_residuals = pd.ExcelWriter(residuals_path, engine='openpyxl')

# Evaluate the models and save results
for i, col in enumerate(y.columns):
    y_pred = svr_models[col].predict(X_test)
    y_pred_rescaled = scalers_y[col].inverse_transform(y_pred.reshape(-1, 1)).flatten()
    #y_test_rescaled = y_test[:, i]  # Accessing the i-th column using integer index
    y_test_rescaled = scalers_y[col].inverse_transform(y_test[:, i].reshape(-1, 1)).flatten()

    # Calculate metrics
    MAE = mean_absolute_error(y_test_rescaled, y_pred_rescaled)
    MSE = mean_squared_error(y_test_rescaled, y_pred_rescaled)
    R2 = r2_score(y_test_rescaled, y_pred_rescaled)

    print(f'For dye {col}:')
    print(f'MAE: {MAE}, MSE: {MSE}, R2: {R2}')

    # Save results to Excel files
    results_df = pd.DataFrame({'Actual': y_test_rescaled, 'Predicted': y_pred_rescaled})
    residuals_df = pd.DataFrame({'Residuals': y_test_rescaled - y_pred_rescaled})
    count, bin_edges = np.histogram(residuals_df, bins=30)
    histogram_data = pd.DataFrame({'BinEdges': bin_edges[:-1], 'Count': count})
    results_df.to_excel(writer_predictions, sheet_name = f'{col}', index=False)
    histogram_data.to_excel(writer_residuals, sheet_name= f'{col}', index=False)

    # Concentration-wise MAE and MSE
    concentration_wise = results_df.groupby('Actual').agg(['mean', 'std'])
    concentration_wise.columns = ['_'.join(col).strip() for col in concentration_wise.columns.values]
    concentration_wise['MAE'] = abs(concentration_wise['Predicted_mean'] - concentration_wise.index)
    concentration_wise['MSE'] = (concentration_wise['Predicted_mean'] - concentration_wise.index) ** 2
    concentration_wise[['MAE', 'MSE']].to_excel(writer_mae_mse, sheet_name= f'{col}')



# Save the models and scalers
models_dir = os.path.join('models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)


for col, svr_model in svr_models.items():
    # Save the SVR mode
    model_path = os.path.join(models_dir, f'SVR-trained-model-{col}.pkl')
    joblib.dump(svr_model, model_path)
    # Save the corresponding scaler
    y_scaler_path = os.path.join(models_dir, f'SVR-y-scaler-{col}.pkl')
    joblib.dump(scalers_y[col],y_scaler_path)

print('SVR modeling and evaluation complete. Models and scalers saved.')