import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Construct the path to the dataset
dataset_path = os.path.join('datasets', 'scalability-four-fluorophore-data.xlsx')
# Read and preprocess the data
data = pd.read_excel(dataset_path)
y_inverse = data[['ATTO425', 'FAM', 'ATTO550', 'Cy5']]  # Corrected column names
X_inverse = data.drop(columns=['ATTO425', 'FAM', 'ATTO550', 'Cy5'])  # Corrected column names
scaler_X_inverse = StandardScaler().fit(X_inverse)
scaler_y_inverse = StandardScaler().fit(y_inverse)
X_normalized_inverse = scaler_X_inverse.transform(X_inverse)
y_normalized_inverse = scaler_y_inverse.transform(y_inverse)
X_train_inverse, X_test_inverse, y_train_inverse, y_test_inverse = train_test_split(X_normalized_inverse, y_normalized_inverse, test_size=0.20, random_state=41)
train_inverse, X_test_inverse, y_train_inverse, y_test_inverse = train_test_split(X_normalized_inverse, y_normalized_inverse, test_size=0.20, random_state=41)

# Convert data to PyTorch tensors
X_train_tensor_inverse = torch.tensor(X_train_inverse, dtype=torch.float32)
y_train_tensor_inverse = torch.tensor(y_train_inverse, dtype=torch.float32)
X_test_tensor_inverse = torch.tensor(X_test_inverse, dtype=torch.float32)
y_test_tensor_inverse = torch.tensor(y_test_inverse, dtype=torch.float32)

save_to_file = True
# Define the neural network architecture
class InverseNeuralNetwork(nn.Module):
    def __init__(self):
        super(InverseNeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(24, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, 16)
        self.fc5 = nn.Linear(16, 4)

    def forward(self, x):
        # Apply the layers with ReLU activations
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return self.fc5(x)  # No activation on the output layer

# Train the model
train_dataset_inverse = TensorDataset(X_train_tensor_inverse, y_train_tensor_inverse)
train_loader_inverse = DataLoader(train_dataset_inverse, batch_size=32, shuffle=True)
model_inverse = InverseNeuralNetwork()
criterion = nn.MSELoss()
optimizer = optim.Adam(model_inverse.parameters(), lr=0.001)
num_epochs = 1000
for epoch in range(num_epochs):
    model_inverse.train()
    for inputs, targets in train_loader_inverse:
        optimizer.zero_grad()
        outputs = model_inverse(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

# Evaluate the model
model_inverse.eval()
with torch.no_grad():
    predictions_inverse = model_inverse(X_test_tensor_inverse)
predictions_test_array = predictions_inverse.numpy()
predicted_concentrations_test = scaler_y_inverse.inverse_transform(predictions_test_array)
actual_concentrations_test = scaler_y_inverse.inverse_transform(y_test_inverse)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

dyes = ['ATTO425', 'FAM', 'ATTO550', 'Cy5']
# Define the path to the outputs directory
outputs_dir = os.path.join('outputs')
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)
writer_predictions = pd.ExcelWriter(outputs_dir, 'NN-scalability-four-fluorophore-predictions.xlsx', engine='openpyxl')
writer_mae_mse = pd.ExcelWriter(outputs_dir, 'NN-scalability-four-fluorophore-concentrationWise_MAE&MSE.xlsx', engine='openpyxl')
writer_residuals = pd.ExcelWriter(outputs_dir, 'NN-scalability-four-fluorophore-histogramOfResiduals.xlsx', engine='openpyxl')

for i, dye in enumerate(dyes):
    actual_values = actual_concentrations_test[:, i]
    predicted_values = predicted_concentrations_test[:, i]
    # Creating a DataFrame for easier grouping
    df = pd.DataFrame({
        'Actual': actual_values,
        'Predicted': predicted_values
    })
    # Grouping by each unique concentration value
    grouped = df.groupby('Actual')

    # Calculating MAE and MSE for each group and storing them along with the concentration value
    mae_per_concentration = grouped.apply(lambda g: mean_absolute_error(g['Actual'], g['Predicted']))
    mse_per_concentration = grouped.apply(lambda g: mean_squared_error(g['Actual'], g['Predicted']))

    # Creating a DataFrame to store results
    results_df = pd.DataFrame({
        'Concentration': mae_per_concentration.index,
        'MAE': mae_per_concentration.values,
        'MSE': mse_per_concentration.values
    })

    results_df.to_excel(writer_mae_mse, sheet_name=f'{dye}', index=False)

    MAE = mean_absolute_error(actual_values, predicted_values)
    MSE = mean_squared_error(actual_values, predicted_values)
    R2 = r2_score(actual_values, predicted_values)

    print(f"For dye {dye}:")
    print(f"Mean Absolute Error (MAE): {MAE}")
    print(f"Mean Squared Error (MSE): {MSE}")
    print(f"R-squared (R2): {R2}\n")

# Save the trained model
import torch
import pickle

# Save the trained model parameters
models_dir = os.path.join('models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

model_path = os.path.join(models_dir, 'NN-Scalability-four-fluorophore-trained-model.pth' )
torch.save(model_inverse.state_dict(), model_path)

# Save the scalers
x_scaler_path = os.path.join(models_dir, 'NN-Scalability-four-fluorophore-scaler-x-inverse.pkl' )
with open(x_scaler_path, 'wb') as file:
    pickle.dump(scaler_X_inverse, file)

y_scaler_path = os.path.join(models_dir, 'NN-Scalability-four-fluorophore-scaler-y-inverse.pkl' )
with open(y_scaler_path, 'wb') as file:
    pickle.dump(scaler_y_inverse, file)

# Visualize the results
import matplotlib.pyplot as plt
import numpy as np

dyes = ['ATTO425', 'FAM', 'ATTO550', 'Cy5']

# Loop through each dye to create the visualizations
for i, dye in enumerate(dyes):
    actual_values = actual_concentrations_test[:, i]
    predicted_values = predicted_concentrations_test[:, i]
    residuals = actual_values - predicted_values

    # 1. Scatter Plot of Actual vs. Predicted Values
    plt.figure(figsize=(4, 2))
    plt.scatter(actual_values, predicted_values, color='blue', alpha=0.6)
    plt.plot([min(actual_values), max(actual_values)],
             [min(actual_values), max(actual_values)], color='red')
    plt.xlabel(f'Actual', fontsize=9)
    plt.ylabel(f'Predicted', fontsize=9)
    plt.title(f'Actual vs Predicted Concentration for {dye}', fontsize=9)
    plt.grid(True)
    plt.tight_layout()
    outputs_dir = os.path.join('outputs')
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    prediction_fig_path = os.path.join(outputs_dir, f'NN-scalability-four-fluorophore-prediction-for-{dye}.svg')
    plt.savefig(prediction_fig_path, dpi=300)
    plt.show()

    # 2. Histogram of Residuals
    plt.figure(figsize=(4, 2))
    count, bin_edges = np.histogram(residuals, bins=30)
    plt.hist(residuals, bins=30, color='blue', alpha=0.7)
    plt.xlabel(f'Residual Value for {dye}', fontsize=9)
    plt.ylabel('Frequency', fontsize=9)
    plt.title(f'Histogram of Residuals for {dye}', fontsize=9)
    plt.grid(False)
    plt.tight_layout()
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)
    residuals_fig_path = os.path.join(outputs_dir, f'NN-scalability-four-fluorophore-histogramOfResiduals-for-{dye}.svg')
    plt.savefig(residuals_fig_path, dpi=300)
    plt.show()

    # Save to DataFrame
    df = pd.DataFrame({
        'Actual Values': actual_values,
        'Predicted Values': predicted_values,
        'Residuals': residuals
    })

    # Histogram Data
    histogram_data = pd.DataFrame({'BinEdges': bin_edges[:-1], 'Count': count})

    # Save predictions in DataFrame and residuals in histogram_data to Excel
    df.to_excel(writer_predictions, sheet_name=f'{dye}', index=False)
    histogram_data.to_excel(writer_residuals, sheet_name=f'{dye}', index=False)

writer_predictions.close()
writer_mae_mse.close()
writer_residuals.close()
