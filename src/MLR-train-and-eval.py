# %% Startup

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pathlib
from scipy.stats import linregress

# %% Construct the path to the dataset

dataset_path = os.path.join('datasets', 'three-fluorophore-calibration-data-for-model-eval.xlsx')

# %% Read and preprocess the data

data = pd.read_excel(dataset_path, engine='openpyxl')
xConc = data.iloc[:, 0:3]
RFU = data.iloc[:, list(range(3, 27))]

# %% Define Training Data

idx_tr = np.array([range(0, 4, 1)])
idx_te = np.array([range(4, 7, 1)])
train = [idx_tr]
for i in range(0, 874, 7):
    train = np.append(train, [idx_tr + i])

train_xConc = xConc.iloc[train]
train_RFU = RFU.iloc[train]

# %% Define Testing Data

test = [idx_te]
for j in list(range(0, 874, 7)):
    test = np.append(test, [idx_te + j])

test_xConc = xConc.iloc[test]
test_RFU = RFU.iloc[test]

# %% Train the MLR model

mlr_all = np.linalg.lstsq(train_RFU, train_xConc, rcond=-1)

# %% Test the MLR model

tested = np.matmul(test_RFU, mlr_all[0])

# %% Save the model weights
models_dir = os.path.join('models')
if not os.path.exists(models_dir):
    os.makedirs(models_dir)

weights_path = os.path.join(models_dir, 'MLR-ModelWeights.xlsx')
writer_weights = pd.ExcelWriter(weights_path, engine='openpyxl')

mlr = pd.DataFrame(mlr_all[0], columns=['FAM', 'ATTO', 'CY5'])

mlr.to_excel(writer_weights, index=False)

writer_weights.close()

# %% Visualization of Dye performance
outputs_dir = os.path.join('outputs')
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

# FAM
x = np.linspace(0, 20, 100)
plt.figure(figsize=(3, 3))
plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
slope, intercept, r_value, p_value, std_err = linregress(test_xConc.FAM, tested.iloc[:, 0])
plt.scatter(test_xConc.FAM, tested.iloc[:, 0], c="blue", marker='o', alpha=0.1)
plt.plot(x, slope * x + intercept, c="blue", linestyle='--')
plt.text(0.35, 0.8, 'R\u00b2 =' + str(round(r_value ** 2, 3)))
plt.xlabel("Known [C] µM")
plt.ylabel("Model Output [C] µM")
plt.xlim(-0.1, 1.1)
plt.ylim(-0.75, 1.5)
plt.title('FAM')
plt.tight_layout()
fmat = "svg"
FAM_prediction_fig_path = os.path.join(outputs_dir, f'MLR-prediction-for-FAM.svg')
plt.savefig(FAM_prediction_fig_path, dpi='figure', format=fmat, bbox_inches="tight")

# ATTO 550
plt.figure(figsize=(3, 3))
plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
slope, intercept, r_value, p_value, std_err = linregress(test_xConc.ATTO550, tested.iloc[:, 1])
plt.scatter(test_xConc.ATTO550, tested.iloc[:, 1], c="green", marker='o', alpha=0.1)
plt.plot(x, slope * x + intercept, c="green", linestyle='--')
plt.text(0.35, 0.8, 'R\u00b2 =' + str(round(r_value ** 2, 3)))
plt.xlabel("Known [C] µM")
plt.ylabel("Model Output [C] µM")
plt.xlim(-0.1, 1.1)
plt.ylim(-0.75, 1.5)
plt.title('ATTO-550')
plt.tight_layout()
fmat = "svg"
ATTO550_prediction_fig_path = os.path.join(outputs_dir, f'MLR-prediction-for-ATTO550.svg')
plt.savefig(ATTO550_prediction_fig_path, dpi='figure', format=fmat, bbox_inches="tight")

# Cy5
plt.figure(figsize=(3, 3))
plt.rcParams['font.size'] = 9
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
slope, intercept, r_value, p_value, std_err = linregress(test_xConc.Cy5, tested.iloc[:, 2])
plt.scatter(test_xConc.Cy5, tested.iloc[:, 2], c="orange", marker='o', alpha=0.1)
plt.plot(x, slope * x + intercept, c="orange", linestyle='--')
plt.text(0.35, 0.8, 'R\u00b2 =' + str(round(r_value ** 2, 3)))
plt.xlabel("Known [C] µM")
plt.ylabel("Model Output [C] µM")
plt.xlim(-0.1, 1.1)
plt.ylim(-0.75, 1.5)
plt.title('Cy5')
plt.tight_layout()
fmat = "svg"
Cy5_prediction_fig_path = os.path.join(outputs_dir, f'MLR-prediction-for-Cy5.svg')
plt.savefig(Cy5_prediction_fig_path, dpi='figure', format=fmat, bbox_inches="tight")

predictions_path = os.path.join(outputs_dir, 'MLR-predictions.xlsx')
writer_predictions = pd.ExcelWriter(predictions_path, engine='openpyxl')
residuals_path = os.path.join(outputs_dir, 'MLR-histogramOfResiduals.xlsx')
writer_residuals = pd.ExcelWriter(residuals_path, engine='openpyxl')

# %% Caclulate MSE, MAE, and residuals

# FAM
MAE = sum(abs(test_xConc.FAM - tested.iloc[:, 0])) / len(test_xConc.FAM)
MSE = sum((test_xConc.FAM - tested.iloc[:, 0]) ** 2) / len(test_xConc.FAM)
FAM_stat = np.array((r_value ** 2, MAE, MSE))
for_export_FAM = np.concatenate((test_xConc.FAM.to_numpy()[:, np.newaxis], tested.iloc[:, 0].to_numpy()[:, np.newaxis]),
                                axis=1)
residuals = for_export_FAM[:, 0] - for_export_FAM[:, 1]
dfFAM = pd.DataFrame({
    'Actual Values': for_export_FAM[:, 0],
    'Predicted Values': for_export_FAM[:, 1],
    'Residuals': residuals
})
dfFAM.to_excel(writer_predictions, sheet_name='FAM', index=False)

# ATTO 550
MAE = sum(abs(test_xConc.ATTO550 - tested.iloc[:, 1])) / len(test_xConc.ATTO550)
MSE = sum((test_xConc.ATTO550 - tested.iloc[:, 1]) ** 2) / len(test_xConc.ATTO550)
ATTO_stat = np.array((r_value ** 2, MAE, MSE))
for_export_ATTO550 = np.concatenate(
    (test_xConc.ATTO550.to_numpy()[:, np.newaxis], tested.iloc[:, 1].to_numpy()[:, np.newaxis]), axis=1)
residuals = for_export_ATTO550[:, 0] - for_export_ATTO550[:, 1]
dfATTO = pd.DataFrame({
    'Actual Values': for_export_FAM[:, 0],
    'Predicted Values': for_export_FAM[:, 1],
    'Residuals': residuals
})
dfATTO.to_excel(writer_predictions, sheet_name='ATTO550', index=False)

# CY5
MAE = sum(abs(test_xConc.Cy5 - tested.iloc[:, 2])) / len(test_xConc.Cy5)
MSE = sum((test_xConc.Cy5 - tested.iloc[:, 2]) ** 2) / len(test_xConc.Cy5)
Cy5_stat = np.array((r_value ** 2, MAE, MSE))
for_export_Cy5 = np.concatenate((test_xConc.Cy5.to_numpy()[:, np.newaxis], tested.iloc[:, 2].to_numpy()[:, np.newaxis]),
                                axis=1)
residuals = for_export_Cy5[:, 0] - for_export_Cy5[:, 1]
dfCY5 = pd.DataFrame({
    'Actual Values': for_export_FAM[:, 0],
    'Predicted Values': for_export_FAM[:, 1],
    'Residuals': residuals
})
dfCY5.to_excel(writer_predictions, sheet_name='CY5', index=False)

# %% Plot Residuals and Save Histograms

# FAM
residuals = for_export_FAM[:, 0] - for_export_FAM[:, 1]
plt.figure(figsize=(4, 2))
count, bin_edges = np.histogram(residuals, bins=30)
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.xlabel('Residual Value for FAM', fontsize=9)
plt.ylabel('Frequency', fontsize=9)
plt.title('Histogram of Residuals for FAM', fontsize=9)
plt.grid(False)
plt.tight_layout()
fmat = "svg"
FAM_residuals_fig_path = os.path.join(outputs_dir, f'MLR-HistogramOfResiduals-for-FAM.svg')
plt.savefig(FAM_residuals_fig_path, dpi='figure', format=fmat, bbox_inches="tight")
histogram_data = pd.DataFrame({'BinEdges': bin_edges[:-1], 'Count': count})
histogram_data.to_excel(writer_residuals, sheet_name='FAM', index=False)

# ATTO

plt.figure(figsize=(4, 2))
count, bin_edges = np.histogram(residuals, bins=30)
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.xlabel('Residual Value for ATTO550', fontsize=9)
plt.ylabel('Frequency', fontsize=9)
plt.title('Histogram of Residuals for ATTO550', fontsize=9)
plt.grid(False)
plt.tight_layout()
fmat = "svg"
ATTO550_residuals_fig_path = os.path.join(outputs_dir, f'MLR-HistogramOfResiduals-for-ATTO550.svg')
plt.savefig(ATTO550_residuals_fig_path, dpi='figure', format=fmat, bbox_inches="tight")
histogram_data = pd.DataFrame({'BinEdges': bin_edges[:-1], 'Count': count})
histogram_data.to_excel(writer_residuals, sheet_name='ATTO550', index=False)

# CY5

plt.figure(figsize=(4, 2))
count, bin_edges = np.histogram(residuals, bins=30)
plt.hist(residuals, bins=30, color='blue', alpha=0.7)
plt.xlabel('Residual Value for CY5', fontsize=9)
plt.ylabel('Frequency', fontsize=9)
plt.title('Histogram of Residuals for CY5', fontsize=9)
plt.grid(False)
plt.tight_layout()
fmat = "svg"
Cy5_residuals_fig_path = os.path.join(outputs_dir, f'MLR-HistogramOfResiduals-for-Cy5.svg')
plt.savefig(Cy5_residuals_fig_path, dpi='figure', format=fmat, bbox_inches="tight")
histogram_data = pd.DataFrame({'BinEdges': bin_edges[:-1], 'Count': count})
histogram_data.to_excel(writer_residuals, sheet_name='CY5', index=False)

writer_predictions.close()
writer_residuals.close()
