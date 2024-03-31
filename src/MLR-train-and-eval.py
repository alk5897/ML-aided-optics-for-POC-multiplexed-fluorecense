#%% Start Clean

print("\014")
# %reset -sf

#%% Startup

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#
import os
from IPython import get_ipython
import pathlib
import openpyxl
import re
from scipy.stats import linregress
import scipy


#%% Import Data
# Read the Excel sheet into a Pandas DataFrame
dataset_path = os.path.join('datasets', 'three-fluorophore-calibration-data-for-model-eval.xlsx')
data = pd.read_excel(dataset_path,engine = "openpyxl")

xConc = data.iloc[:,0:3]
RFU = data.iloc[:,list(range(3,27))]
# reorder = list(range(0, 24, 3)) + list(range(1, 24, 3)) + list(range(2, 24, 3))
# RFU = RFU.iloc[:, reorder]

# for i in range(0,len(RFU.iloc[:,1])):
#     RFU.iloc[i,range(0,8)] = RFU.iloc[i,range(0,8)]/RFU.iloc[i,2]
#     RFU.iloc[i,range(8,16)] = RFU.iloc[i,range(8,16)]/RFU.iloc[i,11]
#     RFU.iloc[i,range(16,24)] = RFU.iloc[i,range(16,24)]/RFU.iloc[i,22]

# yRFU_466x = data.iloc[:,list(range(3,25,3))]
# yRFU_515x = data.iloc[:,list(range(4,26,3))]
# yRFU_630x = data.iloc[:,list(range(5,27,3))]


#%% Set training data 4:2

idx_tr = np.array([range(0,4,1)])
idx_te = np.array([range(4,7,1)])

train = [idx_tr]
for i in range(0,874,7):
    train = np.append(train,[idx_tr+i])
  
train_xConc = xConc.iloc[train]    
train_RFU = RFU.iloc[train]
# train_466x = yRFU_466x.iloc[train]
# train_515x = yRFU_515x.iloc[train]
# train_630x = yRFU_630x.iloc[train]
#%% Set test data     
  
test = [idx_te]
for j in list(range(0,874,7)):
    test = np.append(test,[idx_te+j])
    
test_xConc = xConc.iloc[test]  
test_RFU = RFU.iloc[test]  
# test_466x = yRFU_466x.iloc[test]
# test_515x = yRFU_515x.iloc[test]
# test_630x = yRFU_630x.iloc[test]

#%% Solve Multivariate linear regression for weights

mlr_all = np.linalg.lstsq(train_RFU,train_xConc,rcond=-1)


# mlr_466x = np.linalg.lstsq(train_466x,train_xConc,rcond=None)

# mlr_515x = np.linalg.lstsq(train_515x,train_xConc,rcond=None)

# mlr_630x = np.linalg.lstsq(train_630x,train_xConc,rcond=None)

#%% Test model accuracy

tested = np.matmul(test_RFU,mlr_all[0])

for_export_MLR = mlr_all[0]
# test_466 = np.matmul(test_466x,mlr_466x[0])
# test_515 = np.matmul(test_515x,mlr_515x[0])
# test_630 = np.matmul(test_630x,mlr_630x[0])

# pAcc_466 = (1-np.divide(abs(np.subtract(test_xConc,np.array(test_466))),test_xConc))*100
# pAcc_515 = (1-np.divide(abs(np.subtract(test_xConc,np.array(test_515))),test_xConc))*100
# pAcc_630 = (1-np.divide(abs(np.subtract(test_xConc,np.array(test_630))),np.array(test_xConc)))*100

#%% plot model accuracy
x = np.linspace(0,20,100)

# 466nm Excitation
# FAM
plt.figure(figsize=(3,3))

# Set Font Size
plt.rcParams['font.size'] = 9

# Set Figure DPI
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

# Find linear regress and R
slope, intercept, r_value, p_value, std_err = linregress(test_xConc.FAM,tested.iloc[:,0])

plt.scatter(test_xConc.FAM,tested.iloc[:,0],c ="blue",marker='o',alpha=0.1)
plt.plot(x,slope*x+intercept,c="blue",linestyle = '--')
plt.text(0.35,0.8,'R\u00b2 ='+ str(round(r_value**2,3)))



plt.xlabel("Known [C] µM")
plt.ylabel("Model Output [C] µM")
plt.xlim(-0.1, 1.1)
plt.ylim(-0.75, 1.5)
plt.title('FAM')

MAE = sum(abs(test_xConc.FAM-tested.iloc[:,0]))/len(test_xConc.FAM)
MSE = sum((test_xConc.FAM-tested.iloc[:,0])**2)/len(test_xConc.FAM)
for_export_FAM = np.concatenate((test_xConc.FAM[:,np.newaxis],tested.iloc[:,0][:,np.newaxis]),axis=1)
FAM_stat = np.array((r_value**2,MAE,MSE))
#plt.tight_layout()
# Show the plot

fmat = "svg"
outputs_folder = pathlib.Path('outputs')
fn = outputs_folder+"/"+"MLR-FAM."+fmat
plt.savefig(fn, dpi = 'figure',format=fmat, bbox_inches="tight")
print('Figure Save Confirmed')
#

# ROX
plt.figure(figsize=(3,3))

# Set Font Size
plt.rcParams['font.size'] = 9

# Set Figure DPI
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600

# Find linear regress and R
slope, intercept, r_value, p_value, std_err = linregress(test_xConc.ATTO550,tested.iloc[:,1])

plt.scatter(test_xConc.ATTO550,tested.iloc[:,1],c ="green",marker='o',alpha=0.1)
plt.plot(x,slope*x+intercept,c="green",linestyle = '--')
plt.text(0.35,0.8,'R\u00b2 ='+ str(round(r_value**2,3)))
plt.xlabel("Known [C] µM")
plt.ylabel("Model Output [C] µM")
plt.xlim(-0.1, 1.1)
plt.ylim(-0.75, 1.5)
plt.title('ATTO-550')


MAE = sum(abs(test_xConc.ATTO550-tested.iloc[:,1]))/len(test_xConc.ATTO550)
MSE = sum((test_xConc.ATTO550-tested.iloc[:,1])**2)/len(test_xConc.ATTO550)

ATTO_stat = np.array((r_value**2,MAE,MSE))
for_export_Atto550 = np.concatenate((test_xConc.ATTO550[:,np.newaxis],tested.iloc[:,1][:,np.newaxis]),axis=1)
#plt.tight_layout()
# Show the plot

fmat = "svg"
fn = outputs_folder+"/"+"MLR-ATTO550."+fmat
plt.savefig(fn, dpi = 'figure',format=fmat, bbox_inches="tight")
print('Figure Save Confirmed')
#

# Cy5
plt.figure(figsize=(3,3))

# Set Font Size
plt.rcParams['font.size'] = 9

# Set Figure DPI
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
# Find linear regress and R
slope, intercept, r_value, p_value, std_err = linregress(test_xConc.Cy5,tested.iloc[:,2])

plt.scatter(test_xConc.Cy5,tested.iloc[:,2],c ="orange",marker='o',alpha=0.1)
plt.plot(x,slope*x+intercept,c="orange",linestyle = '--')
plt.text(0.35,0.8,'R\u00b2 ='+ str(round(r_value**2,3)))
plt.xlabel("Known [C] µM")
plt.ylabel("Model Output [C] µM")
plt.xlim(-0.1, 1.1)
plt.ylim(-0.75, 1.5)
plt.title('Cy5')


MAE = sum(abs(test_xConc.Cy5-tested.iloc[:,2]))/len(test_xConc.Cy5)
MSE = sum((test_xConc.Cy5-tested.iloc[:,2])**2)/len(test_xConc.Cy5)

Cy5_stat = np.array((r_value**2,MAE,MSE))

for_export_Cy5 = np.concatenate((test_xConc.Cy5[:,np.newaxis],tested.iloc[:,2][:,np.newaxis]),axis=1)
#plt.tight_layout()
# Show the plot

fmat = "svg"
fn = outputs_folder+"/"+"MLR-Cy5."+fmat
plt.savefig(fn, dpi = 'figure',format=fmat, bbox_inches="tight")
print('Figure Save Confirmed')
#

