# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 14:36:12 2024

@author: Gibaek
"""
from sklearn.model_selection import KFold

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


import seaborn as sns
import os
import sys

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path)

#%% Analysis effect of weight

# Data load
x0 = np.load(dir_path + '\\embedded_dataset\\embedded_X_factor_0.npy')
x1 = np.load(dir_path + '\\embedded_dataset\\embedded_1_X_factor_2.npy')
x2 = np.load(dir_path + '\\embedded_dataset\\embedded_1_X_factor_5.npy')
x3 = np.load(dir_path + '\\embedded_dataset\\embedded_1_X_factor_10.npy')

# Load labels and reshape them
y = np.load(dir_path + '\\embedded_dataset\\embedded_Y.npy').reshape(len(x0),1)

# Add labels to each dataset
x0 = np.hstack([x0,y])
x1 = np.hstack([x1,y])
x2 = np.hstack([x2,y])
x3 = np.hstack([x3,y])

# Create column number
columns = [f'Column {ii}' if ii < 200 else 'Label' for ii in range(201)]

# Convert datasets to DataFrames
df0 = pd.DataFrame(x0, columns=columns)
df1 = pd.DataFrame(x1, columns=columns)
df2 = pd.DataFrame(x2, columns=columns)
df3 = pd.DataFrame(x3, columns=columns)

# Compute correlation matrices
corr0 = df0.corr()  
corr1 = df1.corr()
corr2 = df2.corr()
corr3 = df3.corr()

# Extract correlations with the label column
label_corr0 = corr0['Label'].drop('Label')
label_corr1 = corr1['Label'].drop('Label')
label_corr2 = corr2['Label'].drop('Label')
label_corr3 = corr3['Label'].drop('Label')

# Analyze correlation statistics
summary = pd.DataFrame({
    'Dataset': ['No weight', 'Weight factor 2', 'Weight factor 5', 'Weight factor 10'],
    'Mean Correlation': [label_corr0.abs().mean(), label_corr1.abs().mean(), label_corr2.abs().mean(), label_corr3.abs().mean()],
    'Median Correlation': [label_corr0.abs().median(), label_corr1.abs().median(), label_corr2.abs().median(), label_corr3.abs().median()],
    'Max Correlation': [label_corr0.abs().max(), label_corr1.abs().max(), label_corr2.abs().max(), label_corr3.abs().max()],
    'Min Correlation': [label_corr0.abs().min(), label_corr1.abs().min(), label_corr2.abs().min(), label_corr3.abs().min()]
})
#%%
# Plot histogram to compare correlation distributions
plt.figure(figsize=(8, 5))
sns.histplot(label_corr0.abs(), kde=True, color='blue', label='No weight', bins=20)
sns.histplot(label_corr1.abs(), kde=True, color='orange', label='Weight factor: 2', bins=20)
sns.histplot(label_corr2.abs(), kde=True, color='green', label='Weight factor: 5', bins=20)
sns.histplot(label_corr3.abs(), kde=True, color='red', label='Weight factor: 10', bins=20)
plt.title('Distribution of Correlations with Label', fontsize=16)
plt.xlabel('Absolute Correlation Coefficient', fontsize=16)
plt.ylabel('Frequency', fontsize=13)
plt.xticks(fontsize=13)
plt.yticks(fontsize=13)

# Add a legend with a larger font size
plt.legend(fontsize=12, title_fontsize=14)
plt.show()

#%% Analysis correlation of tweet ration with label

enbedded_method = 1 # with keyword stored in embedded_1_factor_.npy
include_count = True
weight = True
factor = 5
X1 = np.load(dir_path + f'\\embedded_dataset\\embedded_{enbedded_method}_X_factor_{factor}.npy')
X2 = np.load(dir_path + '\\embedded_dataset\\embedded_X_count_ratio.npy')
X = np.hstack([X1,X2])
y = np.load(dir_path + '\\embedded_dataset\\embedded_Y.npy')

data = np.hstack([X,y.reshape(y.shape[0],1)])
columns = [f'Vector {ii}' if ii < 200 else 'Tweet Ratio' if ii == 200 else 'Label' for ii in range(202)]
    

df = pd.DataFrame(data, columns=columns)

correlation_matrix = df.corr()  
label_correlation = correlation_matrix['Label'].drop('Label')

top_10_correlation = label_correlation.abs().sort_values(ascending=False).head(10)
print("Top 10 Correlations with Label:")
print(top_10_correlation)

# Bar graph of correlation coefficient
colors = ['dodgerblue' if i == 9 else 'lightblue' for i in range(10)]  # First bar darker

plt.figure(figsize=(8, 5))
top_10_correlation.sort_values().plot(kind='barh', color=colors)
plt.title('Top 10 Correlations with Label', fontsize = 16)
plt.xlabel('Correlation Coefficient', fontsize = 13)
plt.yticks(fontsize=13)
plt.xticks(fontsize=13)

plt.show()

#%% Train test split methodology (match ID)

match_ID = np.load(dir_path + '\\embedded_dataset\\embedded_matchID.npy')

plt.hist(match_ID, range=(-1, 20), bins=50)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Mach ID")
plt.show()

#%% DNN hyperparameter optimization

# Candidates for optimization
trial_num = 1

# Candidates for optimization
candi1 = ['norm', "min-max", 'mean-std', "MaxAbs"]
candi2 = ['optim', 'Adam', 'SGD', 'RMSprop', 'Adagrad']
candi3 = ['batch', 8, 16, 32, 64]
candi4 = ['structure', [1024, 512, 256, 128], [512, 256, 128], [512, 256], [512]]
candi5 = ['dropout', 0, 0.1, 0.2, 0.3, 0.4, 0.5]
candi6 = ['lr', 1e-5, 1e-4, 1e-3, 1e-2]

# All candidates
candidates = [candi1, candi2, candi3, candi4, candi5, candi6]

# Subplot grid
fig, ax = plt.subplots(2, 3, figsize=(15, 10))  # Adjust figure size as needed

# Collect handles and labels for unified legend
handles, labels = [], []

# Loop through candidates and subplots
for idx, candidate in enumerate(candidates):
    row, col = idx // 3, idx % 3
        
    # Initialize result arrays for mean, std, min, max
    result_mean = np.zeros([len(candidate) - 1, 2])  # Shape: [n_candidates, 2]
    result_std = np.zeros([len(candidate) - 1, 2])
    result_min = np.zeros([len(candidate) - 1, 2])
    result_max = np.zeros([len(candidate) - 1, 2])

    # Loop through candidate values
    for ii in range(1, len(candidate)):
        # Load data for each candidate from CSV
        file_path = f'{dir_path}\\model\\optimization\\info\\trial_{trial_num}\\Optim_{candidate[0]}_{candidate[ii]}.csv'
        temp = np.genfromtxt(file_path, delimiter = ',')
        
            
        # Calculate statistics
        result_mean[ii - 1, :] = np.mean(temp, axis=0)
        result_std[ii - 1, :] = np.std(temp, axis=0)
        result_min[ii - 1, :] = np.min(temp, axis=0)
        result_max[ii - 1, :] = np.max(temp, axis=0)
        candidate[ii] = str(candidate[ii])  # Convert to string for x-ticks
    
    if idx == 3:
        for jj in range(len(candidate)):
            if jj == 0:
                candidate[jj] = ' # of hidden Layer'
            else:
                candidate[jj] = f'{5-jj}'
            

    # Min-Max shading
    h1 = ax[row, col].fill_between(
        candidate[1:], result_min[:, 0], result_max[:, 0], 
        color='skyblue', alpha=0.3, label='Min-Max Range (Loss)'
    )
    h2 = ax[row, col].fill_between(
        candidate[1:], result_min[:, 1], result_max[:, 1], 
        color='red', alpha=0.3, label='Min-Max Range (Accuracy)'
    )

    # Mean with Std Dev
    h3 = ax[row, col].errorbar(
        candidate[1:], result_mean[:, 0], yerr=result_std[:, 0],
        fmt='s-', capsize=5, color='blue', label='Validation Loss (Mean ± Std)', 
        linewidth=1
    )
    h4 = ax[row, col].errorbar(
        candidate[1:], result_mean[:, 1], yerr=result_std[:, 1],
        fmt='D-', capsize=5, color='r', label='Validation Accuracy (Mean ± Std)', 
        linewidth=1
    )

    # Collect handles and labels from the first subplot only
    if idx == 0:
        handles.extend([h1, h2, h3, h4])
        labels.extend(['Min-Max Range (Loss)', 'Min-Max Range (Accuracy)', 
                       'Validation Loss (Mean ± Std)', 'Validation Accuracy (Mean ± Std)'])

    # Title and labels
    ax[row, col].set_xlabel(f'{candidate[0]}', fontsize=16)  # Increased font size
    ax[row, col].set_ylabel('Metric Values', fontsize=16)  # Increased font size
    ax[row, col].grid(visible=True, linestyle='--', alpha=0.6)
    ax[row, col].tick_params(axis='x', rotation=45, labelsize=16)  # Increased tick label size
    ax[row, col].tick_params(axis='y', labelsize=16)  # Increased tick label size

default = pd.read_csv(dir_path + f"\\model\\optimization\\trial_{trial_num}\\default_setting.csv")
default_str = default['structure'].values[0]
default_Norm = default['Normalization'].values[0]
default_batch = default['Batch_size'].values[0]
default_lr = default['learning rate'].values[0]
default_drop = default['Drop out'].values[0]
# Unified legend outside the grid
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=16, frameon=True)
plt.suptitle(f'Trial: {trial_num}  structure: {default_str}  Norm: {default_Norm}  Batch: {default_batch}  lr: {default_lr}  drop: {default_drop}', fontsize = 20)
# Adjust layout for better spacing
plt.tight_layout(rect=[0, -0.1, 1, 1])  # Leave space at the bottom for the legend
plt.show()


#%% Acc w.r.t numer of fold and hyperparameter
# Candidates for optimization
trial_num = 4
candi1 = ['norm', "min-max", 'mean-std', "MaxAbs"]  # Normalization techniques
candi2 = ['optim', 'Adam', 'SGD', 'RMSprop', 'Adagrad']  # Optimizers
candi3 = ['batch', 8, 16, 32, 64]  # Batch sizes
candi4 = ['structure', [1024, 512, 256, 128], [512, 256, 128], [512, 256], [512]]  # DNN structures
candi5 = ['dropout', 0, 0.1, 0.2, 0.3, 0.4, 0.5] if trial_num == 4 else ['dropout', 0.1, 0.2, 0.3, 0.4, 0.5]
candi6 = ['lr',1e-5, 1e-4, 1e-3, 1e-2] # Learning rate

candidates = [candi1]

# Subplot grid
fig, ax = plt.subplots(2, 3, figsize=(15, 10))  # Adjust figure size as needed

# Loop through candidates and subplots
for idx, candidate in enumerate(candidates):
    row, col = idx // 3, idx % 3
        
    # Initialize result arrays for mean, std, min, max
    result_mean = np.zeros([len(candidate) - 1, 2])  # Shape: [n_candidates, 2]
    result_std = np.zeros([len(candidate) - 1, 2])
    result_min = np.zeros([len(candidate) - 1, 2])
    result_max = np.zeros([len(candidate) - 1, 2])

    # Loop through candidate values
    for ii in range(1, len(candidate)):
        # Load data for each candidate from CSV
        file_path = f'{dir_path}\\model\\optimization\\info\\trial_{trial_num}\\Optim_{candidate[0]}_{candidate[ii]}.csv'
        temp = np.genfromtxt(file_path, delimiter = ',')
        ax[row, col].plot(np.arange(1,6,1), temp[:,1],marker = 'o', label = f'{candidate[0]}: {candidate[ii]}')
    
    # Title and labels
    ax[row, col].set_xlabel('Number of fold', fontsize=16)  # Increased font size
    ax[row, col].set_ylabel('Accuracy', fontsize=16)  # Increased font size
    ax[row, col].grid(visible=True, linestyle='--', alpha=0.6)
    ax[row, col].tick_params(axis='x', rotation=45, labelsize=16)  # Increased tick label size
    ax[row, col].tick_params(axis='y', labelsize=16)  # Increased tick label size
    ax[row, col].legend() # Increased tick label size
    
# Unified legend outside the grid

# Adjust layout for better spacing
plt.tight_layout(rect=[0, -0.1, 1, 1])  # Leave space at the bottom for the legend
plt.show()

#%%
file_path = f'{dir_path}\\model\\optimization\\info\\Optim_{candidate[0]}_{candidate[ii]}.csv'
fold_accuracies = np.genfromtxt(file_path, delimiter=',')[:,1]

# Extract unique Match ID
match_ID = np.load(dir_path + '\\embedded_dataset\\embedded_matchID.npy')
unique_ID = np.unique(match_ID)

kfold = KFold(n_splits=5, shuffle=True, random_state=2)


# K-Fold Validation
fold_val_indices = []

for fold, (train_index, val_index) in enumerate(kfold.split(unique_ID)):
    fold_val_indices.append(val_index)
print(fold_val_indices)

idx_to_accuracy = {}
for fold, indices in enumerate(fold_val_indices):
    for idx in indices:
        idx_to_accuracy[idx] = fold_accuracies[fold]

# DataFrame generation
data = pd.DataFrame(list(idx_to_accuracy.items()), columns=["idx", "accuracy"])

# calculate the correlation
correlation = data["idx"].corr(data["accuracy"])
print(f"Correlation between idx and accuracy: {correlation:.2f}")

plt.scatter(data["idx"], data["accuracy"], alpha=0.7)
plt.xlabel("Index (idx)")
plt.ylabel("Accuracy")
plt.title("Relationship between idx and Accuracy")
plt.show()
