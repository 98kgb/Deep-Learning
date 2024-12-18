# -*- coding: utf-8 -*-
"""
This code is for analysis and visualization of the whole pipeline.
"""

import os
import sys
# Add the current file path to the system path
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path)
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#%% Keyword distribution

event_words = np.load(f"{dir_path}\\embedded_dataset\\event_words.npy")
event_counts = np.load(f"{dir_path}\\embedded_dataset\\event_counts.npy")

trim_words = np.load(f"{dir_path}\\embedded_dataset\\trim_words.npy")
trim_counts = np.load(f"{dir_path}\\embedded_dataset\\trim_counts.npy")

# Create subplots
fig, col = plt.subplots(1, 2, figsize=(22, 10))

# Plot for event type tweets
col[0].bar(event_words[:20], event_counts[:20], color='blue', edgecolor='black', alpha=0.7)
col[0].tick_params(labelsize=14)
col[0].set_ylabel('Number of words', fontsize=28)
col[0].set_xlabel('Words', fontsize=28)
col[0].set_title('Top 20 frequent words in event type tweets', fontsize=20)
col[0].set_xticklabels(event_words[:20], rotation=45, ha='right')
col[0].tick_params(axis='both', labelsize=24)  # Adjust font size for x-axis labels

# Plot for trimmed words
col[1].bar(trim_words[:20], trim_counts[:20], color='red', edgecolor='black', alpha=0.7)
col[1].tick_params(labelsize=14)
col[1].set_ylabel('Number of words', fontsize=28)
col[1].set_xlabel('Words', fontsize=28)
col[1].set_title('Top 20 frequent words after removing countries', fontsize=20)
col[1].set_xticklabels(trim_words[:20], rotation=45, ha='right')
col[1].tick_params(axis='both', labelsize=24)  # Adjust font size

# Adjust layout
plt.tight_layout()
plt.show()

#%% Analysis effect of weight

# Data load
x0 = np.load(dir_path + '\\embedded_dataset\\embedded_X_factor_0.npy')
x1 = np.load(dir_path + '\\embedded_dataset\\embedded_X_factor_2.npy')
x2 = np.load(dir_path + '\\embedded_dataset\\embedded_X_factor_5.npy')
x3 = np.load(dir_path + '\\embedded_dataset\\embedded_X_factor_10.npy')
x4 = np.load(dir_path + '\\embedded_dataset\\embedded_X_factor_15.npy')

# Load labels and reshape them
y = np.load(dir_path + '\\embedded_dataset\\embedded_Y.npy').reshape(len(x0),1)

# Add labels to each dataset
x0 = np.hstack([x0,y])
x1 = np.hstack([x1,y])
x2 = np.hstack([x2,y])
x3 = np.hstack([x3,y])
x4 = np.hstack([x4,y])

# Create column number
columns = [f'Column {ii}' if ii < 200 else 'Label' for ii in range(201)]

# Convert datasets to DataFrames
df0 = pd.DataFrame(x0, columns=columns)
df1 = pd.DataFrame(x1, columns=columns)
df2 = pd.DataFrame(x2, columns=columns)
df3 = pd.DataFrame(x3, columns=columns)
df4 = pd.DataFrame(x4, columns=columns)

# Compute correlation matrices
corr0 = df0.corr()  
corr1 = df1.corr()
corr2 = df2.corr()
corr3 = df3.corr()
corr4 = df4.corr()

# Extract correlations with the label column
label_corr0 = corr0['Label'].drop('Label')
label_corr1 = corr1['Label'].drop('Label')
label_corr2 = corr2['Label'].drop('Label')
label_corr3 = corr3['Label'].drop('Label')
label_corr4 = corr4['Label'].drop('Label')

# Analyze correlation statistics
summary = pd.DataFrame({
    'Dataset': ['No weight', 'Weight factor 2', 'Weight factor 5', 'Weight factor 10', 'Weight factor 15'],
    'Mean Correlation': [label_corr0.abs().mean(), label_corr1.abs().mean(), label_corr2.abs().mean(), label_corr3.abs().mean(), label_corr4.abs().mean()],
    'Median Correlation': [label_corr0.abs().median(), label_corr1.abs().median(), label_corr2.abs().median(), label_corr3.abs().median(), label_corr4.abs().median()],
    'Max Correlation': [label_corr0.abs().max(), label_corr1.abs().max(), label_corr2.abs().max(), label_corr3.abs().max(), label_corr4.abs().max()],
    'Min Correlation': [label_corr0.abs().min(), label_corr1.abs().min(), label_corr2.abs().min(), label_corr3.abs().min(), label_corr4.abs().min()]
})

# Plot histogram to compare correlation distributions
plt.figure(figsize=(8, 5))
sns.histplot(label_corr0.abs(), kde=True, color='blue', label='No weight', bins=20)
sns.histplot(label_corr1.abs(), kde=True, color='orange', label='Weight factor: 2', bins=20)
sns.histplot(label_corr2.abs(), kde=True, color='green', label='Weight factor: 5', bins=20)
sns.histplot(label_corr3.abs(), kde=True, color='red', label='Weight factor: 10', bins=20)
sns.histplot(label_corr4.abs(), kde=True, color='purple', label='Weight factor: 15', bins=20)
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
X1 = np.load(dir_path + f'\\embedded_dataset\\embedded_X_factor_{factor}.npy')
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


#%% DNN hyperparameter optimization

# Candidates for optimization
trial_num = 3

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
        # Break loop if we dont have result for specific optimzation result
        if not os.path.isdir(file_path):
            print(f"There is no optimization result for trial {trial_num} with {candidate[0]}")
            break
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

# Break loop if we dont have result for specific optimzation result otherwise show plot
if not os.path.isdir(dir_path + f"\\model\\optimization\\trial_{trial_num}\\default_setting.csv"):
    print(f"There is no optimization result for trial {trial_num}")
else:
    default = pd.read_csv(dir_path + f"\\model\\optimization\\trial_{trial_num}\\default_setting.csv")
    default_str = default['structure'].values[0]
    default_Norm = default['Normalization'].values[0]
    default_batch = default['Batch_size'].values[0]
    default_lr = default['learning rate'].values[0]
    default_drop = default['Drop out'].values[0]
    default_optim = default['Optim'].values[0]
    
    # Unified legend outside the grid
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=16, frameon=True)
    plt.suptitle(f'Trial: {trial_num}  structure: {default_str}  Norm: {default_Norm}  Batch: {default_batch}  Optim: {default_optim}  lr: {default_lr}  drop: {default_drop}', fontsize = 20)
    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, -0.1, 1, 1])  # Leave space at the bottom for the legend
    plt.show()


#%% Acc w.r.t numer of fold and hyperparameter
# Candidates for optimization
trial_num = 1
candi1 = ['norm', "min-max", 'mean-std', "MaxAbs"]  # Normalization techniques
candi2 = ['optim', 'Adam', 'SGD', 'RMSprop', 'Adagrad']  # Optimizers
candi3 = ['batch', 8, 16, 32, 64]  # Batch sizes
candi4 = ['structure', [1024, 512, 256, 128], [512, 256, 128], [512, 256], [512]]  # DNN structures
candi5 = ['dropout', 0.1, 0.2, 0.3, 0.4, 0.5]
candi6 = ['lr',1e-5, 1e-4, 1e-3, 1e-2] # Learning rate

candidates = [candi1, candi2, candi3, candi4, candi5, candi6]

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
        # Break loop if we dont have result for specific optimzation result
        if not os.path.isdir(file_path):
            print(f"There is no optimization result for trial {trial_num} with {candidate[0]}")
            break
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
#%% model weight distribution

def get_parameters(mdl):
    weights = torch.Tensor().to(device)
    for param_group in list(mdl.parameters()):
        weights = torch.cat((param_group.view(-1), weights))
    ws = weights.detach().cpu().numpy()
    return ws

model_name = ['Overfitted', 'EarlyStopped', 'Scheduler']
_, ax = plt.subplots(1,3, figsize = (15,5))

for ii in range(3):
    model = torch.load(f'{dir_path}\\model\\{model_name[ii]}.pt')
    w_scheduler = get_parameters(model)
    ax[ii].hist(w_scheduler.reshape(-1), range=(-.5, .5), bins=1024, color ='r', alpha=0.6)
    ax[ii].set_xlabel('Weight value', fontsize = 16)
    ax[ii].set_ylabel('Frequency', fontsize = 16)
    ax[ii].set_title(f'{model_name[ii]}', fontsize = 18)
    ax[ii].tick_params(axis='both', labelsize=16)
    
plt.tight_layout()
plt.show()

#%% Confusion matrix

weight = True
factor = 5
enbedded_method = 1 # with keyword stored in embedded_1_factor_.npy

include_count = True

norm = "min-max"

if weight:
    if include_count:
        X_count = np.load(dir_path + '\\embedded_dataset\\embedded_X_count_ratio.npy')
        X = np.load(dir_path + f'\\embedded_dataset\\embedded_X_factor_{factor}.npy')
        X = np.hstack([X,X_count])
        
    else:
        X = np.load(dir_path + f'\\embedded_dataset\\embedded_X_factor_{factor}.npy')
        
    keywords = np.load(dir_path + '\\embedded_dataset\\embedded_keyword.npy')
    
else:
    X = np.load(dir_path + '\\embedded_dataset\\embedded_X_factor_0.npy')

y = np.load(dir_path + '\\embedded_dataset\\embedded_Y.npy')

if norm == 'min-max':
    X_min = X.min(axis = 0)
    X_max = X.max(axis = 0)
    print("min-max normatlization completed!\n")
    X = (X-X_min) / (X_max-X_min)

elif norm == 'mean-std':
    X_mean = X.mean(axis = 0)
    X_std = X.std(axis = 0)
    print("mean-std normalization!\n")
    X = (X-X_mean) / (X_std)

k = 5
kfold = KFold(n_splits=k, shuffle=True, random_state=2)

# K-Fold Validation
fold_results = []

for fold, (train_index, val_index) in enumerate(kfold.split(X)):
    if fold == 2: # 2 shows highest accuracy in test dataset
        print(f"\nOptimization for {fold+1}/{k}\n")
        # Train and Validation IDs
        model_name = 'best'
        # model_name = 'Overfitted'
        # model_name = 'EarlyStopped'
        # model_name = 'Scheduler'
        
        X_train, X_test = X[train_index], X[val_index]
        y_train, y_test = y[train_index], y[val_index]
        
        X_train, X_test = torch.from_numpy(X_train).float().to(device), torch.from_numpy(X_test).float().to(device)
        Y_train, Y_test = torch.from_numpy(y_train).float().to(device), torch.from_numpy(y_test).float().to(device)# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = torch.load(f"{dir_path}/model/{model_name}.pt")
        
        train_hat = model(X_train).detach().cpu().numpy()
        test_hat = model(X_test).detach().cpu().numpy()
        
        train_hat = (train_hat >= 0.5).astype(int)
        test_hat = (test_hat >= 0.5).astype(int)
        
        # Compute confusion matrix
        train_cm = confusion_matrix(Y_train.detach().cpu().numpy(), train_hat, labels=[0, 1])
        test_cm = confusion_matrix(Y_test.detach().cpu().numpy(), test_hat, labels=[0, 1])
        
                
        train_acc = accuracy_score(Y_train.detach().cpu().numpy(), train_hat)
        test_acc = accuracy_score(Y_test.detach().cpu().numpy(), test_hat)

        # Create a heatmap
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))  # 1 row, 2 columns

        # Train Confusion Matrix
        title_size = 20
        label_size = 20
        sns.heatmap(train_cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1], ax=axes[0])
        axes[0].set_title(f"Train Confusion Matrix (Acc: {train_acc:.2f})", fontsize=title_size)
        axes[0].set_xlabel("Predicted Label", fontsize = label_size)
        axes[0].set_ylabel("True Label", fontsize = label_size)
        
        # Test Confusion Matrix
        sns.heatmap(test_cm, annot=True, fmt="d", cmap="Blues", xticklabels=[0, 1], yticklabels=[0, 1], ax=axes[1])
        axes[1].set_title(f"Test Confusion Matrix (Acc: {test_acc:.2f})", fontsize=title_size)
        axes[1].set_xlabel("Predicted Label", fontsize = label_size)
        axes[1].set_ylabel("True Label", fontsize = label_size)
        
        # Adjust layout
        plt.tight_layout()
        plt.show()
    
    
