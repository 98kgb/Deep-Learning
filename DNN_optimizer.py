# -*- coding: utf-8 -*-
"""
This code is for optimizing the structure and hyperparameters of a Deep Neural Network (DNN).

It performs hyperparameter tuning using K-fold cross-validation.
"""
import os
import sys
# Add the current directory to the system path for importing custom modules
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path)
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from Deep_Learning_class import DNN_structure

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device\n")

#%% Dataset preprocessing and hyperparameter optimization setup

# K-fold cross-validation and hyperparameter settings
k = 5 # Number of folds for cross-validation
weight_factor = 10 # Embedding factor for weighting
save = True # Whether to save models and results
Early = True # Whether to apply early stopping
patience = 300 # Number of epochs to wait for improvement before early stopping
epoch = 300 # Maximum number of training epochs
include_count = True # Whether to include additional count-based features in the dataset

# Define hyperparameter candidates for optimization
candi1 = ['norm', "min-max", 'mean-std', "MaxAbs"] # normalization
candi2 = ['optim', 'Adam', 'SGD', 'RMSprop', 'Adagrad'] # optimizer
candi3 = ['batch', 8, 16, 32, 64] # batch size
candi4 = ['structure', [1024, 512, 256, 128], [512, 256, 128], [512, 256], [512]] # DNN structure
candi5 = ['dropout',0, 0.1, 0.2, 0.3, 0.4, 0.5] # dropout rate
candi6 = ['lr',1e-5, 1e-4, 1e-3, 1e-2] # Learning rate

# Combine all candidates into a list for iteration
candidates = [candi1]

# Trial number for organizing results
trial_num = 4

# Ensure the directories for saving models and information exist
if not os.path.exists(dir_path + f'\\model\\optimization\\trial_{trial_num}'):
    os.makedirs(dir_path + f'\\model\\optimization\\trial_{trial_num}')
if not os.path.exists(dir_path + f'\\model\\optimization\\info\\trial_{trial_num}'):
    os.makedirs(dir_path + f'\\model\\optimization\\info\\trial_{trial_num}')

# Iterate over hyperparameter candidates
for candidate in candidates:
    # Default hyperparameter settings
    structure = [512, 256, 128] # Default DNN structure
    norm = "min-max" # Default normalization method
    batch_size = 8 # Default batch size
    lr = 1e-3 # Default learning rate
    dropout = 0.1 # Default dropout rate
    optimizer = 'Adam' # Default optimizer
    default = [structure, norm, batch_size, lr, dropout, optimizer]
    
    # Iterate over the values for the current hyperparameter candidate
    for ii in range(1, len(candidate)):    
        
        # Update the current hyperparameter
        if candidate[0] == 'norm':
            norm = candidate[ii]
        elif candidate[0] == 'batch':
            batch_size = candidate[ii]
        elif candidate[0] == 'structure':
            structure = candidate[ii]
        elif candidate[0] == 'dropout':
            dropout = candidate[ii]
        elif candidate[0] == 'lr':
            lr = candidate[ii]
        
        # Load dataset and additional features if specified
        if include_count:
            X1 = np.load(dir_path +  f'\\embedded_dataset\\embedded_X_factor_{weight_factor}.npy')
            X2 = np.load(dir_path + '\\embedded_dataset\\embedded_X_count_ratio.npy')
            X = np.hstack([X1,X2]) # Combine feature datasets
        else:
            X = np.load(dir_path + f'\\embedded_dataset\\embedded_X_factor_{weight_factor}.npy')
            
        y = np.load(dir_path + '\\embedded_dataset\\embedded_Y.npy') # Load labels
        
        # Apply the selected normalization method
        if norm == 'min-max':
            X_min, X_max = X.min(axis = 0), X.max(axis = 0) 
            X = (X-X_min) / (X_max-X_min)
        elif norm == 'mean-std':
            X_mean, X_std = X.mean(axis = 0), X.std(axis = 0)
            X = (X-X_mean) / (X_std)
        elif norm == "MaxAbs":
            X_max = abs(X).max(axis = 0)
            X = X/X_max
        else:
            print("Inappropriate normalization method")
            break
        
        # Initialize K-fold cross-validation
        kfold = KFold(n_splits=k, shuffle=True, random_state=2)
        fold_results = []
        
        # Perform K-fold validation
        for fold, (train_index, test_index) in enumerate(kfold.split(X)):
            
            print(f"\nOptimization for {candidate[0]} fold {fold+1}/{k}\n")
            
            model_name = f'Optim_{candidate[0]}_{candidate[ii]}_fold{fold}'
            
            # Train/Validation
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            # Convert data to PyTorch tensors
            X_train, X_test = torch.from_numpy(X_train).float().to(device), torch.from_numpy(X_test).float().to(device)
            y_train, y_test = torch.from_numpy(y_train).float().to(device), torch.from_numpy(y_test).float().to(device)
    
            # Create datasets and data loaders
            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(test_dataset, batch_size)
            
            # Initialize the model, loss function, and optimizer
            model = DNN_structure(input_size = X.shape[1], hidden_layers = structure, output_size=1, dropout_prob=dropout)
            criterion = torch.nn.BCEWithLogitsLoss()
            scheduler = False
            
            # Update optimizer based on the candidate
            if candidate[0] == 'optim':
                if ii == 1:
                    optimizer = optim.AdamW(model.parameters(), lr = lr)
                elif ii == 2:
                    optimizer = optim.SGD(model.parameters(), lr = lr)
                elif ii == 3:
                    optimizer = optim.RMSprop(model.parameters(), lr = lr)
                elif ii == 4:
                    optimizer = optim.Adagrad(model.parameters(), lr = lr)
            else:
                optimizer = optim.Adagrad(model.parameters(), lr = lr)
            
            
            # Train the model
            model.train(train_loader, val_loader, model, optimizer, scheduler,
                                               criterion, patience,
                                               path = dir_path + f'\\model\\optimization\\trial_{trial_num}\\{model_name}.pt',
                                               Early = Early, epoch = epoch)
            
            # Save training results
            train_info = np.zeros([epoch, 4])
            train_info[:,0], train_info[:,1], train_info[:,2], train_info[0,3] = model.train_loss, model.val_loss, model.val_acc, model.train_time
            train_info= pd.DataFrame(train_info, columns=['Train loss', 'Val loss', 'Val acc', 'train_time'])
            train_info.to_csv(dir_path + f"\\model\\optimization\\info\\trial_{trial_num}\\{model_name}_info.csv")
            
            idx = np.argmin(model.val_loss)
            fold_results.append((min(model.val_loss), model.val_acc[idx]))
            
        fold_results = np.array(fold_results)
        np.savetxt(dir_path + f'\\model\\optimization\\info\\trial_{trial_num}\\Optim_{candidate[0]}_{candidate[ii]}.csv',
                   fold_results, delimiter = ',')

# Save the default configuration
default[0] = str(default[0])
default = [default]
default_info = pd.DataFrame(default,
                            columns= ['structure', 'Normalization', 'Batch_size', 'learning rate', 'Drop out', 'Optim'])
default_info.to_csv(dir_path + f'\\model\\optimization\\trial_{trial_num}\\default_setting.csv')
