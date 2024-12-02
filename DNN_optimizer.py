# -*- coding: utf-8 -*-
"""
This code is for optimizing DNN structure and hyperparameters

@author: Gibaek
"""
import os
import sys

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

#%% Dataset preprocessing

# Optimization hyperparameters
k = 5 # for K-fold validation
weight = True
factor = 10 # Embedding factor
save = True
Early = True
patience = 300
epoch = 300
include_count = True
# enbedded_method = 0 # with keyword stored in embedded_0_factor_.npy
enbedded_method = 1 # with keyword stored in embedded_1_factor_.npy

candi1 = ['norm', "min-max", 'mean-std', "MaxAbs"] # normalization
candi2 = ['optim', 'Adam', 'SGD', 'RMSprop', 'Adagrad'] # optimizer
candi3 = ['batch', 8, 16, 32, 64] # batch size
candi4 = ['structure', [1024, 512, 256, 128], [512, 256, 128], [512, 256], [512]] # DNN structure
candi5 = ['dropout',0, 0.1, 0.2, 0.3, 0.4, 0.5] # dropout rate
candi6 = ['lr',1e-5, 1e-4, 1e-3, 1e-2] # Learning rate

candidates = [candi1, candi2, candi3, candi4, candi5, candi6]

trial_num = 3

if os.path.exists(dir_path + f'\\model\\optimization\\trial_{trial_num}'):
    pass
else:
    os.makedirs(dir_path + f'\\model\\optimization\\trial_{trial_num}')
    print(f'{dir_path} + \\model\\optimization\\trial_{trial_num} created!!')

if os.path.exists(dir_path + f'\\model\\optimization\\info\\trial_{trial_num}'):
    pass
else:
    os.makedirs(dir_path + f'\\model\\optimization\\info\\trial_{trial_num}')
   
for candidate in candidates:
    
    structure = [512, 256, 128]
    norm = "mean-std"
    batch_size = 8 # new optimzed parameter (previous : 32, 16)
    lr = 1e-3
    dropout = 0.1
    default = [structure, norm, batch_size, lr, dropout]
    for ii in range(1, len(candidate)):    
        
        # Sweep the hyperparameter
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
        
        if include_count:
            X1 = np.load(dir_path + f'\\embedded_dataset\\embedded_{enbedded_method}_X_factor_{factor}.npy')
            X2 = np.load(dir_path + '\\embedded_dataset\\embedded_X_count_ratio.npy')
            X = np.hstack([X1,X2])
        else:
            X = np.load(dir_path + f'\\embedded_dataset\\embedded_{enbedded_method}_X_factor_{factor}.npy')
            
        y = np.load(dir_path + '\\embedded_dataset\\embedded_Y.npy')
        
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
        
        kfold = KFold(n_splits=k, shuffle=True, random_state=2)
    
        # K-Fold Validation
        fold_results = []
        
        for fold, (train_index, test_index) in enumerate(kfold.split(X)):
            
            print(f"\nOptimization for {candidate[0]} fold {fold+1}/{k}\n")
            
            model_name = f'Optim_{candidate[0]}_{candidate[ii]}_fold{fold}'
            
            # Train/Validation
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            X_train, X_test = torch.from_numpy(X_train).float().to(device), torch.from_numpy(X_test).float().to(device)
            y_train, y_test = torch.from_numpy(y_train).float().to(device), torch.from_numpy(y_test).float().to(device)
    
            # convert it the tensor dataset
            train_dataset = TensorDataset(X_train, y_train)
            test_dataset = TensorDataset(X_test, y_test)
                
            # Define Batch size and DataLoader
            train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
            val_loader = DataLoader(test_dataset, batch_size)
            
            # itialize model, loss function and optimizer
            model = DNN_structure(input_size = X.shape[1], hidden_layers = structure, output_size=1, dropout_prob=dropout)
            criterion = torch.nn.BCEWithLogitsLoss()
            scheduler = False
            
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
                optimizer = optim.AdamW(model.parameters(), lr = lr)
            
            
            # train
            
            model.train(train_loader, val_loader, model, optimizer, scheduler,
                                               criterion, patience,
                                               path = dir_path + f'\\model\\optimization\\trial_{trial_num}\\{model_name}.pt',
                                               Early = Early, epoch = epoch)
            
            # save the train results
            train_info = np.zeros([epoch, 4])
            train_info[:,0], train_info[:,1], train_info[:,2], train_info[0,3] = model.train_loss, model.val_loss, model.val_acc, model.train_time
            train_info= pd.DataFrame(train_info, columns=['Train loss', 'Val loss', 'Val acc', 'train_time'])
            train_info.to_csv(dir_path + f"\\model\\optimization\\info\\trial_{trial_num}\\{model_name}_info.csv")
            
            idx = np.argmin(model.val_loss)
            fold_results.append((min(model.val_loss), model.val_acc[idx]))
            
        fold_results = np.array(fold_results)
        np.savetxt(dir_path + f'\\model\\optimization\\info\\trial_{trial_num}\\Optim_{candidate[0]}_{candidate[ii]}.csv',
                   fold_results, delimiter = ',')

#%% Default 
default[0] = str(default[0])
default = [default]
default_info = pd.DataFrame(default, columns= ['structure', 'Normalization', 'Batch_size', 'learning rate', 'Drop out'])
default_info.to_csv(dir_path + f'\\model\\optimization\\trial_{trial_num}\\default_setting.csv')
