# -*- coding: utf-8 -*-
"""
This module defines the fundamental structure and functions for a deep neural network (DNN),

including model training and early stopping.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# Define the DNN structure
class DNN_structure(nn.Module):
    def __init__(self, input_size, hidden_layers, output_size, dropout_prob = 0):
        """
        Initialize a deep neural network.

        Args:
            input_size (int): Number of input features.
            hidden_layers (list of int): Number of neurons for each hidden layer (e.g., [64, 128, 64]).
            output_size (int): Number of output features.
            dropout_prob (float): Dropout probability for regularization.
        """
        super(DNN_structure, self).__init__()
        
        # list for storing network layers
        layers = []
        
        # construct DNN structure
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size)) # Fully connected layer
            layers.append(nn.LeakyReLU())  # Activation function (LeakyReLU)
            layers.append(nn.Dropout(dropout_prob)) # Dropout layer for regularization
            prev_size = hidden_size
        
        # Add the final output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        # Create the sequential model
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (Tensor): Input data.

        Returns:
            Tensor: Output of the network.
        """
        return self.network(x)
    
    def train(self, train_dataloader, val_dataloader, model, optimizer, scheduler, criterion, patience,
              path, Early = False, epoch = 2500, verbose = True):
        
        """
        Train the model with early stopping.

        Args:
            train_dataloader (DataLoader): Training data loader.
            val_dataloader (DataLoader): Validation data loader.
            model (nn.Module): The model to be trained.
            optimizer (torch.optim.Optimizer): Optimizer for weight updates.
            scheduler (torch.optim.lr_scheduler): Learning rate scheduler.
            criterion (nn.Module): Loss function.
            patience (int): Number of epochs to wait before stopping if no improvement.
            path (str): Path to save the model.
            Early (bool): Whether to apply early stopping.
            epoch (int): Maximum number of training epochs.
            verbose (bool): Whether to print progress messages.
        """
        
        if Early:
            # Initialize early stopping mechanism
            early_stopping = EarlyStopping(patience = patience,
                                           verbose = True, path = path)
        # Track training start time
        start = time.time()
        
        # Initialize arrays for tracking losses and accuracy
        train_loss = torch.zeros(epoch).to(device)
        val_loss = torch.zeros(epoch).to(device)
        val_acc = np.zeros(epoch)
        
        print("========       Iterations started       ========")
        for ii in range(epoch):
            num_batch = 0
            
            # Training loop
            for X_train, Y_train in tqdm(train_dataloader, desc = 'Epoch {}/{}'.format(ii+1, epoch)):
                optimizer.zero_grad() # Reset gradients
                
                # Forward pass
                Y_hat = model(X_train)
                loss = criterion(Y_hat.squeeze(), Y_train.squeeze())                
                
                # Update training loss
                train_loss[ii] += loss
               
                # Backward and optimize
                loss.backward()
                optimizer.step()
                num_batch += 1
                
            # Update the scheduler, if applicable
            if scheduler:
                scheduler.step()
            
            # Compute average training loss for the epoch
            train_loss[ii] = train_loss[ii]/num_batch
            
            # Validation loop
            with torch.no_grad(): # No gradients for validation
                num_batch = 0
                for X_val, Y_val in val_dataloader:
                    val_hat = model(X_val)
                    
                    # Compute validation loss
                    loss = criterion(val_hat.squeeze(), Y_val.squeeze())
                    val_loss[ii] += loss
                    
                    # Compute validation accuracy
                    y_pred = (val_hat >= 0.5).int()
                    val_acc[ii] += accuracy_score(Y_val.detach().cpu().numpy(), y_pred)
                    num_batch += 1
                
                # Compute average validation loss and accuracy for the epoch
                val_loss[ii] = val_loss[ii]/num_batch
                val_acc[ii] = val_acc[ii]/num_batch
                
            # Early stopping check
            if Early:
                early_stopping(val_loss[ii], model)
                if early_stopping.early_stop:
                    print("Early stopping at epoch {}".format(ii + 1))
                    self.epoch_stop = ii+1
                    break
            elif ii == epoch-1:
                # Save the model if early stopping is not enabled
                torch.save(model, path)
                
            # Save the last epoch if early stopping is not triggered
            if ii == epoch-1:
                self.epoch_stop = ii+1
                
            # Print training progress
            print(f"Loss_train: {train_loss[ii]:.6f} Loss_val: {val_loss[ii]:.6f} Acc_val: {val_acc[ii]:.3f}")
            
        print("==========     Gain Trained model saved     ==========\n")
        print("Training time: {:.6f} sec".format(time.time()-start))
        
        # Save training statistics
        self.train_loss = train_loss.cpu().detach().numpy()
        self.val_loss = val_loss.cpu().detach().numpy()
        self.val_acc = val_acc
        self.train_time = time.time()-start

#%% Early stopping implementation

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, path='.//model//checkpoint.pt'):
        """
        Initialize early stopping.

        Args:
            patience (int): Number of epochs to wait before stopping if no improvement.
            verbose (bool): Whether to print detailed messages.
            path (str): Path to save the best model checkpoint.
        """
        self.patience = patience
        self.verbose = verbose
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):
        """
        Check whether the validation loss has improved. If not, increment the counter.

        Args:
            val_loss (float): Current validation loss.
            model (nn.Module): Current model to be evaluated.
        """
        if self.best_score is None:
            self.best_score = val_loss # Set initial best score
        elif val_loss > self.best_score:
            # Validation loss did not improve
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                # Stop training if patience is exceeded
                self.early_stop = True
        else:
            # Validation loss improved, save the model
            self.best_score = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """
        Save the model when validation loss improves.

        Args:
            val_loss (float): Current validation loss.
            model (nn.Module): Current model to be saved.
        """
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        torch.save(model, self.path) # Save the model to the specified path
        self.val_loss_min = val_loss # Update the minimum validation loss