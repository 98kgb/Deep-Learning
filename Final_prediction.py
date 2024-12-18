# -*- coding: utf-8 -*-
"""
This script is designed for evaluating a deep neural network (DNN) model trained on World Cup-related tweet data.

The script loads the trained model, applies it to a new dataset of tweets, and predicts event types based on tweet embeddings.
"""
import os
import sys
import gensim.downloader as api
import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm
# Add the current file path to the system path
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path)
# Import text preprocessing and embedding functions
from Embedding import preprocess_text, get_embeddings

device = "cuda" if torch.cuda.is_available() else "cpu"

# Select a specific fold for evaluation
fold = 2
model_name = 'best' # For consistency

# Load training information for normalization parameters and other settings
train_info = pd.read_csv(f"{dir_path}/model/info/{model_name}_info.csv")
# Load the pre-trained model
model = torch.load(f"{dir_path}/model/{model_name}.pt")

#%% Embedding settings and checking final accuracy of the model

# Settings for embedding, normalization, and predictions
factor = 5
include_count = True
save = True
sub_name = model_name

# Load keywords, normalization type, and dataset
keywords = np.load(dir_path + '\\embedded_dataset\\embedded_keyword.npy')
norm = train_info['Norm'][0]

# Load the input dataset (X) and labels (y)
if include_count:
    X1 = np.load(dir_path + f'\\embedded_dataset\\embedded_X_factor_{factor}.npy')
    X2 = np.load(dir_path + '\\embedded_dataset\\embedded_X_count_ratio.npy')
    X = np.hstack([X1, X2])
else:
    X = np.load(dir_path + f'\\embedded_dataset\\embedded_X_factor_{factor}.npy')

y = np.load(dir_path + '\\embedded_dataset\\embedded_Y.npy')

# Normalize the dataset based on the stored method
if norm == 'min-max':
    X_min = train_info['norm_param1'].values[:X.shape[1]]
    X_max = train_info['norm_param2'].values[:X.shape[1]]
    X = (X-X_min) / (X_max-X_min)
    
elif norm == 'mean-std':
    X_mean = train_info['norm_param1'].values[:X.shape[1]]
    X_std = train_info['norm_param2'].values[:X.shape[1]]
    X = (X-X_mean) / (X_std)
elif norm == "MaxAbs":
    X_max = train_info['norm_param1'].values[:X.shape[1]]
    X = X/X_max

# Convert the dataset to PyTorch tensors
X = torch.from_numpy(X).float().to(device)

# Predict using the trained model
y_pred = model(X).detach().cpu().numpy()
preds = (y_pred >= 0.5).astype(int)

# Calculate accuracy on the entire dataset
acc = accuracy_score(preds, y)
print(f"Accuracy for total dataset: {acc}\n")

#%% Predicting event types for new tweet data
predictions = []

# Load the embedding model if not already loaded
if "embeddings_model" in locals():
    print("embedding model exist.\n")
else:
    embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings

# Loop through files containing new tweet data for evaluation
for fname in tqdm(os.listdir(dir_path+"/eval_tweets"), desc = 'evaluating'):
    
    # Load and preprocess the tweet data
    val_df = pd.read_csv(dir_path+"/eval_tweets/" + fname)
    val_df['Tweet'] = val_df['Tweet'].apply(preprocess_text)
    
    vector_size = 200  # Adjust based on the chosen GloVe model
    val_tweet = val_df['Tweet']
    
    val_tweet_vectors = []
    for ii in range(int(len(val_df['Tweet']))):
        a, _ = get_embeddings(val_tweet[ii], embeddings_model, keywords, vector_size, weight_factor = factor)
        val_tweet_vectors.append(a)

    val_df['TweetVector'] = list(val_tweet_vectors)
    final_df = val_df
    
    if include_count:
        # Calculate tweet counts and ratios
        period_tweet_counts = final_df.groupby(['MatchID', 'PeriodID']).size().reset_index(name='TweetCount')
        match_tweet_counts = final_df.groupby(['MatchID']).size().reset_index(name='TotalTweetCount')
        period_tweet_counts = period_tweet_counts.merge(match_tweet_counts, on='MatchID')
        period_tweet_counts['TweetRatio'] = period_tweet_counts['TweetCount'] / period_tweet_counts['TotalTweetCount']
        
        # Add tweet count ratios to the dataset
        final_df = final_df.drop(columns=['Timestamp', 'Tweet'])
        final_df = final_df.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
        final_df['TweetCount'] = period_tweet_counts['TweetRatio'].values
                
        X = final_df.drop(columns=['MatchID', 'PeriodID', 'ID']).values
        
        # Reshape the data for input to the model
        X_1, X_2 = X[:,0], X[:,1]
        X_1, X_2 = np.array(X_1.tolist()).reshape(len(X_1), -1), np.array(X_2.tolist()).reshape(len(X_2), -1)
        X = np.hstack([X_1,X_2])
        
    else:
        final_df = final_df.drop(columns=['Timestamp', 'Tweet'])
        final_df = final_df.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
        X = final_df.drop(columns=['MatchID', 'PeriodID', 'ID']).values
        X = np.array(X.tolist()).reshape(len(X), -1)
    
    if norm == 'min-max':
        X_min = train_info['norm_param1'].values[:X.shape[1]]
        X_max = train_info['norm_param2'].values[:X.shape[1]]
        X = (X-X_min) / (X_max-X_min)
        
    elif norm == 'mean-std':
        X_mean = train_info['norm_param1'].values[:X.shape[1]]
        X_std = train_info['norm_param2'].values[:X.shape[1]]
        X = (X-X_mean) / (X_std)
    elif norm == "MaxAbs":
        X_max = train_info['norm_param1'].values[:X.shape[1]]
        X = X/X_max
    
    # Predict event types
    preds = model(torch.from_numpy(X).float().to(device)).detach().numpy()
    preds = (preds >= 0.5).astype(int)
    final_df['EventType'] = preds
    predictions.append(final_df[['ID', 'EventType']])

# Concatenate predictions for all files
pred_df = pd.concat(predictions)
#%%
# Save predictions to a CSV file if specified
if save:
    # make sure the presence of the directory
    if not os.path.exists(f'{dir_path}\\predictions'):
        os.makedirs(f'{dir_path}\\predictions')
    pred_df.to_csv(dir_path+'\\predictions\\DNN_JIBAEK_{}.csv'.format(sub_name), index=False)
