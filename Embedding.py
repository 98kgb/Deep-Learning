# -*- coding: utf-8 -*-
"""
This code is for embedding tweets using pre-trained GloVe embeddings.

It preprocesses text data, analyzes it based on event types, and computes embedding vectors.
"""
import os
import sys
# Add the current file path to the system path
dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path)
import time
import re
import pycountry
import numpy as np
import pandas as pd
import nltk
from tqdm import tqdm
from collections import Counter
from nltk.corpus import stopwords 
from nltk.stem import WordNetLemmatizer
import gensim.downloader as api

# Ensure required NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Define functions for embedding and preprocessing

def get_embeddings(tweet, model, keywords, vector_size=200, weight_factor = 2):
    """
    Convert a tweet into an embedding vector by averaging word embeddings.
    
    Parameters:
        tweet (str): Text of the tweet
        model: GloVe pre-trained embedding model
        keywords (list): List of keywords
        vector_size (int): Dimensionality of embedding
        weight_factor (float): Weight applied to keywords
        
    Returns:
        tuple: Average embedding vector, list of individual word vectors
    """
    
    words = tweet.split()  # Tokenize by whitespace
    keywords = keywords
    word_vectors = []
    weight_factor = weight_factor
    for word in words:
        if word in model:
            # Apply extra weight to keywords
            if word in keywords:
                word_vectors.extend([model[word]] * weight_factor)

            else:
                word_vectors.append(model[word])
    
    if not word_vectors:  # If no valid words, return zero vector
        return np.zeros(vector_size), np.zeros(vector_size)
    
    return np.mean(word_vectors, axis=0), word_vectors

def analysis(tweet, EventType):
    
    """
    Separate tweets based on event type and count word frequencies.
    
    Parameters:
        tweet (Series): Series of tweet texts
        EventType (Series): Series indicating event type (1 for event, 0 for normal)
        
    Returns:
        tuple: Event tweets, and their word counter
    """
    
    event_tweet = []
    event_counter = Counter()  
    
    for ii in tqdm(range(tweet.shape[0]), desc = "Analyzing..."):
        if EventType[ii] == 1:
            event_tweet.append([tweet[ii]])  # store event tweet
            words = tweet[ii].split()  
            event_counter.update(words)  # Add words in event tweet to counter
            
    event_tweet = np.array(event_tweet)
    
    return event_tweet, event_counter
    
def preprocess_text(text):
    """
    Preprocess text data: lowercasing, removing stopwords, punctuation, and lemmatization.
    
    Parameters:
        text (str): Input text
    
    Returns:
        str: Cleaned and preprocessed text
    """
    text = text.lower() # Lowercasing
    text = re.sub(r'\brt\b', '', text) # Remove 'rt' specifically
    text = re.sub(r'[^\w\s]', '', text) # Remove punctuation
    text = re.sub(r'\d+', '', text) # Remove numbers
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', '', text) # # Remove URLs, mentions, hashtags
    words = text.split() # Tokenize
    stop_words = set(stopwords.words('english')) # Tokenize
    words = [word for word in words if word not in stop_words] # Remove stop words
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words] # Lemmatize words
    
    return ' '.join(words)

#%% Main script
if __name__ == "__main__":
    # Configuration settings
    include_count = True
    weight_factor = 5
    verbose = False
    
    # Ensure the save directory exists
    save_dir = dir_path+'\\embedded_dataset'
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    
    # Load GloVe model with Gensim's API
    if "embeddings_model" in locals():
        print("Embedding model already loaded.\n")
    else:
        temp = time.time()
        embeddings_model = api.load("glove-twitter-200")  # Load 200-dimensional GloVe model
        print(f"\nEmbedding model loaded in {np.round(time.time() - temp)} seconds\n")
    
    # Load tweet data
    start_time = time.time()
    li = []
    for filename in tqdm(os.listdir(dir_path+ "\\train_tweets"), desc = "Data loading"):
        df = pd.read_csv(dir_path+"\\train_tweets\\" + filename)
        li.append(df)
    df = pd.concat(li, ignore_index=True)
    
    # Preprocess tweets
    temp = time.time()
    print("\nPreprocessing tweets...  It would take about half an hour.\n")
    df['Tweet'] = df['Tweet'].apply(preprocess_text)
    print(f"Preprocessing completed in {np.round(time.time() - temp)} seconds\n")
    
    # Analyze tweets based on event type
    event_tweet, event_counter = analysis(df['Tweet'], df['EventType'])
    
    # Collect country-related terms
    country_names = {country.name.lower() for country in pycountry.countries}
    country_names = {country.name.lower() for country in pycountry.countries} 
    country_alpha_2 = {country.alpha_2.lower() for country in pycountry.countries}  
    country_alpha_3 = {country.alpha_3.lower() for country in pycountry.countries}  
    country_names =  country_names | country_alpha_2 | country_alpha_3  
    country_names.update(['alg', 'ned', 'chi', 'ger', 'por'])
    
    # Display event word distributions
    print("\nWord distribution in event tweets:")
    event_words, event_counts = [], []
    for word, count in event_counter.most_common(20):
        if verbose:
            print(f"{word}: {count}")
        event_words.append(word)
        event_counts.append(count)
    np.save(f"{dir_path}\\embedded_dataset\\event_words.npy", np.array(event_words))
    np.save(f"{dir_path}\\embedded_dataset\\event_counts.npy",np.array(event_counts))
    
    # Remove country-related words and save results
    print("\nFiltered word distribution:")
    trim_words, trim_counts = [], []
    for word, count in event_counter.most_common(50):
        if word in country_names:
            pass
        else:
            if verbose:
                print(f"{word}: {count}")
            trim_words.append(word)
            trim_counts.append(count)
    np.save(f"{dir_path}\\embedded_dataset\\trim_words.npy",np.array(trim_words))
    np.save(f"{dir_path}\\embedded_dataset\\trim_counts.npy", np.array(trim_counts))
    
    # Determine unique keywords
    event = event_counter.most_common(50)
    event_keywords = set([word for word, _ in event])
    country_names = set(country_names)
    unique_event_keywords = event_keywords - country_names
    keywords = []
    for keyword in unique_event_keywords:
        keywords.append(keyword)
    
    # Generate embeddings for tweets
    vector_size = 200
    tweet = df['Tweet']
    tweet_vectors = []
    for ii in tqdm(range(int(len(df['Tweet']))), desc = 'Embeding process'):
        a, _ = get_embeddings(tweet[ii], embeddings_model, keywords,vector_size, weight_factor)
        tweet_vectors.append(a)
    
    # Add tweet vectors to dataframe
    final_df = df
    final_df['TweetVector'] = list(tweet_vectors)
    
    # Group by MatchID and PeriodID
    if include_count:
        # Step 1: Count tweets for each MatchID and PeriodID
        period_tweet_counts = final_df.groupby(['MatchID', 'PeriodID']).size().reset_index(name='TweetCount')
        # Step 2: Count total tweets for each MatchID
        match_tweet_counts = final_df.groupby(['MatchID']).size().reset_index(name='TotalTweetCount')
        # Step 3: Merge the two DataFrames to include TotalTweetCount for each MatchID
        period_tweet_counts = period_tweet_counts.merge(match_tweet_counts, on='MatchID')
        # Step 4: Calculate the ratio between them
        period_tweet_counts['TweetRatio'] = period_tweet_counts['TweetCount'] / period_tweet_counts['TotalTweetCount']
        # Save Tweet ratio
        count_ratio = period_tweet_counts['TweetRatio'].values
        count_ratio = np.array(count_ratio.tolist()).reshape(len(count_ratio), -1)
        np.save(dir_path + '\\embedded_dataset\\embedded_X_count_ratio.npy', count_ratio)
    
    # remove useless features (timestamp and tweet) and sort dataframe with IDs
    final_df = final_df.drop(columns=['Timestamp', 'Tweet']).groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()
    
    # Create X (input sequences) and y (labels)
    X = final_df.drop(columns=['EventType', 'MatchID', 'PeriodID', 'ID']).values
    X = np.array(X.tolist()).reshape(len(X), -1)
    y = final_df['EventType'].values
    
    # Save features, labels, keywords
    np.save(dir_path + f'\\embedded_dataset\\embedded_X_factor_{weight_factor}.npy', X)
    np.save(dir_path + '\\embedded_dataset\\embedded_Y.npy', y)
    np.save(dir_path + '\\embedded_dataset\\embedded_keyword.npy',keywords)
    
    print("\nEncoding ends in {}".format(np.round(time.time() - start_time)))
    
