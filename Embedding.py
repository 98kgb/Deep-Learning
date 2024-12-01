# -*- coding: utf-8 -*-
"""
This code is for embedding

@author: Gibaek
"""
import os
import sys

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
#%%

def get_embeddings(tweet, model, keywords, vector_size=200, weight = True, weight_factor = 2):
    words = tweet.split()  # Tokenize by whitespace
    # keywords = ['full time', 'goal', 'half time', 'kick off', 'other', 'owngoal', 'penalty', 'red card', 'yellow card']
    keywords = keywords
    word_vectors = []
    weight_factor = weight_factor
    for word in words:
        if word in model:
            if word in keywords and weight:
                # print("{} is weighted".format(word))
                word_vectors.extend([model[word]] * weight_factor)

            else:
                word_vectors.append(model[word])
    
    if not word_vectors:  # If no words in the tweet are in the vocabulary, return a zero vector
        return np.zeros(vector_size), np.zeros(vector_size)
    
    return np.mean(word_vectors, axis=0), word_vectors

def analysis(tweet, EventType):
    event_tweet = []
    normal_tweet = []
    event_counter = Counter()  
    normal_counter = Counter()
    for ii in tqdm(range(tweet.shape[0]), desc = "Analyzing..."):
        if EventType[ii] == 1:
            event_tweet.append([tweet[ii]])  # store meaningful tweet
            words = tweet[ii].split()  # convert tweet to words
            event_counter.update(words)  # Add words to counter
        
        else:
            normal_tweet.append([tweet[ii]])
            words = tweet[ii].split()  # convert tweet to words
            normal_counter.update(words)  # Add words to counter
            
    event_tweet = np.array(event_tweet)
    normal_tweet = np.array(normal_tweet)
    
    return event_tweet, normal_tweet, event_counter, normal_counter
    
def preprocess_text(text):
    # Lowercasing
    text = text.lower()
    # Remove 'rt' specifically
    text = re.sub(r'\brt\b', '', text)
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|@\S+|#\S+', '', text)
    # Tokenization
    words = text.split()
    # To eliminate unessasary words (a, an, the, is, ...)
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    # Lemmatization convert the words to basic form (ex. better=> good)
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(words)

if __name__ == "__main__":
    include_count = True
    weight = True
    weight_factor = 5
    embed_method = 1
    
    # Checking directory
    save_dir = dir_path+'\\embedded_dataset'
    if os.path.isdir(save_dir):
        print("The directory exists.")
    else:
        os.makedirs(dir_path+'\\embedded_dataset')
    
    # Load GloVe model with Gensim's API
    if "embeddings_model" in locals():
        print("embedding model already exist.\n")
    else:
        temp = time.time()
        embeddings_model = api.load("glove-twitter-200")  # 200-dimensional GloVe embeddings
        print("\nEmbedding model loading complete in {}\n".format(np.round(time.time()-temp)))
    
    
    start_time = time.time()
    # Load data
    li = []
    for filename in tqdm(os.listdir(dir_path+ "\\train_tweets"), desc = "Data loading"):
        df = pd.read_csv(dir_path+"/train_tweets/" + filename)
        li.append(df)
        
    df = pd.concat(li, ignore_index=True)
    
    # Apply preprocessing to each tweet
    temp = time.time()
    print("\nRemove useless words from the data set. It would take a few tens of minutes.\n")
    df['Tweet'] = df['Tweet'].apply(preprocess_text)
    print("Removing process ends in {} secs\n".format(np.round(time.time()-temp)))
    
    # analysis words with event type.
    event_tweet, normal_tweet, event_counter, normal_counter = analysis(df['Tweet'], df['EventType'])
    
    country_names = {country.name.lower() for country in pycountry.countries}
    country_names = {country.name.lower() for country in pycountry.countries} 
    country_alpha_2 = {country.alpha_2.lower() for country in pycountry.countries}  
    country_alpha_3 = {country.alpha_3.lower() for country in pycountry.countries}  
    country_names =  country_names | country_alpha_2 | country_alpha_3  
    country_names.update(['alg', 'ned', 'chi', 'ger', 'por'])
    
    print("\nWord distribution in event tweets:")
    for word, count in event_counter.most_common(20):
        print(f"{word}: {count}")
    
    print("\nWord distribution in normal tweets:")
    
    for word, count in normal_counter.most_common(20):
        print(f"{word}: {count}")
    
    event = event_counter.most_common(50)
    normal = normal_counter.most_common(50)
    # Convert to sets of keywords
    event_keywords = set([word for word, _ in event])
    normal_keywords = set([word for word, _ in normal])
    country_names = set(country_names)
    # Find keywords that are in event but not in normal
    
    if embed_method == 1 or embed_method == 3:
        unique_event_keywords = event_keywords - country_names
    elif embed_method == 2:
        unique_event_keywords = event_keywords - normal_keywords - country_names
    
    # Print the unique keywords
    keywords = []
    
    for keyword in unique_event_keywords:
        keywords.append(keyword)

    # Get embedding vectors
    vector_size = 200  # Adjust based on the chosen GloVe model
    tweet = df['Tweet']
    tweet_vectors = []
    
    for ii in tqdm(range(int(len(df['Tweet']))), desc = 'Embeding process'):
        a, b = get_embeddings(tweet[ii], embeddings_model, keywords,vector_size, weight, weight_factor)
        tweet_vectors.append(a)
    #%%
    final_df = df
    final_df['TweetVector'] = list(tweet_vectors)
    
    # Group by MatchID and PeriodID to create sequences of tweet vectors for each period
    if include_count:
        # Step 1: Count tweets for each MatchID and PeriodID
        period_tweet_counts = final_df.groupby(['MatchID', 'PeriodID']).size().reset_index(name='TweetCount')
        # Step 2: Count total tweets for each MatchID
        match_tweet_counts = final_df.groupby(['MatchID']).size().reset_index(name='TotalTweetCount')
        # Step 3: Merge the two DataFrames to include TotalTweetCount for each MatchID
        period_tweet_counts = period_tweet_counts.merge(match_tweet_counts, on='MatchID')
        # Step 4: Calculate the ratio of PeriodID TweetCount to MatchID TotalTweetCount
        period_tweet_counts['TweetRatio'] = period_tweet_counts['TweetCount'] / period_tweet_counts['TotalTweetCount']

    final_df = final_df.drop(columns=['Timestamp', 'Tweet'])
    final_df = final_df.groupby(['MatchID', 'PeriodID', 'ID']).mean().reset_index()

    if include_count:
        final_df['TweetCount'] = period_tweet_counts['TweetRatio'].values
    #%%
    
    # Create X (input sequences) and y (labels)
    X = final_df.drop(columns=['EventType', 'MatchID', 'PeriodID', 'ID']).values
    y = final_df['EventType'].values
    #%%
    if include_count:
        X_1, X_2 = X[:,0], X[:,1]
        X_1, X_2 = np.array(X_1.tolist()).reshape(len(X_1), -1), np.array(X_2.tolist()).reshape(len(X_2), -1)
        X = np.hstack([X_1,X_2])
    else:
        X = np.array(X.tolist()).reshape(len(X), -1)
    
    np.save(dir_path + f'\\embedded_dataset\\embedded_{embed_method}_X_weighted_{weight}_factor_{weight_factor}_include_count_{include_count}.npy', X)
    np.save(dir_path + f'\\embedded_dataset\\embedded_{embed_method}_Y_weighted_{weight}_factor_{weight_factor}_include_count_{include_count}.npy', y)
    np.save(dir_path + f'\\embedded_dataset\\embedded_{embed_method}_keyword.npy', keywords)
    
    print("\nEncoding ends in {}".format(np.round(time.time()-start_time)))
    