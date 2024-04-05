import torch
import csv
import numpy as np
from torch.utils.data import Dataset

class SentimentDataset(Dataset):
    
    def __init__(self, csv_path, training_set=True):
        tweets = []
        labels = []
        with open(csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                tweets.append(row[0])
                labels.append(int(row[1]))
        
        total_tweets = len(tweets)
        train_indices = np.random.choice(total_tweets, int(0.8 * total_tweets), replace=False)
        test_indices = np.setdiff1d(np.arange(total_tweets), train_indices)

        if training_set:
            self.tweets = [tweets[i] for i in train_indices]
            self.labels = [labels[i] for i in train_indices]
        else:
            self.tweets = [tweets[i] for i in test_indices]
            self.labels = [labels[i] for i in test_indices]
    
    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, idx):
        return self.tweets[idx], self.labels[idx]