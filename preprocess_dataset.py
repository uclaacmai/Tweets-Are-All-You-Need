import pandas as pd
import csv

def process_csv(input_file, output_file):

    df = pd.read_csv(input_file, encoding='latin1')
    # Keep only the target and the tweet columns
    df = df.iloc[:, [0, 5]]
    # Since we have only two labels, edit the labels so that 0 ~ negative, 1 ~ positive
    df.iloc[:, 0] = df.iloc[:, 0].replace(4, 1)
    # Switch columns so that we have tweet-label pairs
    df = df.iloc[:, [1, 0]]
    df.to_csv(output_file, index=False, quoting=csv.QUOTE_ALL)

if __name__ == '__main__':
    input_file = 'training.1600000.processed.noemoticon.csv'
    output_file = 'dataset.csv'
    process_csv(input_file, output_file)