import argparse
import ast
import os
import time
from scipy.stats import describe

import numpy as np
import pandas as pd
from tqdm import tqdm
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from tslearn.utils import to_time_series_dataset, save_time_series_txt


def generate_ts_data(texts):
    vader_model = SentimentIntensityAnalyzer()
    ts = []
    ts_global_features = []
    for text in tqdm(texts):
        t = list(map(
            lambda x: vader_model.polarity_scores(x)['compound'], text))
        ts.append(t)
        t_stats = list(describe(t))[1:]
        t_stats = list(t_stats[0]) + t_stats[1:]        
        ts_global_features.append(t_stats)
    ts = to_time_series_dataset(ts)
    ts_global_features = np.array(ts_global_features)
    return (ts_global_features, ts)

def write_files(directory_path, ts_global_features, ts, y):
    sub_dir_path = f'{directory_path}/sentiment' 
    os.makedirs(sub_dir_path, exist_ok=True)    
    with open(f'{sub_dir_path}/X.npy', 'wb') as f:
        np.save(f, ts_global_features, allow_pickle=True)    
    save_time_series_txt(f'{sub_dir_path}/ts.txt', ts)
    with open(f'{sub_dir_path}/y_class.npy', 'wb') as f:
        np.save(f, y, allow_pickle=True)
    
def write_execution_time(start_time, end_time, n_texts, directory_path):
    execution_time = end_time - start_time
    with open('execution_time.txt', 'a') as f:
        f.write('Script: {}\n'.format(__file__))
        f.write('Texts processed: {}\n'.format(n_texts))
        f.write('Execution time: {:.2f} seconds, {:.2f} minutes, {:.2f} hours\n'
        	.format(execution_time, execution_time/60, execution_time/3600)) 
        f.write('Directory path: {}\n'.format(directory_path))
        f.write('-' * 50 + '\n')

def main(file_path):
    start_time = time.time()
    df = pd.read_csv(file_path)
    directory_path = os.path.dirname(file_path)
    texts = df['text'].apply(ast.literal_eval)
    y = np.array(df['class'])
    ts_global_features, ts = generate_ts_data(texts)
    end_time = time.time()
    write_files(directory_path, ts_global_features, ts, y)
    write_execution_time(start_time, end_time, len(texts), directory_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process text data.')
    parser.add_argument('file_path', type=str, help='path to the input CSV file')
    args = parser.parse_args()
    main(args.file_path)